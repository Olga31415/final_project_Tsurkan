"""Слой хранилища данных: SQLite-база с четырьмя таблицами.

Таблицы:
    predictions    — предсказания от ML-моделей
    actuals        — фактические (правильные) ответы
    metric_history — история рассчитанных метрик качества
    alert_outbox   — очередь тревог для надёжной доставки

Все datetime хранятся как строки ISO-8601 в UTC.
Все значения predictions/actuals хранятся как JSON-строки,
чтобы поддерживать int, float, str и bool без отдельных колонок.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Iterator

# версия схемы БД — увеличивать при изменении структуры таблиц
SCHEMA_VERSION = 1


def to_iso(dt: datetime) -> str:
    """Конвертирует datetime в строку ISO-8601 в UTC для хранения в SQLite."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def from_iso(value: str) -> datetime:
    """Восстанавливает datetime из строки ISO-8601, приводя к UTC.

    Заменяет суффикс 'Z' на '+00:00', так как Python < 3.11
    не понимает 'Z' в fromisoformat.
    """
    return datetime.fromisoformat(value.replace('Z', '+00:00')).astimezone(timezone.utc)


class SQLiteStore:
    """Все операции с базой данных: создание схемы, запись, чтение.

    Каждый метод открывает и закрывает соединение самостоятельно —
    это безопасно для многопоточного доступа из фоновых воркеров.
    """

    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Контекстный менеджер: открывает соединение и закрывает его после блока with."""
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        # Row позволяет обращаться к колонкам по имени: row['model_name'] вместо row[1]
        conn.row_factory = sqlite3.Row
        try:
            # WAL (Write-Ahead Log) — режим журналирования: читатели не блокируют писателей
            conn.execute('PRAGMA journal_mode=WAL')
            # NORMAL — сбрасывает данные на диск реже чем FULL, но безопасно для нашего случая
            conn.execute('PRAGMA synchronous=NORMAL')
            # проверяет ссылочную целостность: actuals.prediction_id → predictions.prediction_id
            conn.execute('PRAGMA foreign_keys=ON')
            yield conn
        finally:
            conn.close()

    def init_db(self) -> None:
        """Создаёт таблицы и индексы, если они ещё не существуют.

        Безопасно вызывать повторно — IF NOT EXISTS защищает от ошибок.
        """
        with self.connect() as conn:
            conn.executescript(
                '''
                -- служебная таблица для отслеживания версии схемы
                CREATE TABLE IF NOT EXISTS schema_meta (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                );

                -- предсказания от ML-моделей
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    y_pred TEXT NOT NULL,           -- JSON: int/float/str/bool
                    predicted_at TEXT NOT NULL,     -- ISO-8601 UTC
                    latency_ms REAL,                -- NULL если не передано
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                -- индекс по модели + времени: используется при выборке данных за окно
                CREATE INDEX IF NOT EXISTS idx_predictions_model_time
                    ON predictions(model_name, predicted_at);

                -- фактические (правильные) ответы
                CREATE TABLE IF NOT EXISTS actuals (
                    prediction_id TEXT PRIMARY KEY,
                    y_true TEXT NOT NULL,           -- JSON: int/float/str/bool
                    arrived_at TEXT NOT NULL,       -- ISO-8601 UTC
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(prediction_id) REFERENCES predictions(prediction_id)
                );
                CREATE INDEX IF NOT EXISTS idx_actuals_arrived_at ON actuals(arrived_at);

                -- история рассчитанных метрик (снимки по временным окнам)
                CREATE TABLE IF NOT EXISTS metric_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    window_start TEXT NOT NULL,
                    window_end TEXT NOT NULL,
                    samples INTEGER NOT NULL,
                    accuracy REAL,
                    precision REAL,
                    recall REAL,
                    f1 REAL,
                    latency_p50_ms REAL,
                    latency_p95_ms REAL,
                    latency_p99_ms REAL,
                    created_at TEXT NOT NULL,
                    -- уникальность: один снимок на (модель, окно)
                    UNIQUE(model_name, window_start, window_end)
                );
                CREATE INDEX IF NOT EXISTS idx_metric_history_model_window
                    ON metric_history(model_name, window_end DESC);

                -- outbox-очередь тревог: тревога сначала сохраняется, потом доставляется
                CREATE TABLE IF NOT EXISTS alert_outbox (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    threshold REAL NOT NULL,
                    payload_json TEXT NOT NULL,         -- тело HTTP-запроса или лог-сообщения
                    status TEXT NOT NULL DEFAULT 'pending',  -- pending / delivered / failed
                    attempts INTEGER NOT NULL DEFAULT 0,
                    next_attempt_at TEXT NOT NULL,      -- когда снова попробовать доставить
                    last_error TEXT,                    -- текст последней ошибки
                    created_at TEXT NOT NULL,
                    delivered_at TEXT,
                    -- дедупликация: одна тревога на (модель, метрика, окно)
                    dedup_key TEXT NOT NULL UNIQUE
                );
                -- индекс для выборки тревог, готовых к отправке
                CREATE INDEX IF NOT EXISTS idx_alert_outbox_status_next
                    ON alert_outbox(status, next_attempt_at);
                '''
            )
            conn.execute(
                'INSERT OR IGNORE INTO schema_meta(version, applied_at) VALUES(?, ?)',
                (SCHEMA_VERSION, to_iso(datetime.now(timezone.utc))),
            )

    def upsert_prediction(self, *, prediction_id: str, model_name: str, y_pred: Any,
                          predicted_at: datetime, latency_ms: float | None,
                          metadata: dict[str, Any]) -> None:
        """Сохраняет предсказание; если запись уже есть — обновляет её (upsert)."""
        now = to_iso(datetime.now(timezone.utc))
        with self.connect() as conn:
            conn.execute(
                '''
                INSERT INTO predictions(prediction_id, model_name, y_pred, predicted_at,
                                        latency_ms, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(prediction_id) DO UPDATE SET
                    model_name=excluded.model_name,
                    y_pred=excluded.y_pred,
                    predicted_at=excluded.predicted_at,
                    latency_ms=excluded.latency_ms,
                    metadata_json=excluded.metadata_json
                ''',
                (prediction_id, model_name, json.dumps(y_pred), to_iso(predicted_at),
                 latency_ms, json.dumps(metadata), now),
            )

    def upsert_actual(self, *, prediction_id: str, y_true: Any, arrived_at: datetime) -> None:
        """Сохраняет фактический ответ; если запись уже есть — обновляет её."""
        now = to_iso(datetime.now(timezone.utc))
        with self.connect() as conn:
            conn.execute(
                '''
                INSERT INTO actuals(prediction_id, y_true, arrived_at, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(prediction_id) DO UPDATE SET
                    y_true=excluded.y_true,
                    arrived_at=excluded.arrived_at
                ''',
                (prediction_id, json.dumps(y_true), to_iso(arrived_at), now),
            )

    def fetch_joined_window(self, model_name: str, start: datetime, end: datetime) -> list[sqlite3.Row]:
        """Возвращает строки JOIN(predictions, actuals) за указанное временное окно.

        Только пары, у которых есть и предсказание, и фактический ответ (INNER JOIN).
        Строки без фактического ответа не попадают в расчёт метрик.
        """
        with self.connect() as conn:
            return list(conn.execute(
                '''
                SELECT p.prediction_id, p.model_name, p.y_pred, p.predicted_at, p.latency_ms,
                       a.y_true, a.arrived_at
                FROM predictions p
                JOIN actuals a ON a.prediction_id = p.prediction_id
                WHERE p.model_name = ? AND p.predicted_at >= ? AND p.predicted_at < ?
                ORDER BY p.predicted_at ASC
                ''',
                (model_name, to_iso(start), to_iso(end)),
            ))

    def list_models(self) -> list[str]:
        """Возвращает список всех моделей, для которых есть хотя бы одно предсказание."""
        with self.connect() as conn:
            return [row['model_name'] for row in conn.execute('SELECT DISTINCT model_name FROM predictions')]

    def save_metric_point(self, point: dict[str, Any]) -> None:
        """Сохраняет срез метрик; если снимок для этого окна уже есть — перезаписывает."""
        with self.connect() as conn:
            conn.execute(
                '''
                INSERT INTO metric_history(model_name, window_start, window_end, samples,
                    accuracy, precision, recall, f1, latency_p50_ms, latency_p95_ms, latency_p99_ms, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(model_name, window_start, window_end) DO UPDATE SET
                    samples=excluded.samples,
                    accuracy=excluded.accuracy,
                    precision=excluded.precision,
                    recall=excluded.recall,
                    f1=excluded.f1,
                    latency_p50_ms=excluded.latency_p50_ms,
                    latency_p95_ms=excluded.latency_p95_ms,
                    latency_p99_ms=excluded.latency_p99_ms,
                    created_at=excluded.created_at
                ''',
                (
                    point['model_name'], to_iso(point['window_start']), to_iso(point['window_end']),
                    point['samples'], point.get('accuracy'), point.get('precision'), point.get('recall'),
                    point.get('f1'), point.get('latency_p50_ms'), point.get('latency_p95_ms'),
                    point.get('latency_p99_ms'), to_iso(point['created_at'])
                ),
            )

    def fetch_metric_history(self, model_name: str, limit: int = 100) -> list[dict[str, Any]]:
        """Возвращает историю метрик для модели, начиная с самого свежего окна."""
        with self.connect() as conn:
            rows = list(conn.execute(
                'SELECT * FROM metric_history WHERE model_name = ? ORDER BY window_end DESC LIMIT ?',
                (model_name, limit),
            ))
        return [dict(row) for row in rows]

    def enqueue_alert(self, *, model_name: str, metric_name: str, metric_value: float,
                      threshold: float, payload: dict[str, Any], dedup_key: str) -> None:
        """Добавляет тревогу в очередь доставки.

        INSERT OR IGNORE: если тревога с таким dedup_key уже есть — пропускаем.
        Это предотвращает дублирование тревог при повторном запуске агрегации.
        """
        now = to_iso(datetime.now(timezone.utc))
        with self.connect() as conn:
            conn.execute(
                '''
                INSERT OR IGNORE INTO alert_outbox(model_name, metric_name, metric_value, threshold,
                    payload_json, status, attempts, next_attempt_at, created_at, dedup_key)
                VALUES (?, ?, ?, ?, ?, 'pending', 0, ?, ?, ?)
                ''',
                (model_name, metric_name, metric_value, threshold, json.dumps(payload), now, now, dedup_key),
            )

    def fetch_due_alerts(self, now: datetime, limit: int = 50) -> list[sqlite3.Row]:
        """Возвращает тревоги, чьё время следующей попытки уже наступило."""
        with self.connect() as conn:
            return list(conn.execute(
                '''
                SELECT * FROM alert_outbox
                WHERE status = 'pending' AND next_attempt_at <= ?
                ORDER BY created_at ASC
                LIMIT ?
                ''',
                (to_iso(now), limit),
            ))

    def mark_alert_delivered(self, alert_id: int) -> None:
        """Помечает тревогу как успешно доставленную."""
        with self.connect() as conn:
            conn.execute(
                '''UPDATE alert_outbox SET status='delivered', delivered_at=? WHERE id=?''',
                (to_iso(datetime.now(timezone.utc)), alert_id),
            )

    def mark_alert_failed_attempt(self, alert_id: int, attempts: int, next_attempt_at: datetime,
                                  error: str, max_attempts: int) -> None:
        """Записывает неудачную попытку доставки.

        Если исчерпаны все попытки — статус становится 'failed'.
        Иначе остаётся 'pending' и планируется следующая попытка.
        """
        status = 'failed' if attempts >= max_attempts else 'pending'
        with self.connect() as conn:
            conn.execute(
                '''
                UPDATE alert_outbox
                SET status=?, attempts=?, next_attempt_at=?, last_error=?
                WHERE id=?
                ''',
                (status, attempts, to_iso(next_attempt_at), error[:1000], alert_id),
            )

    def fetch_alerts(self, limit: int = 100) -> list[dict[str, Any]]:
        """Возвращает все тревоги (всех статусов), начиная с самой свежей."""
        with self.connect() as conn:
            rows = list(conn.execute(
                'SELECT * FROM alert_outbox ORDER BY created_at DESC LIMIT ?',
                (limit,),
            ))
        return [dict(row) for row in rows]
