"""FastAPI-приложение: HTTP API для приёма данных и чтения результатов.

Точка входа всего сервиса. При старте запускает фоновые воркеры,
при остановке — корректно их завершает.

Эндпоинты:
    GET  /health                     — проверка работоспособности
    POST /predictions                — принять предсказание от модели
    POST /actuals                    — принять фактический ответ
    POST /aggregate                  — вручную запустить расчёт метрик
    GET  /metrics/{model_name}       — история метрик для модели
    GET  /alerts                     — список тревог
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Query

from .aggregator import MetricsAggregator
from .config import settings
from .models import ActualIn, AlertStatus, MetricPoint, PredictionIn
from .storage import SQLiteStore, from_iso
from .worker import WorkerGroup

# настраиваем формат логов для всего приложения
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')

# единственный экземпляр хранилища, разделяемый между запросами и воркерами
store = SQLiteStore(settings.db_path)
workers: WorkerGroup | None = None


def _row_to_metric(row: dict[str, Any]) -> MetricPoint:
    """Преобразует строку из БД в Pydantic-модель MetricPoint."""
    return MetricPoint(
        model_name=row['model_name'],
        window_start=from_iso(row['window_start']),
        window_end=from_iso(row['window_end']),
        samples=row['samples'],
        accuracy=row['accuracy'],
        precision=row['precision'],
        recall=row['recall'],
        f1=row['f1'],
        latency_p50_ms=row['latency_p50_ms'],
        latency_p95_ms=row['latency_p95_ms'],
        latency_p99_ms=row['latency_p99_ms'],
        created_at=from_iso(row['created_at']),
    )


def _row_to_alert(row: dict[str, Any]) -> AlertStatus:
    """Преобразует строку из БД в Pydantic-модель AlertStatus."""
    return AlertStatus(
        id=row['id'],
        model_name=row['model_name'],
        metric_name=row['metric_name'],
        metric_value=row['metric_value'],
        threshold=row['threshold'],
        status=row['status'],
        attempts=row['attempts'],
        created_at=from_iso(row['created_at']),
        last_error=row.get('last_error'),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Жизненный цикл приложения: код до yield — при старте, после yield — при остановке.

    FastAPI вызывает эту функцию автоматически при запуске и завершении сервера.
    """
    global workers
    workers = WorkerGroup(store, settings)
    workers.start()   # запускаем фоновые потоки
    yield             # здесь приложение обрабатывает запросы
    if workers:
        workers.stop()  # корректно останавливаем потоки при shutdown


app = FastAPI(
    title='ML Quality Monitor',
    version='1.0.0',
    description='Service for ingesting ML predictions/actuals, computing rolling quality metrics and reliably delivering degradation alerts.',
    lifespan=lifespan,
)


@app.get('/health')
def health() -> dict[str, str]:
    """Проверка работоспособности сервиса. Используется мониторингом и load balancer'ом."""
    return {'status': 'ok'}


@app.post('/predictions', status_code=202)
def ingest_prediction(item: PredictionIn) -> dict[str, str]:
    """Принимает предсказание от ML-модели и сохраняет его в БД.

    Статус 202 Accepted означает: данные получены, но обработка (расчёт метрик)
    произойдёт асинхронно в фоновом воркере.
    """
    store.upsert_prediction(
        prediction_id=item.prediction_id,
        model_name=item.model_name,
        y_pred=item.y_pred,
        predicted_at=item.predicted_at,
        latency_ms=item.latency_ms,
        metadata=item.metadata,
    )
    return {'status': 'accepted', 'prediction_id': item.prediction_id}


@app.post('/actuals', status_code=202)
def ingest_actual(item: ActualIn) -> dict[str, str]:
    """Принимает фактический ответ и связывает его с предсказанием по prediction_id.

    Возвращает 400, если prediction_id не найден в БД (нарушение foreign key).
    """
    try:
        store.upsert_actual(prediction_id=item.prediction_id, y_true=item.y_true, arrived_at=item.arrived_at)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {'status': 'accepted', 'prediction_id': item.prediction_id}


@app.post('/aggregate', response_model=list[MetricPoint])
def aggregate_now() -> list[MetricPoint]:
    """Принудительно запускает расчёт метрик прямо сейчас.

    Используется в тестах и при демонстрации без ожидания фонового воркера.
    """
    points = MetricsAggregator(store, settings).aggregate_once(datetime.now(timezone.utc))
    return [MetricPoint(**point) for point in points]


@app.get('/metrics/{model_name}', response_model=list[MetricPoint])
def metric_history(model_name: str, limit: int = Query(100, ge=1, le=5000)) -> list[MetricPoint]:
    """Возвращает историю метрик для указанной модели (свежие окна первыми)."""
    return [_row_to_metric(row) for row in store.fetch_metric_history(model_name, limit)]


@app.get('/alerts', response_model=list[AlertStatus])
def alerts(limit: int = Query(100, ge=1, le=5000)) -> list[AlertStatus]:
    """Возвращает список всех тревог (pending, delivered, failed), свежие первыми."""
    return [_row_to_alert(row) for row in store.fetch_alerts(limit)]
