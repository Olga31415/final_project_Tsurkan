"""Система тревог: проверка порогов, очередь outbox и доставка с ретраями.

Паттерн Outbox: тревога сначала сохраняется в БД (alert_outbox),
и только потом доставляется. Это гарантирует, что тревога не потеряется
при временной недоступности получателя или перезапуске сервиса.

Классы:
    AlertSink          — интерфейс «куда отправлять тревогу»
    LoggingAlertSink   — пишет тревогу в лог (используется по умолчанию)
    WebhookAlertSink   — отправляет HTTP POST на указанный URL
    AlertManager       — проверяет пороги и кладёт тревоги в очередь
    AlertDispatcher    — забирает тревоги из очереди и доставляет их
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import logging
from typing import Protocol

import httpx

from .config import Settings
from .storage import SQLiteStore, from_iso

logger = logging.getLogger(__name__)


class AlertSink(Protocol):
    """Интерфейс для отправки тревог.

    Protocol — это «утиная типизация» в Python: любой класс с методом send()
    автоматически реализует этот интерфейс, без явного наследования.
    """
    def send(self, payload: dict) -> None: ...


class LoggingAlertSink:
    """Доставка тревог через стандартный лог приложения.

    Используется если MLQM_WEBHOOK_URL не задан.
    """
    def send(self, payload: dict) -> None:
        logger.warning('ML QUALITY ALERT: %s', json.dumps(payload, ensure_ascii=False))


class WebhookAlertSink:
    """Доставка тревог через HTTP POST на внешний URL.

    Вызывает response.raise_for_status(), поэтому HTTP 4xx/5xx
    трактуются как ошибка и попадают в механизм ретраев.
    """
    def __init__(self, url: str, timeout_seconds: float = 3):
        self.url = url
        self.timeout_seconds = timeout_seconds

    def send(self, payload: dict) -> None:
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(self.url, json=payload)
            response.raise_for_status()  # бросает исключение при HTTP >= 400


def build_sink(settings: Settings) -> AlertSink:
    """Выбирает реализацию AlertSink в зависимости от настроек."""
    if settings.webhook_url:
        return WebhookAlertSink(settings.webhook_url, settings.request_timeout_seconds)
    return LoggingAlertSink()


class AlertManager:
    """Проверяет метрики после каждой агрегации и ставит тревоги в очередь."""

    def __init__(self, store: SQLiteStore, settings: Settings):
        self.store = store
        self.settings = settings

    def check_and_enqueue(self, point: dict) -> None:
        """Проверяет accuracy и f1 против настроенных порогов.

        Если значение ниже порога — кладёт тревогу в alert_outbox.
        dedup_key гарантирует, что одна и та же тревога (модель + метрика + окно)
        не попадёт в очередь дважды.
        """
        checks = [
            ('accuracy', self.settings.accuracy_threshold),
            ('f1', self.settings.f1_threshold),
        ]
        for metric_name, threshold in checks:
            value = point.get(metric_name)
            # пропускаем: None означает недостаточно данных, >= threshold — всё в норме
            if value is None or value >= threshold:
                continue
            payload = {
                'type': 'ml_quality_degradation',
                'model_name': point['model_name'],
                'metric_name': metric_name,
                'metric_value': value,
                'threshold': threshold,
                'window_start': point['window_start'].isoformat(),
                'window_end': point['window_end'].isoformat(),
                'samples': point['samples'],
                'created_at': datetime.now(timezone.utc).isoformat(),
            }
            # ключ дедупликации: одна тревога на (модель, метрика, временное окно)
            dedup_key = f"{point['model_name']}:{metric_name}:{point['window_start'].isoformat()}:{point['window_end'].isoformat()}"
            self.store.enqueue_alert(
                model_name=point['model_name'],
                metric_name=metric_name,
                metric_value=float(value),
                threshold=threshold,
                payload=payload,
                dedup_key=dedup_key,
            )


class AlertDispatcher:
    """Забирает тревоги из очереди и доставляет их через AlertSink.

    Вызывается фоновым воркером каждые N секунд.
    При ошибке доставки применяет exponential backoff:
    следующая попытка через base * 2^(attempts-1) секунд.
    """

    def __init__(self, store: SQLiteStore, settings: Settings, sink: AlertSink | None = None):
        self.store = store
        self.settings = settings
        # sink можно подменить в тестах; в продакшне выбирается автоматически
        self.sink = sink or build_sink(settings)

    def deliver_due(self, limit: int = 50) -> int:
        """Доставляет все тревоги, чьё время следующей попытки наступило.

        Возвращает количество успешно доставленных тревог.
        """
        now = datetime.now(timezone.utc)
        delivered = 0
        for row in self.store.fetch_due_alerts(now, limit=limit):
            attempts = int(row['attempts']) + 1
            try:
                self.sink.send(json.loads(row['payload_json']))
            except Exception as exc:  # noqa: BLE001: любая ошибка уходит в ретрай
                # exponential backoff: 2с → 4с → 8с → ... → максимум 2^8 = 256с
                delay = self.settings.alert_base_backoff_seconds * (2 ** min(attempts - 1, 8))
                next_attempt_at = now + timedelta(seconds=delay)
                self.store.mark_alert_failed_attempt(
                    int(row['id']), attempts, next_attempt_at, repr(exc), self.settings.alert_max_attempts
                )
                logger.info('Alert delivery failed; will retry', exc_info=True)
                continue
            self.store.mark_alert_delivered(int(row['id']))
            delivered += 1
        return delivered
