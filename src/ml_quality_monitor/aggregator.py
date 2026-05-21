"""Агрегатор метрик: запускает расчёт по всем известным моделям за скользящее окно.

Вызывается фоновым воркером каждые N секунд, а также вручную
через endpoint POST /aggregate.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging

from .alerts import AlertManager
from .config import Settings
from .metrics import classification_metrics
from .storage import SQLiteStore

logger = logging.getLogger(__name__)


class MetricsAggregator:
    """Считает метрики качества за скользящее окно и проверяет пороги тревог."""

    def __init__(self, store: SQLiteStore, settings: Settings):
        self.store = store
        self.settings = settings
        # AlertManager будет проверять пороги после каждого расчёта метрик
        self.alert_manager = AlertManager(store, settings)

    def aggregate_once(self, now: datetime | None = None) -> list[dict]:
        """Выполняет один проход агрегации для всех моделей.

        Окно расчёта: [now - window_minutes, now), выровненное по минуте.
        Для каждой модели сохраняет MetricPoint и проверяет пороги тревог.

        Возвращает список словарей с рассчитанными метриками.
        """
        now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        # округляем до минуты, чтобы окна были стабильными при повторных вызовах
        window_end = now.replace(second=0, microsecond=0)
        window_start = window_end - timedelta(minutes=self.settings.window_minutes)

        points: list[dict] = []
        for model_name in self.store.list_models():
            # берём только пары (предсказание + факт) из текущего окна
            rows = self.store.fetch_joined_window(model_name, window_start, window_end)
            metrics = classification_metrics(rows)
            point = {
                'model_name': model_name,
                'window_start': window_start,
                'window_end': window_end,
                'created_at': now,
                **metrics,  # распаковываем accuracy, f1, latency и т.д.
            }
            self.store.save_metric_point(point)
            # если метрика ниже порога — тревога попадёт в очередь alert_outbox
            self.alert_manager.check_and_enqueue(point)
            points.append(point)

        logger.info('Aggregated %d model metric windows', len(points))
        return points
