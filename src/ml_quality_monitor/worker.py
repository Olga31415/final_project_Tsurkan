"""Фоновые потоки: периодически запускают агрегацию метрик и доставку тревог.

RepeatingWorker — универсальный поток, повторяющий одну функцию с интервалом.
WorkerGroup     — группа из двух воркеров, запускается при старте FastAPI-приложения.
"""
from __future__ import annotations

import logging
import threading
import time

from .aggregator import MetricsAggregator
from .alerts import AlertDispatcher
from .config import Settings
from .storage import SQLiteStore

logger = logging.getLogger(__name__)


class RepeatingWorker:
    """Поток, который вызывает заданную функцию с заданным интервалом.

    Гарантирует, что исключение внутри функции не убьёт поток —
    ошибка логируется и поток продолжает работу.
    """

    def __init__(self, name: str, interval_seconds: float, fn):
        self.name = name
        self.interval_seconds = interval_seconds
        self.fn = fn
        # Event используется вместо time.sleep(), чтобы stop() мог разбудить поток немедленно
        self._stop = threading.Event()
        # daemon=True: поток автоматически завершится, когда завершится главный процесс
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)

    def start(self) -> None:
        """Запускает фоновый поток."""
        self._thread.start()

    def stop(self) -> None:
        """Сигнализирует потоку остановиться и ждёт завершения (до 5 секунд)."""
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        """Основной цикл потока: выполняет fn(), ждёт остаток интервала, повторяет."""
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self.fn()
            except Exception:  # noqa: BLE001 — воркер не должен падать из-за одной ошибки
                logger.exception('%s failed', self.name)
            elapsed = time.monotonic() - started
            # ждём оставшееся время интервала; минимум 0.1с чтобы не крутиться в tight loop
            self._stop.wait(max(0.1, self.interval_seconds - elapsed))


class WorkerGroup:
    """Объединяет всех фоновых воркеров приложения.

    Создаёт два потока:
        metrics-aggregator — пересчитывает метрики каждые N секунд
        alert-dispatcher   — доставляет тревоги из очереди каждые M секунд
    """

    def __init__(self, store: SQLiteStore, settings: Settings):
        aggregator = MetricsAggregator(store, settings)
        dispatcher = AlertDispatcher(store, settings)
        self.workers = [
            RepeatingWorker('metrics-aggregator', settings.aggregation_interval_seconds, aggregator.aggregate_once),
            RepeatingWorker('alert-dispatcher', settings.alert_delivery_interval_seconds, dispatcher.deliver_due),
        ]

    def start(self) -> None:
        """Запускает все воркеры."""
        for worker in self.workers:
            worker.start()

    def stop(self) -> None:
        """Останавливает все воркеры (вызывается при завершении приложения)."""
        for worker in self.workers:
            worker.stop()
