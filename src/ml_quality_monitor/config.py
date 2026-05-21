"""Настройки приложения, считываемые из переменных окружения.

Все настройки читаются один раз при старте через os.getenv().
Если переменная не задана — используется значение по умолчанию.
Единственный экземпляр `settings` импортируется во все модули.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


def _csv_floats(value: str, default: list[float]) -> list[float]:
    """Разбирает строку вида '0.5,0.95,0.99' в список float.

    Возвращает default, если строка пустая.
    """
    if not value.strip():
        return default
    return [float(x.strip()) for x in value.split(',') if x.strip()]


@dataclass(frozen=True)  # frozen=True — объект неизменяем после создания (защита от случайного изменения)
class Settings:
    """Все настраиваемые параметры сервиса.

    Каждый параметр соответствует переменной окружения с префиксом MLQM_.
    Значения по умолчанию подходят для локального запуска.
    """
    db_path: str = os.getenv('MLQM_DB_PATH', 'ml_quality_monitor.db')
    # размер скользящего окна для агрегации метрик
    window_minutes: int = int(os.getenv('MLQM_WINDOW_MINUTES', '15'))
    # как часто фоновый воркер пересчитывает метрики
    aggregation_interval_seconds: int = int(os.getenv('MLQM_AGGREGATION_INTERVAL_SECONDS', '30'))
    # как часто фоновый воркер пытается доставить тревоги
    alert_delivery_interval_seconds: int = int(os.getenv('MLQM_ALERT_DELIVERY_INTERVAL_SECONDS', '5'))
    # максимальное число попыток доставки тревоги перед тем как пометить её как failed
    alert_max_attempts: int = int(os.getenv('MLQM_ALERT_MAX_ATTEMPTS', '12'))
    # базовая задержка для exponential backoff: попытка 1 = 2с, 2 = 4с, 3 = 8с, ...
    alert_base_backoff_seconds: float = float(os.getenv('MLQM_ALERT_BASE_BACKOFF_SECONDS', '2'))
    # порог accuracy: тревога, если значение опускается ниже
    accuracy_threshold: float = float(os.getenv('MLQM_ACCURACY_THRESHOLD', '0.9'))
    # порог F1: тревога, если значение опускается ниже
    f1_threshold: float = float(os.getenv('MLQM_F1_THRESHOLD', '0.85'))
    # перцентили latency для расчёта (заполняется в __post_init__)
    latency_quantiles: list[float] = None  # type: ignore[assignment]
    # URL для webhook-уведомлений; если не задан — тревоги пишутся только в лог
    webhook_url: str | None = os.getenv('MLQM_WEBHOOK_URL') or None
    # таймаут HTTP-запроса при отправке webhook
    request_timeout_seconds: float = float(os.getenv('MLQM_REQUEST_TIMEOUT_SECONDS', '3'))

    def __post_init__(self) -> None:
        # frozen dataclass не позволяет присваивать атрибуты напрямую,
        # поэтому используем object.__setattr__ для инициализации latency_quantiles
        object.__setattr__(
            self,
            'latency_quantiles',
            _csv_floats(os.getenv('MLQM_LATENCY_QUANTILES', '0.5,0.95,0.99'), [0.5, 0.95, 0.99]),
        )
        # создаём папку под базу данных заранее, чтобы SQLite не падал с ошибкой
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)


# единственный экземпляр настроек — создаётся один раз при импорте модуля
settings = Settings()
