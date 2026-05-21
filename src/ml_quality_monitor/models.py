"""Pydantic-схемы для входных данных API и для ответов.

Каждый класс — это «форма» данных: Pydantic автоматически
валидирует типы и ограничения при создании объекта.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator


def utc_now() -> datetime:
    """Возвращает текущее время в UTC — используется как значение по умолчанию для datetime-полей."""
    return datetime.now(timezone.utc)


class PredictionIn(BaseModel):
    """Входные данные одного предсказания от ML-модели.

    Отправляется на POST /predictions сразу после того,
    как модель выдала результат.
    """
    # extra='forbid' — если в запросе будут лишние поля, вернём ошибку 422
    model_config = ConfigDict(extra='forbid')

    prediction_id: str = Field(..., min_length=1, max_length=128)  # уникальный ID предсказания
    model_name: str = Field(..., min_length=1, max_length=128)      # название модели, например "fraud-detector-v1"
    y_pred: int | str | float | bool                                 # само предсказание (любой скалярный тип)
    predicted_at: datetime = Field(default_factory=utc_now)         # когда было получено предсказание
    latency_ms: float | None = Field(default=None, ge=0)            # время инференса в миллисекундах (опционально)
    metadata: dict[str, Any] = Field(default_factory=dict)          # произвольные дополнительные данные

    @field_validator('predicted_at')
    @classmethod
    def ensure_tz(cls, value: datetime) -> datetime:
        """Гарантирует, что дата всегда хранится в UTC.

        Если пришёл datetime без timezone — добавляем UTC.
        Если пришёл с другим timezone — конвертируем в UTC.
        """
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


class ActualIn(BaseModel):
    """Входные данные фактического (правильного) ответа.

    Отправляется на POST /actuals позже, когда стало известно,
    что на самом деле произошло. Связывается с предсказанием
    через prediction_id.
    """
    model_config = ConfigDict(extra='forbid')

    prediction_id: str = Field(..., min_length=1, max_length=128)  # должен совпадать с ID из PredictionIn
    y_true: int | str | float | bool                                # правильный ответ
    arrived_at: datetime = Field(default_factory=utc_now)           # когда стал известен правильный ответ

    @field_validator('arrived_at')
    @classmethod
    def ensure_tz(cls, value: datetime) -> datetime:
        """Аналогично PredictionIn — приводим к UTC."""
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


class MetricPoint(BaseModel):
    """Срез метрик качества за одно временное окно.

    Возвращается в ответах GET /metrics/{model_name}
    и POST /aggregate.
    """
    model_name: str
    window_start: datetime   # начало временного окна
    window_end: datetime     # конец временного окна
    samples: int             # количество пар (предсказание + факт) в этом окне
    accuracy: float | None   # доля правильных предсказаний; None если нет данных
    precision: float | None  # точность (micro-averaged по всем классам)
    recall: float | None     # полнота (micro-averaged)
    f1: float | None         # F1 = гармоническое среднее precision и recall
    latency_p50_ms: float | None = None   # медиана времени инференса
    latency_p95_ms: float | None = None   # 95-й перцентиль времени инференса
    latency_p99_ms: float | None = None   # 99-й перцентиль времени инференса
    created_at: datetime


class AlertStatus(BaseModel):
    """Состояние одной тревоги в очереди доставки.

    Возвращается в ответах GET /alerts.
    """
    id: int
    model_name: str
    metric_name: str          # метрика, которая упала (например "accuracy")
    metric_value: float       # фактическое значение метрики
    threshold: float          # порог, ниже которого сработала тревога
    status: Literal['pending', 'delivered', 'failed']  # текущий статус доставки
    attempts: int             # сколько попыток доставки было сделано
    created_at: datetime
    last_error: str | None = None  # текст последней ошибки при доставке
