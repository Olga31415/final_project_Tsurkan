"""Вычисление метрик качества классификации.

Поддерживает бинарную и многоклассовую классификацию.
Все метрики считаются в micro-averaged режиме:
сначала суммируются TP/FP/FN по всем классам, потом делятся.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import math
from typing import Any


def _same(a: Any, b: Any) -> bool:
    """Сравнивает предсказание с фактом. Вынесено отдельно для читаемости."""
    return a == b


def _quantile(values: list[float], q: float) -> float | None:
    """Вычисляет q-й перцентиль списка значений с линейной интерполяцией.

    Примеры: q=0.5 → медиана, q=0.95 → p95, q=0.99 → p99.
    Возвращает None если список пустой.
    """
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    # pos — дробная позиция в отсортированном массиве
    pos = (len(values) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return values[int(pos)]
    # линейная интерполяция между соседними значениями
    return values[lower] * (upper - pos) + values[upper] * (pos - lower)


def classification_metrics(rows: list[Any]) -> dict[str, float | int | None]:
    """Считает метрики качества по списку строк из JOIN(predictions, actuals).

    Каждая строка должна содержать поля: y_pred (JSON), y_true (JSON), latency_ms.

    Возвращает словарь с ключами:
        samples, accuracy, precision, recall, f1,
        latency_p50_ms, latency_p95_ms, latency_p99_ms.

    Все метрики равны None если нет данных (samples=0).

    Accuracy работает для любого числа классов.
    Precision/recall/F1 — micro-averaged по всем классам,
    что для однолейбловой классификации совпадает с accuracy,
    но API остаётся стабильным и готовым к macro/weighted-режимам.
    """
    if not rows:
        # нет данных за окно — возвращаем None вместо делений на ноль
        return {
            'samples': 0,
            'accuracy': None,
            'precision': None,
            'recall': None,
            'f1': None,
            'latency_p50_ms': None,
            'latency_p95_ms': None,
            'latency_p99_ms': None,
        }

    # десериализуем y_pred и y_true из JSON-строк (хранятся как '"spam"' или '1')
    y_pred = [json.loads(row['y_pred']) for row in rows]
    y_true = [json.loads(row['y_true']) for row in rows]
    total = len(rows)
    correct = sum(1 for pred, true in zip(y_pred, y_true) if _same(pred, true))

    # собираем все уникальные метки из обоих списков
    labels = set(y_pred) | set(y_true)

    # считаем TP, FP, FN для каждого класса (micro-averaging)
    tp = Counter()  # True Positive:  предсказал A и правда A
    fp = Counter()  # False Positive: предсказал A, а было не A
    fn = Counter()  # False Negative: не предсказал A, а было A
    for pred, true in zip(y_pred, y_true):
        for label in labels:
            if pred == label and true == label:
                tp[label] += 1
            elif pred == label and true != label:
                fp[label] += 1
            elif pred != label and true == label:
                fn[label] += 1

    # суммируем по всем классам для micro-average
    sum_tp = sum(tp.values())
    sum_fp = sum(fp.values())
    sum_fn = sum(fn.values())

    # None если делитель равен нулю (все предсказания неопределённы)
    precision = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) else None
    recall = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and (precision + recall) else None

    # собираем latency только для строк, где она была передана
    latencies = [float(row['latency_ms']) for row in rows if row['latency_ms'] is not None]
    return {
        'samples': total,
        'accuracy': correct / total,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'latency_p50_ms': _quantile(latencies, 0.5),
        'latency_p95_ms': _quantile(latencies, 0.95),
        'latency_p99_ms': _quantile(latencies, 0.99),
    }
