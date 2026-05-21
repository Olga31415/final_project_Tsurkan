from __future__ import annotations

from types import SimpleNamespace

from ml_quality_monitor.metrics import classification_metrics


class Row(dict):
    def __getattr__(self, name):
        return self[name]


def test_classification_metrics_accuracy_and_f1():
    rows = [
        Row(y_pred='1', y_true='1', latency_ms=10),
        Row(y_pred='0', y_true='0', latency_ms=20),
        Row(y_pred='1', y_true='0', latency_ms=30),
        Row(y_pred='0', y_true='1', latency_ms=40),
    ]
    metrics = classification_metrics(rows)
    assert metrics['samples'] == 4
    assert metrics['accuracy'] == 0.5
    assert metrics['precision'] == 0.5
    assert metrics['recall'] == 0.5
    assert metrics['f1'] == 0.5
    assert metrics['latency_p50_ms'] == 25


def test_classification_metrics_empty_window():
    metrics = classification_metrics([])
    assert metrics['samples'] == 0
    assert metrics['accuracy'] is None
    assert metrics['f1'] is None
