from __future__ import annotations

from datetime import datetime, timedelta, timezone

from ml_quality_monitor.aggregator import MetricsAggregator
from ml_quality_monitor.config import Settings
from ml_quality_monitor.storage import SQLiteStore


def test_aggregation_saves_metric_and_enqueues_alert(tmp_path):
    db_path = tmp_path / 'test.db'
    store = SQLiteStore(str(db_path))
    settings = Settings(db_path=str(db_path), window_minutes=15, accuracy_threshold=0.8, f1_threshold=0.8)
    now = datetime(2026, 1, 1, 12, 10, tzinfo=timezone.utc)

    for i in range(10):
        prediction_id = f'p-{i}'
        store.upsert_prediction(
            prediction_id=prediction_id,
            model_name='model-a',
            y_pred=1 if i < 4 else 0,
            predicted_at=now - timedelta(minutes=5),
            latency_ms=10 + i,
            metadata={},
        )
        store.upsert_actual(
            prediction_id=prediction_id,
            y_true=1,
            arrived_at=now,
        )

    points = MetricsAggregator(store, settings).aggregate_once(now)
    assert len(points) == 1
    assert points[0]['samples'] == 10
    assert points[0]['accuracy'] == 0.4

    history = store.fetch_metric_history('model-a')
    assert len(history) == 1

    alerts = store.fetch_alerts()
    assert len(alerts) == 2
    assert {alert['metric_name'] for alert in alerts} == {'accuracy', 'f1'}
