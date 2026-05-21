from __future__ import annotations

from ml_quality_monitor.alerts import AlertDispatcher
from ml_quality_monitor.config import Settings
from ml_quality_monitor.storage import SQLiteStore


class FlakySink:
    def __init__(self):
        self.calls = 0

    def send(self, payload: dict) -> None:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError('temporary unavailable')


def test_alert_delivery_retries_then_delivers(tmp_path):
    db_path = tmp_path / 'alerts.db'
    store = SQLiteStore(str(db_path))
    settings = Settings(db_path=str(db_path), alert_base_backoff_seconds=0, alert_max_attempts=3)
    store.enqueue_alert(
        model_name='m', metric_name='accuracy', metric_value=0.1, threshold=0.9,
        payload={'hello': 'world'}, dedup_key='m:accuracy:1'
    )

    sink = FlakySink()
    dispatcher = AlertDispatcher(store, settings, sink=sink)
    assert dispatcher.deliver_due() == 0
    assert store.fetch_alerts()[0]['status'] == 'pending'
    assert dispatcher.deliver_due() == 1
    assert store.fetch_alerts()[0]['status'] == 'delivered'
