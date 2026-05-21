from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from ml_quality_monitor.app import app


def test_health_ok():
    with TestClient(app) as client:
        response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}


def test_prediction_actual_aggregate_and_metrics():
    suffix = uuid.uuid4().hex[:12]
    model_name = f'api-test-{suffix}'
    prediction_id = f'pred-{suffix}'
    now = datetime.now(timezone.utc)
    predicted_at = (now - timedelta(minutes=1)).isoformat().replace('+00:00', 'Z')
    arrived_at = now.isoformat().replace('+00:00', 'Z')

    with TestClient(app) as client:
        r_pred = client.post(
            '/predictions',
            json={
                'prediction_id': prediction_id,
                'model_name': model_name,
                'y_pred': 1,
                'predicted_at': predicted_at,
                'latency_ms': 12.0,
                'metadata': {},
            },
        )
        assert r_pred.status_code == 202

        r_act = client.post(
            '/actuals',
            json={
                'prediction_id': prediction_id,
                'y_true': 1,
                'arrived_at': arrived_at,
            },
        )
        assert r_act.status_code == 202

        r_agg = client.post('/aggregate')
        assert r_agg.status_code == 200
        points = r_agg.json()
        assert isinstance(points, list)
        assert any(p['model_name'] == model_name for p in points)

        r_hist = client.get(f'/metrics/{model_name}', params={'limit': 10})
        assert r_hist.status_code == 200
        hist = r_hist.json()
        assert len(hist) >= 1
        row = next(x for x in hist if x['model_name'] == model_name)
        assert row['samples'] >= 1
