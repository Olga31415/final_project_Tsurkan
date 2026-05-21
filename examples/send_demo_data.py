from __future__ import annotations

from datetime import datetime, timedelta, timezone
import random
import time

import httpx

BASE_URL = 'http://127.0.0.1:8000'
MODEL = 'fraud-detector-v1'


def main() -> None:
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    with httpx.Client(timeout=10) as client:
        for i in range(300):
            y_true = 1 if random.random() < 0.3 else 0
            # First half is healthy, second half is degraded.
            error_rate = 0.05 if i < 150 else 0.35
            y_pred = 1 - y_true if random.random() < error_rate else y_true
            prediction_id = f'demo-{int(time.time())}-{i}'
            predicted_at = now - timedelta(minutes=random.randint(0, 14), seconds=random.randint(0, 59))
            client.post(f'{BASE_URL}/predictions', json={
                'prediction_id': prediction_id,
                'model_name': MODEL,
                'y_pred': y_pred,
                'predicted_at': predicted_at.isoformat(),
                'latency_ms': random.uniform(10, 150),
                'metadata': {'source': 'demo'},
            }).raise_for_status()
            # Simulate delayed label arrival by sending actuals after predictions.
            client.post(f'{BASE_URL}/actuals', json={
                'prediction_id': prediction_id,
                'y_true': y_true,
                'arrived_at': datetime.now(timezone.utc).isoformat(),
            }).raise_for_status()
        response = client.post(f'{BASE_URL}/aggregate')
        response.raise_for_status()
        print('Metric points:')
        print(response.json())
        print('\nAlerts:')
        print(client.get(f'{BASE_URL}/alerts').json())


if __name__ == '__main__':
    main()
