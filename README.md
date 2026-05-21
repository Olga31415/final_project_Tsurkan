# ML Quality Monitor

Сервис мониторинга качества ML-модели.

Внешняя система отправляет предсказания модели, а позже — реальные ответы. Сервис хранит поток событий, считает метрики качества в скользящем временном окне, сохраняет историю метрик для графиков деградации и создаёт алерты при падении метрик ниже порога. Доставка алертов сделана через outbox с ретраями, поэтому временная недоступность получателя не приводит к потере алерта.

## Что реализовано

- REST API на FastAPI.
- Быстрая запись предсказаний и реальных ответов в SQLite с WAL-режимом.
- Скользящие окна по времени для каждой модели.
- Метрики: `accuracy`, `precision`, `recall`, `f1`, `latency_p50_ms`, `latency_p95_ms`, `latency_p99_ms`.
- История метрик в таблице `metric_history` для построения графиков.
- Алерты при падении `accuracy` или `f1` ниже порога.
- Надёжная доставка алертов через таблицу `alert_outbox` с exponential backoff.
- Фоновые воркеры: агрегация метрик и доставка алертов.
- Ручной endpoint `/aggregate` для тестов и demo.
- Тесты на расчёт метрик, агрегацию и retry-доставку.

## Архитектура

```text
External producer
  ├── POST /predictions  -> predictions
  └── POST /actuals      -> actuals

Background workers
  ├── MetricsAggregator  -> metric_history
  └── AlertDispatcher    -> alert_outbox -> webhook/log

Consumers / dashboard
  ├── GET /metrics/{model_name}
  └── GET /alerts
```

## Структура проекта

```text
.
├── .github/workflows/ci.yml   # CI на GitHub Actions
├── docs/
│   ├── TZ.md                  # Техническое задание
│   └── RUN.md                 # Пошаговый запуск и проверка
├── examples/
│   └── send_demo_data.py      # Демо-скрипт отправки данных
├── src/
│   └── ml_quality_monitor/    # Исходный код сервиса
│       ├── app.py
│       ├── models.py
│       ├── config.py
│       ├── storage.py
│       ├── metrics.py
│       ├── aggregator.py
│       ├── alerts.py
│       └── worker.py
├── tests/
│   ├── conftest.py
│   ├── test_alerts.py
│   ├── test_api.py
│   ├── test_metrics.py
│   └── test_service.py
├── pyproject.toml
├── requirements.txt
└── pytest.ini
```

## Быстрый старт

```bash
python -m venv .venv
```

Активация виртуального окружения:

- **Linux / macOS:** `source .venv/bin/activate`
- **Windows (PowerShell):** `.\.venv\Scripts\Activate.ps1`

```bash
pip install -r requirements.txt
python -m ml_quality_monitor
```

Сервис будет доступен на `http://127.0.0.1:8000`.

Swagger UI: `http://127.0.0.1:8000/docs`.

## Пример отправки данных

В отдельном терминале:

```bash
python examples/send_demo_data.py
```

Скрипт отправит демо-предсказания и реальные ответы, вызовет ручную агрегацию и выведет рассчитанные метрики и алерты.

## API

### Health check

```bash
curl http://127.0.0.1:8000/health
```

### Отправить предсказание

```bash
curl -X POST http://127.0.0.1:8000/predictions \
  -H 'Content-Type: application/json' \
  -d '{
    "prediction_id": "p-1",
    "model_name": "fraud-detector-v1",
    "y_pred": 1,
    "predicted_at": "2026-01-01T12:00:00Z",
    "latency_ms": 42.5,
    "metadata": {"request_id": "r-1"}
  }'
```

### Отправить реальный ответ

```bash
curl -X POST http://127.0.0.1:8000/actuals \
  -H 'Content-Type: application/json' \
  -d '{
    "prediction_id": "p-1",
    "y_true": 0,
    "arrived_at": "2026-01-01T12:03:00Z"
  }'
```

### Принудительно пересчитать метрики

```bash
curl -X POST http://127.0.0.1:8000/aggregate
```

### Получить историю метрик

```bash
curl 'http://127.0.0.1:8000/metrics/fraud-detector-v1?limit=100'
```

### Получить историю алертов

```bash
curl 'http://127.0.0.1:8000/alerts?limit=100'
```

## Настройки через переменные окружения

| Переменная | Значение по умолчанию | Описание |
|---|---:|---|
| `MLQM_DB_PATH` | `ml_quality_monitor.db` | Путь к SQLite базе |
| `MLQM_WINDOW_MINUTES` | `15` | Размер скользящего окна |
| `MLQM_AGGREGATION_INTERVAL_SECONDS` | `30` | Частота фоновой агрегации |
| `MLQM_ALERT_DELIVERY_INTERVAL_SECONDS` | `5` | Частота попыток доставки алертов |
| `MLQM_ALERT_MAX_ATTEMPTS` | `12` | Максимум попыток доставки |
| `MLQM_ALERT_BASE_BACKOFF_SECONDS` | `2` | Базовая задержка exponential backoff |
| `MLQM_ACCURACY_THRESHOLD` | `0.9` | Порог accuracy |
| `MLQM_F1_THRESHOLD` | `0.85` | Порог F1 |
| `MLQM_WEBHOOK_URL` | пусто | URL получателя алертов; если не задан, алерты пишутся в лог |
| `MLQM_REQUEST_TIMEOUT_SECONDS` | `3` | Таймаут webhook-запроса |

Пример запуска с webhook и другими порогами:

```bash
MLQM_WEBHOOK_URL='https://example.com/alerts' \
MLQM_ACCURACY_THRESHOLD=0.95 \
MLQM_F1_THRESHOLD=0.9 \
python -m ml_quality_monitor
```

## Тесты

```bash
pytest
```
