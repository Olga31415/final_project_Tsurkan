"""Точка входа для запуска сервиса командой: python -m ml_quality_monitor

Запускает uvicorn (ASGI-сервер) с FastAPI-приложением из app.py.
host='0.0.0.0' — слушает на всех сетевых интерфейсах.
port=8000      — стандартный порт для разработки.
"""
import uvicorn

if __name__ == '__main__':
    uvicorn.run('ml_quality_monitor.app:app', host='0.0.0.0', port=8000, reload=False)
