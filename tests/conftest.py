"""Pytest: переменные окружения до первого импорта ml_quality_monitor.config."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_tmp = tempfile.mkdtemp(prefix='mlqm_pytest_')
os.environ['MLQM_DB_PATH'] = str(Path(_tmp) / 'pytest_app.db')
