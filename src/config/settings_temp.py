from pathlib import Path

from dotenv import load_dotenv

# .env를 먼저 로드한 다음에 settings를 import
PROJECT_ROOT = Path(__file__).resolve().parents[2]
dotenv_path = PROJECT_ROOT / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path, override=False)
else:
    load_dotenv(override=False)

# .env 로드 후 settings import (settings.py의 모듈 레벨 코드가 실행되지만,
# 이미 .env가 로드되어 있으므로 환경변수를 읽을 수 있음)
from .settings import SettingsWrapper, load_settings

_settings_instance = load_settings()
settings_temp = SettingsWrapper(_settings_instance)

__all__ = ["settings_temp"]

