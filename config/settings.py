from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Platform configs
    WHATSAPP_API_TOKEN: str
    WHATSAPP_WEBHOOK_VERIFY: str
    TELEGRAM_BOT_TOKEN: Optional[str] = None
    TELEGRAM_WEBHOOK_PORT_1: int = 8443
    TELEGRAM_WEBHOOK_URL_1: Optional[str] = None
    TELEGRAM_WEBHOOK_SECRET: Optional[str] = None
    
    # LLM configs
    LLM_PROVIDER: str = "gemini"
    GEMINI_API_KEY_1: Optional[str] = None
    GEMINI_MODEL_NAME: str = "gemini-1.5-flash-latest"
    
    # Monitoring
    LANGFUSE_PUBLIC_KEY_1: str
    LANGFUSE_SECRET_KEY_1: str
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"
    
    # Database
    REDIS_URL: str = "redis://localhost:6379"
    
    # Fallback State Backup
    FALLBACK_STATE_FILEPATH: Optional[str] = "fallback_state.json"
    
    # Logging
    LOG_FILE_PATH: Optional[str] = "logs/app.log"
    LOG_ROTATION_WHEN: str = "midnight"
    LOG_ROTATION_INTERVAL: int = 1
    LOG_ROTATION_BACKUP_COUNT: int = 7

    class Config:
        env_file = ".env"

settings = Settings()

# Debug prints to verify loaded settings (optional, remove for production)
print(f"DEBUG: Loaded LLM_PROVIDER in settings.py: {settings.LLM_PROVIDER}")
print(f"DEBUG: Loaded GEMINI_MODEL_NAME in settings.py: {settings.GEMINI_MODEL_NAME}")
if settings.GEMINI_API_KEY_1:
    print("DEBUG: GEMINI_API_KEY_1 is loaded.")
else:
    print("DEBUG: GEMINI_API_KEY_1 is NOT loaded or is empty.")
