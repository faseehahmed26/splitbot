from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Platform configs
    WHATSAPP_API_TOKEN: str
    WHATSAPP_WEBHOOK_VERIFY: str
    TELEGRAM_BOT_TOKEN: Optional[str] = None
    TELEGRAM_WEBHOOK_PORT: int = 8443 # Added default
    TELEGRAM_WEBHOOK_URL: Optional[str] = None # Added
    TELEGRAM_WEBHOOK_SECRET: Optional[str] = None # Added for webhook verification
    
    # LLM configs
    LLM_PROVIDER: str = "gemini"  # Default to groq
    GEMINI_API_KEY_1: Optional[str] = None
    GEMINI_MODEL_NAME: str = "gemini-1.5-flash-latest" # Added Gemini model name
    OPENAI_API_KEY: Optional[str] = None
    LLAMA_API_URL: Optional[str] = None # This might be for self-hosted Llama
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL_NAME: str = "llama3-8b-8192" # Default to Llama 3.1 8B (assuming this is the ID for 3.1)
                                          # The user specified Llama 3.1 8B. Common ID is llama3-8b-8192 or similar.
                                          # Will use this for now, can be configured via .env
    
    # OCR configs
    OCR_PROVIDER: str = "google_vision"  # google_vision, tesseract
    GOOGLE_CLOUD_PROJECT: Optional[str] = None
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = None

    # Monitoring
    LANGFUSE_PUBLIC_KEY_1: str
    LANGFUSE_SECRET_KEY_1: str
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"
    
    # Database
    REDIS_URL: str = "redis://localhost:6379"
    POSTGRES_URL: Optional[str] = None # Made optional as not used in Day 1 MVP
    
    # Voice
    VOICE_PROVIDER: str = "whisper"  # whisper, assembly
    
    # Fallback State Backup
    FALLBACK_STATE_FILEPATH: Optional[str] = "fallback_state.json" # Filepath for in-memory state backup

    # Logging
    LOG_FILE_PATH: Optional[str] = "logs/app.log" # Path for rotating log file
    LOG_ROTATION_WHEN: str = "midnight" # When to rotate logs (e.g., S, M, H, D, W0-W6, midnight)
    LOG_ROTATION_INTERVAL: int = 1 # Interval for rotation (e.g., 1 for daily if WHEN is D or midnight)
    LOG_ROTATION_BACKUP_COUNT: int = 7 # Number of backup log files to keep

    class Config:
        env_file = ".env"

settings = Settings() 
# print(f"DEBUG: Loaded GEMINI_API_KEY in settings.py: {settings.GEMINI_API_KEY_1}") # Debug print
print(f"DEBUG: Loaded LLM_PROVIDER in settings.py: {settings.LLM_PROVIDER}") # Debug print 
print(f"DEBUG: Loaded LLM_PROVIDER in settings.py: {settings.GEMINI_MODEL_NAME}") # Debug print 
