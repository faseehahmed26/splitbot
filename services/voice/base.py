from abc import ABC, abstractmethod
from typing import Dict, Any

class VoiceService(ABC):
    @abstractmethod
    async def transcribe_audio(self, audio_bytes: bytes) -> str:
        """Transcribe audio to text"""
        pass
    
    # Potentially add other methods like detect_language, etc. later 