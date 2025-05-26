from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from monitoring.langfuse_client import LangfuseMonitor # Assuming LangfuseMonitor is in this path

class OCRService(ABC):
    def __init__(self, langfuse_monitor: Optional[LangfuseMonitor] = None):
        self.langfuse_monitor = langfuse_monitor
        self.langfuse_client = langfuse_monitor.get_client() if langfuse_monitor else None

    @abstractmethod
    async def extract_text_from_bytes(self, image_bytes: bytes) -> str:
        """Extracts text from an image provided as bytes."""
        pass

    @abstractmethod
    async def extract_structured_data(self, image_bytes: bytes) -> Dict[str, Any]:
        """Extract structured receipt data"""
        pass 