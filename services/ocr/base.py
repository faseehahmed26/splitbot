from abc import ABC, abstractmethod
from typing import Dict, Any

class OCRService(ABC):
    @abstractmethod
    async def extract_text(self, image_bytes: bytes) -> str:
        """Extract text from image"""
        pass
    
    @abstractmethod
    async def extract_structured_data(self, image_bytes: bytes) -> Dict[str, Any]:
        """Extract structured receipt data"""
        pass 