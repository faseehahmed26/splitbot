from abc import ABC, abstractmethod
from .models import Message

# Define Message class in models.py as per PRD structure
# from .models import Message 

class MessagingInterface(ABC):
    @abstractmethod
    async def send_message(self, user_id: str, message: Message):
        pass
    
    @abstractmethod
    async def send_image(self, user_id: str, image: bytes):
        pass
    
    @abstractmethod
    async def handle_incoming(self, data: dict):
        pass 