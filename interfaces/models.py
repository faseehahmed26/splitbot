from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
import time

@dataclass
class Message:
    text: str
    user_id: Optional[str] = None
    # Potentially add other fields like attachments, message_id, etc. later 

@dataclass
class ReceiptItem:
    name: str
    quantity: float = 1.0
    price_per_unit: Optional[float] = None
    total_price: Optional[float] = None

@dataclass
class ReceiptData:
    restaurant_name: Optional[str] = None
    items: List[ReceiptItem] = field(default_factory=list)
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    tip: Optional[float] = None
    total: Optional[float] = None
    currency: Optional[str] = None
    transaction_date: Optional[str] = None
    transaction_time: Optional[str] = None

@dataclass
class ConversationStateData:
    step: str
    user_id: str
    history: List[Dict[str, Any]] = field(default_factory=list)
    receipt_data: Optional[ReceiptData] = None
    participants: Optional[List[str]] = None
    splits: Optional[Dict[str, float]] = None
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

    # New fields from strategy
    conversation_id: Optional[str] = None
    current_state: str = "INITIAL"
    bill_data: Optional[Dict[str, Any]] = None
    split_data: Optional[Dict[str, Any]] = None
    user_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "user_id": self.user_id,
            "history": self.history,
            "receipt_data": self.receipt_data.__dict__ if self.receipt_data else None,
            "participants": self.participants,
            "splits": self.splits,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "conversation_id": self.conversation_id,
            "current_state": self.current_state,
            "bill_data": self.bill_data,
            "split_data": self.split_data,
            "user_name": self.user_name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationStateData':
        receipt_data_dict = data.get("receipt_data")
        if receipt_data_dict and isinstance(receipt_data_dict, dict):
            items_list = receipt_data_dict.get("items", [])
            receipt_data_dict["items"] = [ReceiptItem(**item) for item in items_list if isinstance(item, dict)]
            data["receipt_data"] = ReceiptData(**receipt_data_dict)
        
        if "step" not in data and "current_state" in data:
            data["step"] = data["current_state"]
        elif "step" not in data:
            data["step"] = "INITIAL"

        return cls(**data)

    def add_message(self, message_type: str, content: Any, caption: Optional[str] = None, transcription: Optional[str] = None, role: str = "user"):
        message_entry = {
            "role": role,
            "type": message_type,
            "content": content,
            "caption": caption,
            "transcription": transcription,
            "timestamp": time.time()
        }
        if not isinstance(self.history, list):
            self.history = []
        self.history.append(message_entry)
        self.last_updated = time.time()

    def get_llm_context_history(self, last_n_messages: int = 5) -> str:
        context_lines = []
        for msg in self.history[-last_n_messages:]:
            role = msg.get("role", "user").capitalize()
            msg_type = msg.get("type", "text")
            line = f"{role}: "
            if msg_type == "text":
                line += str(msg.get("content", ""))
            elif msg_type == "image":
                line += f"[Image"
                if msg.get("caption"):
                    line += f" with caption: {msg.get('caption')}"
                line += "]"
            elif msg_type == "audio":
                line += f"[Audio transcription: {msg.get('transcription', 'N/A')}]"
            else:
                line += f"[{msg_type.capitalize()} message: {str(msg.get('content',''))[:50]}...]"
            context_lines.append(line)
        return "\n".join(context_lines) 