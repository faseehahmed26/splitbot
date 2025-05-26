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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "user_id": self.user_id,
            "history": self.history,
            "receipt_data": self.receipt_data.__dict__ if self.receipt_data else None,
            "participants": self.participants,
            "splits": self.splits,
            "created_at": self.created_at,
            "last_updated": self.last_updated
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationStateData':
        receipt_data_dict = data.get("receipt_data")
        if receipt_data_dict:
            items_list = receipt_data_dict.get("items", [])
            receipt_data_dict["items"] = [ReceiptItem(**item) for item in items_list]
            data["receipt_data"] = ReceiptData(**receipt_data_dict)
        return cls(**data) 