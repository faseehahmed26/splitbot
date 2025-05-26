from abc import ABC, abstractmethod
# from langfuse import Langfuse # No longer directly type hint Langfuse SDK client here
from monitoring.langfuse_client import LangfuseMonitor # Import LangfuseMonitor
from typing import Dict, Any, Optional
import json
class LLMService(ABC):
    def __init__(self, langfuse_monitor: Optional[LangfuseMonitor]): # Accept LangfuseMonitor
        self.langfuse_monitor = langfuse_monitor
        self.langfuse_client = langfuse_monitor.get_client() if langfuse_monitor else None
        self.model_name: Optional[str] = None # To be set by subclasses
        
    @abstractmethod
    async def process_receipt(self, text_input_with_caption_and_ocr: str) -> Dict[str, Any]:
        """Extract structured data from receipt text (potentially combined with caption)."""
        pass
    
    @abstractmethod
    async def attempt_text_bill_parse(self, text_input: str, user_name: str = "User") -> Optional[Dict[str, Any]]:
        """Attempt to parse a short text input to see if it's a bill splitting instruction.
           Should return a dict similar to process_receipt if it is a bill, or include {"is_bill_instruction": False}.
        """
        pass

    @abstractmethod
    async def interpret_split_instructions(
        self, 
        instruction_text: str, 
        current_bill_data: Dict[str, Any], 
        conversation_history: str, # Formatted string of recent messages
        user_name: str
    ) -> Dict[str, Any]:
        """Interpret user's natural language on how to split the bill.
           Returns a dictionary with calculated split, or requests clarification.
           Example output: {"total": 50.0, "currency": "USD", "breakdown": {"Alice": 25.0, "Bob": 25.0}, "is_final": True, "summary_text": "..."}
           Or: {"is_final": False, "clarification_needed": "Which item for Bob?"}
        """
        pass

    @abstractmethod
    async def apply_split_adjustment(
        self, 
        adjustment_text: str, 
        current_split_data: Dict[str, Any],
        current_bill_data: Dict[str, Any],
        conversation_history: str,
        user_name: str
    ) -> Optional[Dict[str, Any]]:
        """Apply user's correction/adjustment to an existing split.
           Returns updated split_data if successful, None otherwise.
        """
        pass

    # Removed _track_llm_call as individual services should use generation or span directly.
    # If a generic LLM call tracker is needed, it should use langfuse_client.generation().
    # def _track_llm_call(self, 
    #                    prompt: str, 
    #                    response: str, 
    #                    metadata: Dict):
    #     """Track all LLM calls in Langfuse"""
    #     if self.langfuse_client:
    #         self.langfuse_client.generation(
    #             name="llm_call", # Or a more specific name passed in
    #             input=prompt,
    #             output=response,
    #             metadata=metadata
    #         ) 