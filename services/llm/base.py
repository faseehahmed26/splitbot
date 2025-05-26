from abc import ABC, abstractmethod
# from langfuse import Langfuse # No longer directly type hint Langfuse SDK client here
from monitoring.langfuse_client import LangfuseMonitor # Import LangfuseMonitor
from typing import Dict, Any, Optional
import json
class LLMService(ABC):
    def __init__(self, langfuse_monitor: Optional[LangfuseMonitor]): # Accept LangfuseMonitor
        self.langfuse_monitor = langfuse_monitor
        self.langfuse_client = langfuse_monitor.get_client() if langfuse_monitor else None
        
    @abstractmethod
    async def process_receipt(self, ocr_text: str) -> Dict[str, Any]:
        """Extract structured data from receipt text"""
        pass
    
    @abstractmethod
    async def process_voice_command(self, 
                                   transcription: str, 
                                   context: Dict) -> Dict[str, Any]:
        """Process voice commands with context"""
        prompt = f"""
            Given the transcription: "{transcription}"
            And the current conversation context: {json.dumps(context, indent=2)}
            
            Determine the user's intent and extract relevant information.
            Return as a valid JSON object.
            Example: {{"intent": "add_item_to_person", "item_name": "pizza", "person_name": "John"}}
            Possible intents: assign_item, query_total, confirm_split, modify_split, etc.
            """
    
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