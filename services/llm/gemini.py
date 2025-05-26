# import google.generativeai as genai
from google import genai
from .base import LLMService
from monitoring.langfuse_client import LangfuseMonitor
from typing import Dict, Any, Optional, List
import json
import logging
import re # For attempt_text_bill_parse
from config.settings import settings
from google.genai import types as genai_types # Assuming 'google.genai' is the new import
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, AsyncRetrying, before_sleep_log
import asyncio # For potential async operations

logger = logging.getLogger(__name__)

# Define retryable Gemini exceptions 
# (Refer to Google AI SDK for specific retryable errors, e.g., google.api_core.exceptions.ResourceExhausted)
# For now, using some common ones that might apply or more generic ones.
RETRYABLE_GEMINI_EXCEPTIONS = (
    # genai.types.generation_types.StopCandidateException, # Old SDK?
    # google.api_core.exceptions.ResourceExhausted, # Common for rate limits - this is from google-api-core
    Exception # Broader catch for now, refine with specific Google API core exceptions
)

class GeminiService(LLMService):
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 model_name: Optional[str] = None, 
                 langfuse_monitor: Optional[LangfuseMonitor] = None):
        super().__init__(langfuse_monitor)
        
        self.api_key = api_key or settings.GEMINI_API_KEY_1
        self.model_name = model_name or settings.GEMINI_MODEL_NAME
        if not self.api_key:
            logger.error("Google API key (for Gemini) not provided or found in settings.")
            raise ValueError("Google API key is required for GeminiService.")
            
        try:
            # New SDK initialization
            self.client = genai.Client(api_key=self.api_key)
            # Test the client - a simple way could be listing models, though not strictly necessary here.
            # self.client.models.list() # Optional: to confirm client works
            logger.info(f"Gemini client (google-genai SDK) initialized with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client (google-genai SDK): {e}", exc_info=True)
            raise

    async def _generate_content_with_retry(
        self, 
        prompt: str, 
        temperature: float,
        is_json_output: bool = False
    ) -> str:
        """Wraps the Gemini API call with tenacity retry logic using the new SDK."""
        retry_config = {
            "stop": stop_after_attempt(3),
            "wait": wait_exponential(multiplier=1, min=1, max=10),
            # "retry": retry_if_exception_type(RETRYABLE_GEMINI_EXCEPTIONS), # Refine
            "before_sleep": before_sleep_log(logger, logging.DEBUG)
        }
        
        gen_config_parts = {
            "temperature": temperature,
        }
        if is_json_output:
            # For the new SDK, response_mime_type is part of GenerateContentConfig
            gen_config_parts["response_mime_type"] = "application/json"
            
        # Construct the configuration object for the new SDK
        generation_config = genai_types.GenerateContentConfig(**gen_config_parts)

        async for attempt in AsyncRetrying(**retry_config):
            with attempt:
                logger.debug(f"Attempting Gemini API call (attempt {attempt.retry_state.attempt_number}) with model {self.model_name} using google-genai SDK")
                
                # New SDK uses client.aio.models.generate_content for async
                # The prompt is passed to 'contents'
                response = await self.client.aio.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=generation_config # Pass the config object
                )
                
                logger.debug(f"Full Gemini API response object: {response}")

                # Response structure might differ. The new SDK docs show response.text directly.
                # Checking response.candidates might still be relevant for safety/errors.
                if not response.candidates or not response.candidates[0].content.parts:
                    logger.warning(f"Gemini response empty or missing candidates/parts. Response: {response}")
                    raise ValueError("Gemini returned an empty or invalid response structure.")
                
                return response.text
        raise RuntimeError("Gemini API call failed after multiple retries.")

    async def process_receipt(self, ocr_text: str) -> Dict[str, Any]:
        prompt = f"""
Analyze the following OCR text from a receipt and extract its details into a structured JSON object.
Follow the JSON output structure and instructions precisely.

**JSON Output Structure:**
{{
  "restaurant_name": "string | null",
  "transaction_date": "YYYY-MM-DD | null",
  "transaction_time": "HH:MM:SS | null",
  "currency": "string | null",
  "items": [
    {{
      "name": "string",
      "quantity": "number",
      "price_per_unit": "number | null",
      "total_price": "number"
    }}
  ],
  "subtotal": "number | null",
  "tax": "number | null",
  "tip": "number | null",
  "total": "number | null"
}}

**Instructions for Extraction:**
1.  Monetary values must be NUMBERS (float or integer), not strings with symbols.
2.  Item quantity defaults to 1 if not specified.
3.  If a field is not found, use `null`. For `items`, use an empty array `[]`.
4.  Infer currency (e.g., $, €, £). If unclear, use `null`.
5.  Be mindful of OCR errors.

**Receipt OCR Text:**
```
{ocr_text}
```

**Return ONLY the valid JSON object described above.**
        """
        
        generation_span = None
        if self.langfuse_monitor and self.langfuse_monitor.client:
            generation_span = self.langfuse_client.generation(
                name="gemini_receipt_processing_v1",
                input={"ocr_text_preview": ocr_text[:300]},
                model=self.model_name,
                metadata={"type": "receipt_processing", "llm_provider": "gemini_google-genai_sdk"}
            )

        try:
            response_text = await self._generate_content_with_retry(
                prompt=prompt,
                temperature=0.1,
                is_json_output=True # This will set response_mime_type in _generate_content_with_retry
            )
            
            if generation_span:
                generation_span.update(output=response_text)
            
            try:
                # The new SDK might parse JSON automatically if response_schema is provided and mime_type is json.
                # Docs: "When possible, the SDK will parse the returned JSON, and return the result in response.parsed."
                # For now, keeping manual json.loads as response_schema isn't explicitly used here for parsing.
                parsed_response = json.loads(response_text) 
                logger.info(f"Successfully processed receipt with Gemini (google-genai SDK). Output (first 200 chars): {str(parsed_response)[:200]}")
                return parsed_response
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response from Gemini (receipt, google-genai SDK): {e}. Raw: {response_text[:500]}", exc_info=True)
                if generation_span: generation_span.end(level="ERROR", status_message=f"JSONDecodeError: {e}")
                return {"error": "Invalid JSON response from LLM", "raw_response": response_text}

        except Exception as e:
            logger.error(f"Error during Gemini receipt processing (google-genai SDK): {e}", exc_info=True)
            if generation_span: generation_span.end(level="ERROR", status_message=str(e))
            # It's good practice to check for specific API errors if the SDK raises them (e.g., API key invalid)
            # For now, a general catch.
            return {"error": f"Gemini API error or unexpected issue (google-genai SDK): {e}"}

    async def process_receipt_from_image(self, image_bytes: bytes, image_mime_type: str = "image/jpeg") -> Dict[str, Any]:
        """Processes a receipt image directly using Gemini multimodal capabilities."""
        # Prompt is similar to process_receipt, but expects the model to "see" the image.
        # The user of this method should ensure the model used (self.model_name) supports multimodal input.
        prompt_text_part = f"""
Analyze the provided receipt image and extract its details into a structured JSON object.
Follow the JSON output structure and instructions precisely.

**JSON Output Structure:**
{{
  "restaurant_name": "string | null",
  "transaction_date": "YYYY-MM-DD | null",
  "transaction_time": "HH:MM:SS | null",
  "currency": "string | null",
  "items": [
    {{
      "name": "string",
      "quantity": "number",
      "price_per_unit": "number | null",
      "total_price": "number"
    }}
  ],
  "subtotal": "number | null",
  "tax": "number | null",
  "tip": "number | null",
  "total": "number | null"
}}

**Instructions for Extraction:**
1.  Monetary values must be NUMBERS (float or integer), not strings with symbols.
2.  Item quantity defaults to 1 if not specified.
3.  If a field is not found, use `null`. For `items`, use an empty array `[]`.
4.  Infer currency (e.g., $, €, £). If unclear, use `null`.
5.  Be mindful of potential OCR errors if interpreting text from the image.

**Return ONLY the valid JSON object described above based on the image content.**
        """

        # Correctly construct the image Part using SDK types
        # Assuming genai_types is imported as from google.genai import types as genai_types
        image_part_sdk = genai_types.Part(inline_data=genai_types.Blob(mime_type=image_mime_type, data=image_bytes))
        
        # Construct the multimodal contents correctly
        # The order can matter: image first, then text prompt.
        contents = [image_part_sdk, prompt_text_part]

        generation_span = None
        if self.langfuse_monitor and self.langfuse_monitor.client:
            generation_span = self.langfuse_client.generation(
                name="gemini_receipt_image_processing_v1",
                input={"image_mime_type": image_mime_type, "image_size": len(image_bytes), "prompt_text_preview": prompt_text_part[:200]},
                model=self.model_name,
                metadata={"type": "receipt_image_processing", "llm_provider": "gemini_google-genai_sdk"}
            )
        
        retry_config = {
            "stop": stop_after_attempt(3),
            "wait": wait_exponential(multiplier=1, min=1, max=10),
            "before_sleep": before_sleep_log(logger, logging.DEBUG)
        }
        
        gen_config_parts = {
            "temperature": 0.1, # Low temperature for structured extraction
            "response_mime_type": "application/json" # Expect JSON output
        }
        generation_config = genai_types.GenerateContentConfig(**gen_config_parts)

        try:
            async for attempt in AsyncRetrying(**retry_config):
                with attempt:
                    logger.debug(f"Attempting Gemini multimodal API call (attempt {attempt.retry_state.attempt_number}) for image processing with model {self.model_name}")
                    response = await self.client.aio.models.generate_content(
                        model=self.model_name, # Ensure this model supports vision
                        contents=contents,
                        config=generation_config
                    )
                    logger.debug(f"Full Gemini API multimodal response object: {response}")
                    if not response.candidates or not response.candidates[0].content.parts:
                        logger.warning(f"Gemini multimodal response empty or missing candidates/parts. Response: {response}")
                        raise ValueError("Gemini returned an empty or invalid response structure from image.")
                    
                    response_text = response.text # Assuming direct text output for JSON

            if generation_span:
                generation_span.update(output=response_text)
            
            try:
                parsed_response = json.loads(response_text)
                logger.info(f"Successfully processed receipt image with Gemini. Output (first 200 chars): {str(parsed_response)[:200]}")
                return parsed_response
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response from Gemini (receipt image): {e}. Raw: {response_text[:500]}", exc_info=True)
                if generation_span: generation_span.end(level="ERROR", status_message=f"JSONDecodeError: {e}")
                return {"error": "Invalid JSON response from LLM (image)", "raw_response": response_text}

        except Exception as e:
            logger.error(f"Error during Gemini receipt image processing: {e}", exc_info=True)
            if generation_span: generation_span.end(level="ERROR", status_message=str(e))
            return {"error": f"Gemini API error or unexpected issue (image processing): {e}"}

    async def attempt_text_bill_parse(self, text_input: str, user_name: str = "User") -> Optional[Dict[str, Any]]:
        processed_text_input = re.sub(r'\\bme\\b', user_name, text_input, flags=re.IGNORECASE)
        processed_text_input = re.sub(r'\\bI\\b', user_name, processed_text_input) # 'I' is usually capitalized

        prompt = f"""
Analyze the text to determine if it's a bill splitting request involving '{user_name}'.
Extract: total amount, currency, participants (including '{user_name}' if "me"/"I" was used), and venue.

**Input Text:** "{processed_text_input}"

**JSON Output Structure:**
{{
  "is_bill_instruction": boolean,
  "total_amount": number | null,
  "currency": "string | null",
  "participants": ["string"] | null,
  "venue": "string | null"
}}

**Instructions:**
- `is_bill_instruction`: true if it's a clear bill split request with amount and participants.
- `total_amount`: Must be a number.
- `currency`: Infer (e.g., "$" -> "USD"). Use `null` if unclear.
- `participants`: List all unique names. Convert "me"/"I" to '{user_name}'.
- `venue`: Restaurant/store name if mentioned.
- If not a bill instruction, `is_bill_instruction` is false, other fields can be null.

**Return ONLY the valid JSON object.**
        """
        generation_span = None
        if self.langfuse_monitor and self.langfuse_monitor.client:
            generation_span = self.langfuse_client.generation(
                name="gemini_text_bill_parse_v1",
                input={"text_input": text_input, "user_name": user_name},
                model=self.model_name,
                metadata={"type": "text_bill_parsing", "llm_provider": "gemini_google-genai_sdk"}
            )
        
        try:
            response_text = await self._generate_content_with_retry(
                prompt=prompt,
                temperature=0.2,
                is_json_output=True # For JSON mime type
            )

            if generation_span:
                generation_span.update(output=response_text)

            # Similar to process_receipt, checking for response.parsed if schema was used.
            # Sticking to manual parsing for now.
            parsed_response = json.loads(response_text)
            logger.info(f"Gemini text bill parse result (google-genai SDK): {str(parsed_response)[:200]}")

            if parsed_response.get("is_bill_instruction") is True:
                if parsed_response.get("total_amount") is not None and parsed_response.get("participants"):
                    return parsed_response
                else:
                    logger.warning(f"Gemini indicated bill instruction but missing total/participants (google-genai SDK). Text: {text_input}")
                    if generation_span: generation_span.end(level="WARNING", status_message="Bill instruction true but key fields missing.")
                    return None
            else:
                logger.info(f"Text not a bill instruction by Gemini (google-genai SDK): {text_input}")
                return None

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from Gemini (text_bill_parse, google-genai SDK): {e}. Raw: {response_text[:500]}", exc_info=True)
            if generation_span: generation_span.end(level="ERROR", status_message=f"JSONDecodeError: {e}")
            return None
        except Exception as e:
            logger.error(f"Error during Gemini text bill parsing (google-genai SDK): {e}", exc_info=True)
            if generation_span: generation_span.end(level="ERROR", status_message=str(e))
            return None

    async def process_voice_command(self, transcription: str, context: Dict[str, Any]) -> Dict[str, Any]:
        logger.warning(
            f"GeminiService.process_voice_command is a placeholder and not fully implemented (google-genai SDK). "
            f"Received transcription: {transcription[:100]}"
        )
        if self.langfuse_monitor and self.langfuse_monitor.client:
            trace = self.langfuse_monitor.trace(
                name="gemini_process_voice_command_placeholder",
                input={"transcription_preview": transcription[:100], "context_preview": str(context)[:100]},
                metadata={"llm_provider": "gemini_google-genai_sdk", "status": "placeholder_not_implemented"}
            )
            # For placeholder, not making an actual LLM call.
            # If it did, it would use client.aio.models.generate_content(...)
            trace.update(output={"message": "Placeholder for voice command processing."})

        return {
            "intent": "unknown_voice_intent",
            "message": "Voice command processing with Gemini is not yet implemented.",
            "details": "This is a placeholder response."
        }

    async def interpret_correction_command(self, user_command: str, current_bill_json: str) -> Dict[str, Any]:
        logger.warning("GeminiService.interpret_correction_command is not fully implemented (google-genai SDK).")
        # If this made an LLM call, it would need to be updated to the new SDK style.
        return {"action": "clarify", "message": "Correction interpretation with Gemini is not yet supported."}

    async def generate_conversational_response(self, text_input: str, current_state: Optional[Dict[str, Any]] = None) -> str:
        logger.warning("GeminiService.generate_conversational_response is not fully implemented (google-genai SDK).")
        # If this made an LLM call, it would need to be updated to the new SDK style.
        return "I'm still learning to chat with Gemini (using the new SDK)!"

    async def interpret_split_instructions(
        self, 
        instruction_text: str, 
        current_bill_data: Dict[str, Any], 
        conversation_history: str,
        user_name: str
    ) -> Dict[str, Any]:
        logger.info(f"[MOCK] GeminiService: interpret_split_instructions for user {user_name} - '{instruction_text}'")
        # Simulate LLM processing based on instruction content for mock
        total_bill = float(current_bill_data.get("total", 0))
        currency = current_bill_data.get("currency", "$")
        
        if "equally" in instruction_text.lower() and total_bill > 0:
            # Simplified participant extraction from instruction_text or use a default
            participants_match = re.findall(r'between (.*) and (.*)|with (.*)', instruction_text, re.IGNORECASE)
            # This is very basic, a real LLM would do better participant identification from history/context
            participant_names = [name for group in participants_match for name in group if name and name.lower() != "me"]
            if "me" in instruction_text.lower() or not participant_names: participant_names.append(user_name)
            participant_names = list(set(name.strip().capitalize() for name in participant_names if name.strip()))
            if not participant_names: participant_names = [user_name, "Friend 1"] # Fallback
            
            num_participants = len(participant_names)
            amount_per_person = round(total_bill / num_participants, 2) if num_participants > 0 else total_bill
            
            return {
                "total": total_bill,
                "currency": currency,
                "breakdown": {name: amount_per_person for name in participant_names},
                "is_final": True,
                "summary_text": f"Split {total_bill}{currency} equally among {num_participants} people: {amount_per_person}{currency} each."
            }
        elif "saquib will pay 50%" in instruction_text.lower() and total_bill > 0: # User example
            saquib_pays = round(total_bill * 0.50, 2)
            remaining = round(total_bill - saquib_pays, 2)
            # Assuming "me" and "praneet" are the other two for this mock.
            # A real LLM would infer this from conversation_history or bill_data.participants
            me_pays = round(remaining / 2, 2)
            praneet_pays = round(remaining / 2, 2)
            return {
                "total": total_bill, "currency": currency,
                "breakdown": {"Saquib": saquib_pays, user_name: me_pays, "Praneet": praneet_pays},
                "is_final": True,
                "summary_text": f"Total: {total_bill}{currency}. Saquib pays {saquib_pays}{currency}. {user_name} & Praneet pay {me_pays}{currency} each."
            }
        else:
            return {"is_final": False, "clarification_needed": "I didn't fully understand the split. Can you rephrase? E.g., 'split equally between me and John', or 'item 1 for me, rest for John'."}

    async def apply_split_adjustment(
        self, 
        adjustment_text: str, 
        current_split_data: Dict[str, Any],
        current_bill_data: Dict[str, Any],
        conversation_history: str,
        user_name: str
    ) -> Optional[Dict[str, Any]]:
        logger.info(f"[MOCK] GeminiService: apply_split_adjustment for user {user_name} - '{adjustment_text}'")
        # Mock: Assume any adjustment means we can't auto-apply, return None to re-prompt.
        # A real implementation would try to parse adjustment_text and modify current_split_data.
        return None 