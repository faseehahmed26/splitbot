import groq # type: ignore
from .base import LLMService
from monitoring.langfuse_client import LangfuseMonitor # Import LangfuseMonitor
from typing import Dict, Any, Optional, List # Added List
import json # For parsing JSON from LLM response
import logging
from config.settings import settings # For API key if not passed directly
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, AsyncRetrying, before_sleep_log # Import tenacity
import re # Added re for attempt_text_bill_parse method

logger = logging.getLogger(__name__)

# Define retryable Groq exceptions (refer to Groq SDK for specific retryable errors)
# Common ones might include APIError, RateLimitError, APIConnectionError, APITimeoutError
RETRYABLE_GROQ_EXCEPTIONS = (
    groq.APIError, # General API error
    groq.RateLimitError, 
    groq.APIConnectionError,
    groq.APITimeoutError,
    groq.InternalServerError # Adding this as it's often retryable
)

class GroqService(LLMService):
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 model_name: Optional[str] = None, 
                 langfuse_monitor: Optional[LangfuseMonitor] = None): # Accept LangfuseMonitor
        super().__init__(langfuse_monitor) # Pass LangfuseMonitor to base
        
        self.api_key = api_key or settings.GROQ_API_KEY
        self.model_name = model_name or settings.GROQ_MODEL_NAME
        
        if not self.api_key:
            logger.error("Groq API key not provided or found in settings.")
            raise ValueError("Groq API key is required.")
            
        try:
            self.client = groq.Groq(api_key=self.api_key)
            logger.info(f"Groq client initialized with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}", exc_info=True)
            raise

    async def _create_chat_completion_with_retry(
    self, 
    messages: list, 
    model: str, 
    temperature: float, 
    response_format: Optional[dict]
    ) -> Any:  # Changed from groq.types.chat.chat_completion.ChatCompletion to Any
        """Wraps the Groq API call with tenacity retry logic."""
        retry_config = {
            "stop": stop_after_attempt(3),
            "wait": wait_exponential(multiplier=1, min=1, max=10),
            "retry": retry_if_exception_type(RETRYABLE_GROQ_EXCEPTIONS),
            "before_sleep": before_sleep_log(logger, logging.DEBUG)
        }
        async for attempt in AsyncRetrying(**retry_config):
            with attempt:
                logger.debug(f"Attempting Groq API call (attempt {attempt.retry_state.attempt_number}) with model {model}")
                completion = await self.client.chat.completions.create(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    response_format=response_format
                )
                return completion
        raise RuntimeError("Groq API call failed after multiple retries without re-raising properly.")

    async def process_receipt(self, ocr_text: str) -> Dict[str, Any]:
        # Enhanced prompt for receipt processing
        prompt = f"""
Please analyze the following OCR text from a receipt and extract its details into a structured JSON object.

**JSON Output Structure:**
The JSON object should strictly follow this structure:
{{
  "restaurant_name": "string | null", // Name of the store or restaurant
  "transaction_date": "YYYY-MM-DD | null", // If available
  "transaction_time": "HH:MM:SS | null", // If available
  "currency": "string | null", // e.g., "USD", "EUR". Infer if possible.
  "items": [ // Array of item objects
    {{
      "name": "string", // Description of the item
      "quantity": "number", // Quantity of the item, default to 1 if not specified
      "price_per_unit": "number | null", // Price per single unit of the item, if specified
      "total_price": "number" // Total price for this line item (quantity * price_per_unit if applicable)
    }}
  ],
  "subtotal": "number | null", // Subtotal before tax and tip, if explicitly mentioned
  "tax": "number | null", // Total tax amount
  "tip": "number | null", // Total tip amount
  "total": "number | null" // Grand total amount
}}

**Instructions for Extraction:**
1.  **Accuracy:** Be as accurate as possible. Ensure all monetary values (prices, tax, tip, total, subtotal) are NUMBERS (float or integer), not strings with currency symbols.
2.  **Items:**
    *   For each item, extract its name, quantity, price per unit (if available), and total price for that line.
    *   If quantity is not specified, assume 1.
    *   If only a total price for the line item is given, use that for "total_price".
3.  **Missing Values:** If a field (like `restaurant_name`, `transaction_date`, `tax`, `tip`, `subtotal`) is not found or cannot be confidently extracted, use `null` for its value. For `items`, if no items can be identified, use an empty array `[]`.
4.  **Totals:**
    *   Identify the final grand total.
    *   If a subtotal (sum of items before tax/tip) is explicitly mentioned, extract it.
    *   Extract tax and tip if they are separate line items.
5.  **Currency:** Try to infer the currency (e.g., from symbols like $, €, £). If not clear, set to `null`.
6.  **OCR Ambiguities:** Be mindful of common OCR errors (e.g., 'O' vs '0', 'S' vs '5', 'l' vs '1'). Try to interpret amounts and item names correctly.
7.  **Flexibility:** Receipts vary greatly. Adapt to different layouts. If multiple possible values are found for a field (e.g. multiple totals), try to pick the most likely grand total.

**Receipt OCR Text:**
```
{ocr_text}
```

**Return ONLY the valid JSON object described above.**
        """
        
        generation = None # Use 'generation' to align with Langfuse terminology for LLM calls
        if self.langfuse_client: # Check if langfuse_client (from base class) is available
            generation = self.langfuse_client.generation(
                name="groq_receipt_processing_v2", # Updated name for new prompt version
                input={ "ocr_text_preview": ocr_text[:300], "prompt_template_version": "2.0" }, # Log a preview and prompt version
                model=self.model_name,
                metadata={"type": "receipt_processing", "llm_provider": "groq"}
            )

        try:
            chat_completion = await self._create_chat_completion_with_retry(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                temperature=0.1, # Lower temperature for more deterministic output for data extraction
                response_format={"type": "json_object"} # Request JSON output if model supports it
            )
            
            response_text = chat_completion.choices[0].message.content
            
            cleaned_response_text = response_text.strip()
            if cleaned_response_text.startswith("```json"):
                cleaned_response_text = cleaned_response_text[7:]
            if cleaned_response_text.endswith("```"):
                cleaned_response_text = cleaned_response_text[:-3]
            cleaned_response_text = cleaned_response_text.strip()

            if generation:
                generation.update(output=cleaned_response_text)
            
            try:
                # Attempt to parse the JSON response
                # Llama3 with JSON mode should return valid JSON string.
                parsed_response = json.loads(cleaned_response_text)
                logger.info(f"Successfully processed receipt with Groq. Output (first 200 chars): {str(parsed_response)[:200]}")
                return parsed_response
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response from Groq: {e}. Raw Response: {response_text[:500]}, Cleaned Response: {cleaned_response_text[:500]}")
                if generation: generation.end(level="ERROR", status_message=f"JSONDecodeError: {e}. Raw: {response_text[:100]}")
                return {"error": "Invalid JSON response from LLM", "raw_response": response_text, "cleaned_response": cleaned_response_text}

        except RETRYABLE_GROQ_EXCEPTIONS as e: # Catch retryable exceptions if all retries fail
            logger.error(f"Groq API call for receipt processing failed after retries: {e}", exc_info=True)
            if generation: generation.end(level="ERROR", status_message=f"Groq API call failed after retries: {str(e)}")
            return {"error": f"Groq API error after retries: {e}"}
        except Exception as e:
            logger.error(f"Unexpected error during Groq receipt processing: {e}", exc_info=True)
            if generation: generation.end(level="ERROR", status_message=str(e))
            return {"error": f"Unexpected error: {e}"}

    async def interpret_correction_command(self, user_command: str, current_bill_json: str) -> Dict[str, Any]:
        """Interprets a user's command to correct extracted bill details."""
        prompt = f"""A user wants to correct previously extracted bill information. 
Given the user's command and the current JSON representation of the bill, identify the corrections and 
return a new JSON object representing the *changes* to be applied. If the command is unclear or not a correction, 
indicate that.

**Current Bill Data (JSON string):**
```json
{current_bill_json}
```

**User Correction Command:**
`{user_command}`

**Desired JSON Output Structure for Changes:**
Output a JSON object specifying only the fields that need to be changed. 
For item corrections, specify the 0-based index of the item in the original 'items' array.

Examples:
- User command: "Item 1 name is Super Burger, price 12.99"
  Output: {{ "items": [{{ "index": 0, "name": "Super Burger", "total_price": 12.99 }}] }}
- User command: "The total should be 55.20"
  Output: {{ "total": 55.20 }}
- User command: "Restaurant name is Mike's Diner"
  Output: {{ "restaurant_name": "Mike's Diner" }}
- User command: "Tax is 2.50, not 2.00"
  Output: {{ "tax": 2.50 }}
- User command: "Remove item 3"
  Output: {{ "items": [{{ "index": 2, "_delete": true }}] }} // Use _delete marker for removal
- User command: "Add an item: Fries, price 3.50"
  Output: {{ "items_to_add": [{{ "name": "Fries", "quantity": 1, "total_price": 3.50 }}] }}
- User command: "That's not what I meant / this is confusing"
  Output: {{ "action": "clarify", "message": "Command unclear or not a correction." }}

**Instructions:**
1.  Analyze the user command in the context of the current bill data.
2.  Identify which parts of the bill the user wants to change (e.g., restaurant_name, total, specific item name/price/quantity, add/remove item).
3.  If an item is being corrected, use its 0-based index from the provided `current_bill_json`.
4.  If an item is to be removed, include `"_delete": true` for that item index.
5.  If new items are to be added, use an `"items_to_add"` array with the new item details.
6.  If the command is not a clear correction (e.g., a general statement, a question unrelated to correction), return an action to clarify.
7.  Ensure monetary values in the output are NUMBERS.
8.  Return ONLY the JSON object with the changes or clarification action.
"""
        generation = None
        if self.langfuse_client:
            generation = self.langfuse_client.generation(
                name="groq_interpret_correction_v1",
                input={"user_command": user_command, "current_bill_json_preview": current_bill_json[:300]},
                model=self.model_name,
                metadata={"type": "correction_interpretation", "llm_provider": "groq"}
            )
        try:
            chat_completion = await self._create_chat_completion_with_retry(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name, temperature=0.2,
                response_format={"type": "json_object"}
            )
            response_text = chat_completion.choices[0].message.content
            cleaned_response_text = response_text.strip()
            if cleaned_response_text.startswith("```json"):
                cleaned_response_text = cleaned_response_text[7:]
            if cleaned_response_text.endswith("```"):
                cleaned_response_text = cleaned_response_text[:-3]
            cleaned_response_text = cleaned_response_text.strip()
            if generation: generation.update(output=cleaned_response_text)
            try:
                parsed_response = json.loads(cleaned_response_text)
                logger.info(f"Successfully interpreted correction command. Changes: {str(parsed_response)[:200]}")
                return parsed_response
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from correction interpretation: {e}. Raw: {response_text[:500]}")
                if generation: generation.end(level="ERROR", status_message=f"JSONDecodeError from correction: {e}")
                return {"action": "clarify", "message": "Internal error parsing correction structure.", "error_detail": str(e)}
        except RETRYABLE_GROQ_EXCEPTIONS as e:
            logger.error(f"Groq API call for correction failed after retries: {e}", exc_info=True)
            if generation: generation.end(level="ERROR", status_message=f"Groq API call for correction failed: {str(e)}")
            return {"action": "clarify", "message": f"API error during correction: {e}"}
        except Exception as e:
            logger.error(f"Unexpected error during correction interpretation: {e}", exc_info=True)
            if generation: generation.end(level="ERROR", status_message=str(e))
            return {"action": "clarify", "message": f"Unexpected error: {e}"}

    async def process_voice_command(self, transcription: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # This is a placeholder, similar to process_receipt but tailored for voice commands
        prompt = f"""
        Given the transcription: "{transcription}"
        And the current conversation context: {json.dumps(context, indent=2)}
        
        Determine the user's intent and extract relevant information.
        Return as a valid JSON object.
        Example: {{"intent": "add_item_to_person", "item_name": "pizza", "person_name": "John"}}
        Possible intents: assign_item, query_total, confirm_split, modify_split, etc.
        """
        
        generation = None
        if self.langfuse_client:
            generation = self.langfuse_client.generation(
                name="groq_voice_command_processing",
                input={"transcription": transcription, "context": context}, # Input can be a dict
                model=self.model_name,
                metadata={"type": "voice_command_processing", "llm_provider": "groq"}
            )

        try:
            chat_completion = await self._create_chat_completion_with_retry(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                temperature=0.5, # Slightly higher temp for more natural NLU
                response_format={"type": "json_object"}
            )
            response_text = chat_completion.choices[0].message.content
            
            cleaned_response_text = response_text.strip()
            if cleaned_response_text.startswith("```json"):
                cleaned_response_text = cleaned_response_text[7:]
            if cleaned_response_text.endswith("```"):
                cleaned_response_text = cleaned_response_text[:-3]
            cleaned_response_text = cleaned_response_text.strip()

            if generation:
                generation.update(output=cleaned_response_text)
            
            try:
                parsed_response = json.loads(cleaned_response_text)
                logger.info(f"Successfully processed voice command with Groq. Output (first 200 chars): {str(parsed_response)[:200]}")
                return parsed_response
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response from Groq (voice): {e}. Raw: {response_text[:500]}, Cleaned: {cleaned_response_text[:500]}")
                if generation: generation.update(level="ERROR", status_message=f"JSONDecodeError: {e}. Raw: {response_text[:100]}")
                return {"error": "Invalid JSON response from LLM", "raw_response": response_text, "cleaned_response": cleaned_response_text}
                
        except RETRYABLE_GROQ_EXCEPTIONS as e: # Catch retryable exceptions if all retries fail
            logger.error(f"Groq API call for voice command failed after retries: {e}", exc_info=True)
            if generation: generation.end(level="ERROR", status_message=f"Groq API call failed after retries: {str(e)}")
            return {"error": f"Groq API error after retries: {e}"}
        except Exception as e:
            logger.error(f"Unexpected error during Groq voice command processing: {e}", exc_info=True)
            if generation: generation.end(level="ERROR", status_message=str(e))
            return {"error": f"Unexpected error: {e}"}

    async def attempt_text_bill_parse(self, text_input: str, user_name: str = "User") -> Optional[Dict[str, Any]]:
        """
        Attempts to parse a text input to see if it's a bill splitting instruction
        and extracts relevant details like total amount, currency, participants, and venue.
        Replaces "me" or "I" with the provided user_name.
        """
        # Replace "me" or "I" with the actual user's name for the LLM
        # This helps the LLM correctly identify the user in the participant list.
        # Using regex for case-insensitive replacement of "me" and "I" as whole words.
        # This is a simplified replacement; more sophisticated NLP might be needed for complex cases.
        processed_text_input = text_input
        # Replace " me " or " i " (with spaces)
        processed_text_input = re.sub(r'\\bme\\b', user_name, processed_text_input, flags=re.IGNORECASE)
        processed_text_input = re.sub(r'\\bI\\b', user_name, processed_text_input) # 'I' is usually capitalized

        prompt = f"""\
Analyze the following text to determine if it is a request to split a bill.
If it is, extract the total amount, currency, participants, and optionally the venue.
The user who sent this message is named '{user_name}'. If the text mentions "me" or "I", it refers to '{user_name}'.

**Input Text:**
"{processed_text_input}"

**JSON Output Structure:**
Return a JSON object with the following structure:
{{
  "is_bill_instruction": boolean, // true if the text is a bill splitting instruction, otherwise false
  "total_amount": number | null,  // The total amount of the bill as a number
  "currency": "string | null",    // Currency code (e.g., "USD", "EUR") or symbol. Infer if possible.
  "participants": ["string"] | null, // Array of participant names. Include '{user_name}' if mentioned as "me" or "I".
  "venue": "string | null"        // Name of the restaurant or store, if mentioned
}}

**Instructions for Extraction:**
1.  **is_bill_instruction:** Set to `true` only if the text clearly describes a bill to be split, including at least a total amount and participants. Casual mentions of money or spending without intent to split should be `false`.
2.  **total_amount:** Extract the primary total sum mentioned for the bill. Ensure this is a number.
3.  **currency:** Infer from context (e.g., "$", "dollars" -> "USD"; "£", "pounds" -> "GBP"; "€", "euros" -> "EUR"). If no currency is clear, use `null`.
4.  **participants:** List all unique individuals involved. If "me" or "I" is used, include '{user_name}'.
5.  **venue:** If a restaurant or store name is clearly mentioned in relation to the bill, extract it.
6.  **Accuracy:** Be precise. Monetary values must be numbers.
7.  **Missing Values:** If a field (other than `is_bill_instruction`) cannot be extracted, use `null` for its value. If `is_bill_instruction` is `false`, all other fields should ideally be `null` or omitted if the model prefers.

**Return ONLY the valid JSON object described above.**
        """

        generation_name = "groq_text_bill_parse_v1"
        if self.langfuse_client:
            generation = self.langfuse_client.generation(
                name=generation_name,
                input={"text_input": text_input, "processed_text_input": processed_text_input, "user_name": user_name},
                model=self.model_name,
                metadata={"type": "text_bill_parsing", "llm_provider": "groq"}
            )
        else:
            generation = None

        try:
            chat_completion = await self._create_chat_completion_with_retry(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                temperature=0.2, # Slightly higher for interpretation but still factual
                response_format={"type": "json_object"}
            )
            response_text = chat_completion.choices[0].message.content
            
            # Clean the response
            cleaned_response_text = response_text.strip()
            if cleaned_response_text.startswith("```json"):
                cleaned_response_text = cleaned_response_text[7:]
            if cleaned_response_text.endswith("```"):
                cleaned_response_text = cleaned_response_text[:-3]
            cleaned_response_text = cleaned_response_text.strip()

            if generation:
                generation.update(output=cleaned_response_text)

            parsed_response = json.loads(cleaned_response_text)
            logger.info(f"Groq text bill parse result: {str(parsed_response)[:200]}")

            if parsed_response.get("is_bill_instruction") is True:
                # Basic validation: ensure total_amount and participants are present if it's a bill
                if parsed_response.get("total_amount") is not None and parsed_response.get("participants"):
                    return parsed_response
                else:
                    logger.warning(f"LLM indicated bill instruction but missing total_amount or participants. Text: {text_input}")
                    if generation: generation.end(level="WARNING", status_message="Bill instruction true but key fields missing.")
                    return None # Treat as not a valid parse
            else:
                # Not a bill instruction according to the LLM
                logger.info(f"Text input not considered a bill instruction by LLM: {text_input}")
                return None

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response from Groq for text bill parse: {e}. Raw: {response_text[:500]}", exc_info=True)
            if generation: generation.end(level="ERROR", status_message=f"JSONDecodeError: {e}")
            return None
        except RETRYABLE_GROQ_EXCEPTIONS as e:
            logger.error(f"Groq API call for text bill parse failed after retries: {e}", exc_info=True)
            if generation: generation.end(level="ERROR", status_message=f"API call failed after retries: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during Groq text bill parsing: {e}", exc_info=True)
            if generation: generation.end(level="ERROR", status_message=str(e))
            return None

    async def generate_conversational_response(self, text_input: str, current_state: Optional[Dict[str, Any]] = None) -> str:
        # This is a placeholder, similar to process_receipt but tailored for voice commands
        # or general chit-chat if the bot needs to respond conversationally.
        # ... existing code ...
        pass # Add pass to make it a valid empty method for now

    # The _track_llm_call method from LLMService base is available if needed,
    # but langfuse.generation() is used here for more detailed tracing per call. 

    async def interpret_split_instructions(
        self, 
        instruction_text: str, 
        current_bill_data: Dict[str, Any], 
        conversation_history: str,
        user_name: str
    ) -> Dict[str, Any]:
        logger.info(f"[MOCK] GroqService: interpret_split_instructions for user {user_name} - '{instruction_text}'")
        # Simulate LLM processing based on instruction content for mock (same as Gemini for now)
        total_bill = float(current_bill_data.get("total", 0))
        currency = current_bill_data.get("currency", "$")

        if "equally" in instruction_text.lower() and total_bill > 0:
            participants_match = re.findall(r'between (.*) and (.*)|with (.*)', instruction_text, re.IGNORECASE)
            participant_names = [name for group in participants_match for name in group if name and name.lower() != "me"]
            if "me" in instruction_text.lower() or not participant_names: participant_names.append(user_name)
            participant_names = list(set(name.strip().capitalize() for name in participant_names if name.strip()))
            if not participant_names: participant_names = [user_name, "Friend 1"]
            
            num_participants = len(participant_names)
            amount_per_person = round(total_bill / num_participants, 2) if num_participants > 0 else total_bill
            return {
                "total": total_bill, "currency": currency,
                "breakdown": {name: amount_per_person for name in participant_names},
                "is_final": True,
                "summary_text": f"Split {total_bill}{currency} equally: {amount_per_person}{currency} each."
            }
        elif "saquib will pay 50%" in instruction_text.lower() and total_bill > 0:
            saquib_pays = round(total_bill * 0.50, 2); remaining = round(total_bill - saquib_pays, 2)
            me_pays = round(remaining / 2, 2); praneet_pays = round(remaining / 2, 2)
            return {
                "total": total_bill, "currency": currency,
                "breakdown": {"Saquib": saquib_pays, user_name: me_pays, "Praneet": praneet_pays},
                "is_final": True,
                "summary_text": f"Total: {total_bill}{currency}. Saquib: {saquib_pays}{currency}. Others: {me_pays}{currency} each."
            }
        else:
            return {"is_final": False, "clarification_needed": "I couldn't parse that split. Try simpler terms?"}

    async def apply_split_adjustment(
        self, 
        adjustment_text: str, 
        current_split_data: Dict[str, Any],
        current_bill_data: Dict[str, Any],
        conversation_history: str,
        user_name: str
    ) -> Optional[Dict[str, Any]]:
        logger.info(f"[MOCK] GroqService: apply_split_adjustment for user {user_name} - '{adjustment_text}'")
        return None # Mock: always fails to adjust automatically 