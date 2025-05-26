import logging
from typing import Optional, Tuple, Any, Dict, List
import re # For parsing item assignments
import json # For serializing current_bill_json for LLM

from core.state_manager import ConversationState
from services.llm.base import LLMService
from services.ocr.base import OCRService
# from monitoring.langfuse_client import LangfuseMonitor # langfuse_client (SDK) is passed

logger = logging.getLogger(__name__)

# Define conversation states
STATE_AWAITING_BILL_IMAGE = "AWAITING_BILL_IMAGE"
STATE_PROCESSING_BILL_IMAGE = "PROCESSING_BILL_IMAGE" # Transient
STATE_BILL_IMAGE_PROCESSED_CONFIRM_OCR = "BILL_IMAGE_PROCESSED_CONFIRM_OCR"
STATE_EXTRACTING_BILL_DETAILS = "EXTRACTING_BILL_DETAILS" # Transient
STATE_BILL_DETAILS_EXTRACTED_CONFIRM_ITEMS = "BILL_DETAILS_EXTRACTED_CONFIRM_ITEMS"
STATE_AWAITING_PARTICIPANTS = "AWAITING_PARTICIPANTS"
STATE_AWAITING_ITEM_ASSIGNMENTS = "AWAITING_ITEM_ASSIGNMENTS"
STATE_CALCULATING_SPLIT = "CALCULATING_SPLIT" # Transient
STATE_SPLIT_CALCULATED_SHOW_SUMMARY = "SPLIT_CALCULATED_SHOW_SUMMARY"
STATE_SPLIT_FINALIZED = "SPLIT_FINALIZED"
STATE_AWAITING_CORRECTION_OCR = "AWAITING_CORRECTION_OCR"
STATE_AWAITING_CORRECTION_ITEMS = "AWAITING_CORRECTION_ITEMS"

# Mapping of human-readable state descriptions
STATE_DESCRIPTIONS = {
    STATE_AWAITING_BILL_IMAGE: "I'm waiting for you to send a bill image.",
    STATE_PROCESSING_BILL_IMAGE: "I'm currently processing the bill image you sent.",
    STATE_BILL_IMAGE_PROCESSED_CONFIRM_OCR: "I've read the text from the bill. I'll show you what I found next, or you can tell me if the image was bad.",
    STATE_EXTRACTING_BILL_DETAILS: "I'm analyzing the details from the bill text.",
    STATE_BILL_DETAILS_EXTRACTED_CONFIRM_ITEMS: "I've extracted items from the bill. Please review them and confirm, or tell me who participated.",
    STATE_AWAITING_PARTICIPANTS: "Please tell me who participated in this bill.",
    STATE_AWAITING_ITEM_ASSIGNMENTS: "Please assign items to participants, or tell me if items are shared.",
    STATE_CALCULATING_SPLIT: "I'm calculating the split now.",
    STATE_SPLIT_CALCULATED_SHOW_SUMMARY: "I've calculated the split. Please review the summary.",
    STATE_SPLIT_FINALIZED: "The bill split is finalized! Ready for a new one when you are.",
    STATE_AWAITING_CORRECTION_OCR: "It seems there might have been an issue with reading the bill. You can send a new image or describe the corrections.",
    STATE_AWAITING_CORRECTION_ITEMS: "Okay, I'm ready for you to tell me what needs to be corrected with the items."
}

class BillSplitProcessor:
    def __init__(self, 
                 llm_service: LLMService, 
                 ocr_service: OCRService, 
                 state_manager: ConversationState,
                 langfuse_client: Optional[Any]): # Langfuse SDK client
        self.llm_service = llm_service
        self.ocr_service = ocr_service
        self.state_manager = state_manager
        self.langfuse_client = langfuse_client
        logger.info("BillSplitProcessor initialized.")

    async def process_command(self, user_id: str, interface_type: str, command: str) -> str:
        """Handles direct commands like /start, /cancel, /status."""
        current_state_tuple = await self.state_manager.get_state(user_id)
        current_state_data = current_state_tuple[0] if current_state_tuple else None # Get data from tuple

        if not current_state_data: # Should ideally not happen if interface calls this after ensuring user exists
            # If state truly doesn't exist, initialize it.
            logger.info(f"No state found for user {user_id} during command processing. Initializing.")
            current_state_data = {"step": STATE_AWAITING_BILL_IMAGE, "history": [], "user_id": user_id}
            # Await state update before proceeding
            await self.state_manager.update_state(user_id, current_state_data) 
        
        current_step = current_state_data.get("step", STATE_AWAITING_BILL_IMAGE)
        response_message = f"Received command: {command}"
        next_step = current_step

        logger.info(f"User {user_id} sent command '{command}' in state '{current_step}'")
        span = None
        if self.langfuse_client:
            span = self.langfuse_client.span(name=f"process_command_{command.replace('/','')}", input={"user_id": user_id, "current_step": current_step})

        if command == "/start" or command == "/split":
            # Reset state for a new bill splitting process
            current_state_data.clear() # Clear all old data
            current_state_data["user_id"] = user_id
            current_state_data["history"] = [] # Optionally keep history or start fresh
            next_step = STATE_AWAITING_BILL_IMAGE
            response_message = "Okay, let's start a new bill! Please send me a photo of the bill."
            logger.info(f"State reset for user {user_id} due to {command} command.")
        elif command == "/cancel":
            # Similar to /start, cancel current operation and reset
            # Check if there is an active non-initial state to cancel
            if current_step != STATE_AWAITING_BILL_IMAGE and current_step != STATE_SPLIT_FINALIZED:
                response_message = "Okay, I've cancelled the current bill splitting operation. Let's start over. Please send me a new bill photo."
            else:
                response_message = "There's nothing active to cancel. Let's start a new bill! Please send me a photo."
            current_state_data.clear()
            current_state_data["user_id"] = user_id
            current_state_data["history"] = [] 
            next_step = STATE_AWAITING_BILL_IMAGE
            logger.info(f"Operation cancelled and state reset for user {user_id}.")
        elif command == "/status":
            status_description = STATE_DESCRIPTIONS.get(current_step, "I'm currently in an unknown state.")
            response_message = f"Current status: {status_description}"
            if current_step == STATE_BILL_DETAILS_EXTRACTED_CONFIRM_ITEMS and "parsed_bill" in current_state_data:
                response_message += "\nI'm waiting for you to confirm the extracted items or tell me who participated."
            elif current_step == STATE_AWAITING_ITEM_ASSIGNMENTS and "participants" in current_state_data:
                response_message += f"\nI have the participants as: { ', '.join(current_state_data['participants']) }. Now waiting for item assignments."
            # next_step remains current_step
        else:
            logger.warning(f"Unknown command '{command}' received for user {user_id}.")
            response_message = "Sorry, I don't understand that command. Try /help for available commands."
            # next_step remains current_step

        current_state_data["step"] = next_step
        # Log command processing in history
        history_entry = {
            "input_command": command,
            "response": response_message,
            "step_before": current_step,
            "step_after": next_step
        }
        if "history" not in current_state_data or not isinstance(current_state_data["history"], list):
            current_state_data["history"] = []
        current_state_data["history"].append(history_entry)

        await self.state_manager.update_state(user_id, current_state_data)
        if span: span.end(output={"response": response_message, "next_step": next_step})
        return response_message

    async def process_message(self, user_id: str, interface_type: str, 
                              current_state_data: Dict[str, Any], # Use passed state data
                              message_text: Optional[str] = None, 
                              image_bytes: Optional[bytes] = None) -> str:

        # Ensure user_id is in state_data for future use, if somehow missing (should be guaranteed by caller)
        if "user_id" not in current_state_data:
            current_state_data["user_id"] = user_id

        current_step = current_state_data.get("step", STATE_AWAITING_BILL_IMAGE)
        # logger.info(f"User {user_id} current step: {current_step}. Input: text='{message_text if message_text else "/N/A/"}', image={'provided' if image_bytes else 'not provided'}")
        logger.info(
            "Processing user message",
            extra={
                "user_id": user_id,
                "current_step": current_step,
                "input": {
                    "text": message_text if message_text else None,
                    "has_image": bool(image_bytes)
                }
            }
        )
        response_message = "Sorry, I encountered an issue." # Default error
        next_step = current_step # By default, stay in current step unless changed

        try:
            if image_bytes:
                # Regardless of current text step, an image usually means a new bill or correction
                logger.info(f"Processing image for user {user_id}.")
                # Create a Langfuse span for image handling if client is available
                span = None
                if self.langfuse_client:
                    span = self.langfuse_client.span(name="handle_image_input_processor", input={"user_id": user_id, "current_step": current_step})
                
                response_message, next_step = await self._handle_image_input(user_id, image_bytes, current_state_data)
                if span: span.end(output={"response": response_message, "next_step": next_step})
            elif message_text:
                span = None
                if self.langfuse_client:
                    span = self.langfuse_client.span(name="handle_text_input_processor", input={"user_id": user_id, "text": message_text, "current_step": current_step})

                logger.info(f"Processing text message for user {user_id}: '{message_text}'")
                response_message, next_step = await self._handle_text_input(user_id, message_text, current_state_data)
                if span: span.end(output={"response": response_message, "next_step": next_step})
            else:
                logger.warning(f"No actionable input (text or image) for user {user_id} in step {current_step}.")
                if current_step == STATE_AWAITING_BILL_IMAGE:
                    response_message = "Please send me a photo of the bill you want to split."
                else:
                    response_message = "I didn't receive any message content. Could you please try again or tell me what you'd like to do?"
                # next_step remains current_step

            # Update state if it has changed or if there's new history
            current_state_data["step"] = next_step
            # Simple history for now. Could be more structured.
            history_entry = {
                "input_text": message_text,
                "input_image": bool(image_bytes),
                "response": response_message,
                "step_before": current_step,
                "step_after": next_step
            }
            if "history" not in current_state_data or not isinstance(current_state_data["history"], list):
                current_state_data["history"] = []
            current_state_data["history"].append(history_entry)
            await self.state_manager.update_state(user_id, current_state_data)

        except Exception as e:
            logger.error(f"Error processing message for user {user_id} in step {current_step}: {e}", exc_info=True)
            response_message = "An unexpected error occurred. Please try again or type /start to begin anew."
            # Reset to a safe state or try to infer
            current_state_data["step"] = STATE_AWAITING_BILL_IMAGE # Fallback to initial state on unhandled error
            await self.state_manager.update_state(user_id, current_state_data)
            # Langfuse trace (created in TelegramInterface) will capture this error if it bubbles up or is re-raised.

        return response_message

    async def _handle_image_input(self, user_id: str, image_bytes: bytes, state_data: Dict[str, Any]) -> Tuple[str, str]:
        current_step = state_data["step"]
        logger.info(f"Performing direct image processing with LLM for user {user_id} from step {current_step}.")
        state_data["step"] = STATE_PROCESSING_BILL_IMAGE # Mark as processing (though it's one step now with LLM)
        
        # Old OCR-based logic (commented out):
        # logger.info(f"Performing OCR for user {user_id} from step {current_step}.")
        # # state_data["step"] = STATE_PROCESSING_BILL_IMAGE # Mark as processing (original position)
        # ocr_span = None
        # if self.langfuse_client:
        #     # Assuming OCR service methods are decorated, so this span is for the orchestration part.
        #     ocr_span = self.langfuse_client.span(name="ocr_orchestration", input={"image_size_bytes": len(image_bytes)})
        # try:
        #     extracted_text = await self.ocr_service.extract_text(image_bytes)
        #     if ocr_span: ocr_span.update(output={"text_extracted_length": len(extracted_text) if extracted_text else 0})

        #     if not extracted_text:
        #         logger.warning(f"OCR returned no text for user {user_id}.")
        #         if ocr_span: ocr_span.end(output={"status": "ocr_no_text"})
        #         return "I couldn't extract any text from the image. Please try a clearer picture or check the image format.", STATE_AWAITING_BILL_IMAGE

        #     logger.info(f"OCR successful for user {user_id}. Text length: {len(extracted_text)}")
        #     state_data["ocr_text"] = extracted_text
        #     state_data["original_image_hash"] = hash(image_bytes) # Store hash to detect re-submissions or edits
        #     if ocr_span: ocr_span.end(output={"status": "ocr_success"})

        #     # Transition to LLM-based extraction
        #     state_data["step"] = STATE_EXTRACTING_BILL_DETAILS 
        #     response_msg, next_process_step = await self._process_extracted_ocr_text(user_id, extracted_text, state_data)
        #     return response_msg, next_process_step
        # except Exception as e:
        #     logger.error(f"OCR processing or subsequent step failed for user {user_id}: {e}", exc_info=True)
        #     if ocr_span: ocr_span.end(level="ERROR", status_message=str(e))
        #     return "There was an error processing the image via OCR. Please try again.", current_step # Revert to original step before image processing

        # New direct LLM image processing logic:
        llm_image_span = None
        if self.langfuse_client:
            llm_image_span = self.langfuse_client.span(
                name="llm_direct_image_bill_processing", 
                input={"image_size_bytes": len(image_bytes), "user_id": user_id}
            )

        try:
            # Directly call LLM for image processing if the method exists
            if not hasattr(self.llm_service, 'process_receipt_from_image'):
                logger.error("LLM service does not have 'process_receipt_from_image' method. Falling back to OCR or error.")
                if llm_image_span: llm_image_span.end(level="ERROR", status_message="LLM service missing process_receipt_from_image")
                # Fallback to old OCR method if you want, or just error out
                # For this change, we'll assume we want to error if the new method isn't there
                return "My apologies, I'm not equipped to process images directly right now. Please try describing the bill.", STATE_AWAITING_BILL_IMAGE

            # Assuming image_bytes is JPEG, common for uploads. Adjust mime_type if needed.
            parsed_bill_data = await self.llm_service.process_receipt_from_image(image_bytes, image_mime_type="image/jpeg")

            if llm_image_span:
                llm_image_span.update(output=parsed_bill_data)

            if not parsed_bill_data or parsed_bill_data.get("error"):
                error_detail = parsed_bill_data.get('error', 'No details provided by LLM.')
                raw_resp_preview = str(parsed_bill_data.get('raw_response', ''))[:200]
                logger.warning(f"LLM direct image processing failed for user {user_id}. Error: {error_detail}. Raw preview: {raw_resp_preview}")
                if llm_image_span: llm_image_span.end(output={"status": "llm_image_processing_failed", "error": error_detail})
                # Provide a more user-friendly message
                user_message = "I had trouble understanding the details from the bill image directly."
                if "timeout" in error_detail.lower() or "deadline" in error_detail.lower():
                    user_message += " It might have been too complex or taken too long. Could you try a simpler bill or a clearer image?"
                else:
                    user_message += " Please try sending a clearer picture, or perhaps describe the main items and total."
                return user_message, STATE_AWAITING_BILL_IMAGE

            # Store the directly parsed bill data
            state_data["parsed_bill"] = parsed_bill_data
            state_data["original_image_hash"] = hash(image_bytes) # Keep hash for consistency
            # No separate OCR text to store now, unless LLM provides it as part of its thinking (not standard)
            # state_data.pop("ocr_text", None) # Remove old ocr_text if it exists

            logger.info(f"LLM successfully parsed bill from image for user {user_id}. Items: {len(parsed_bill_data.get('items', []))}")
            if llm_image_span: llm_image_span.end(output={"status": "llm_image_processing_success"})
            
            response_msg = self._format_extracted_bill_for_confirmation(parsed_bill_data)
            return response_msg, STATE_BILL_DETAILS_EXTRACTED_CONFIRM_ITEMS

        except Exception as e:
            logger.error(f"Direct image processing with LLM failed for user {user_id}: {e}", exc_info=True)
            if llm_image_span: llm_image_span.end(level="ERROR", status_message=str(e))
            return "There was an unexpected error processing the bill image. Please try again.", current_step # Revert to step before image input

    def _format_extracted_bill_for_confirmation(self, parsed_bill: Dict[str, Any]) -> str:
        """Formats the extracted bill details into a human-readable string for confirmation."""
        if not parsed_bill or not isinstance(parsed_bill, dict):
            return "I couldn't find any details to show."

        lines = []
        if "restaurant_name" in parsed_bill and parsed_bill["restaurant_name"]:
            lines.append(f"Restaurant: {parsed_bill['restaurant_name']}")
        
        items = parsed_bill.get("items", [])
        if items:
            lines.append("Items found:")
            for i, item in enumerate(items):
                if isinstance(item, dict):
                    price_str = f" {item.get('price')}" if item.get('price') is not None else f" {item.get('total_price')}" if item.get('total_price') is not None else " (price not found)"
                    lines.append(f"  {i+1}. {item.get('name', 'Unknown Item')}{price_str}")
                else:
                    lines.append(f"  {i+1}. {str(item)} (unexpected format)")
        else:
            lines.append("No specific items were identified.")

        total = parsed_bill.get("total")
        if total is not None:
            lines.append(f"Total Amount: {total} {parsed_bill.get('currency', '')}".strip())
        else:
            lines.append("Total amount not clearly identified.")
        
        confirmation_prompt = "\n\nIs this information correct? You can say:"
        confirmation_prompt += "\n- 'Yes' to confirm."
        confirmation_prompt += "\n- 'List participants' or 'Add [Name1], [Name2]' to tell me who was part of this bill."
        confirmation_prompt += "\n- 'Correct [item number] to [new name] [new price]' to fix an item."
        confirmation_prompt += "\n- Or send a new image if this is completely wrong."
        
        return "\n".join(lines) + confirmation_prompt

    async def _process_extracted_ocr_text(self, user_id: str, ocr_text: str, state_data: Dict[str, Any]) -> Tuple[str, str]:
        """This function is called after OCR is successful. It calls LLM to parse the bill.
           THIS METHOD WILL BE BYPASSED if direct image processing with LLM is used.
        """
        logger.info(f"Calling LLM to parse OCR text for user {user_id}. Text length: {len(ocr_text)}")
        state_data["step"] = STATE_EXTRACTING_BILL_DETAILS # Explicitly set, though might be set by caller

        llm_generation = None
        if self.langfuse_client:
            llm_generation = self.langfuse_client.generation(
                name="llm_parse_ocr_bill", 
                input={"ocr_text_length": len(ocr_text), "ocr_text_preview": ocr_text[:200]},
                model=self.llm_service.model_name if hasattr(self.llm_service, 'model_name') else 'unknown', # Add model if available
                metadata={"user_id": user_id, "current_step_before_llm": state_data.get("step")}
            )

        try:
            # Assuming llm_service has a method like `process_receipt` (as seen in GroqService)
            parsed_bill_data = await self.llm_service.process_receipt(ocr_text)
            if llm_generation: llm_generation.update(output=parsed_bill_data)

            if parsed_bill_data.get("error"):
                logger.error(f"LLM failed to parse bill for user {user_id}: {parsed_bill_data.get('error')}")
                if llm_generation: llm_generation.end(level="WARNING", status_message=f"LLM parsing failed: {parsed_bill_data.get('error')}")
                # Fallback to asking user to check OCR text, as LLM couldn't make sense of it
                response_message = f"I understood the text from the bill but had trouble interpreting the specific items and amounts. Here's a snippet of what I read: '{ocr_text}...'\n\nWould you like to try sending a clearer image, or tell me the items and total manually?"
                return response_message, STATE_BILL_IMAGE_PROCESSED_CONFIRM_OCR
            
            state_data["parsed_bill"] = parsed_bill_data
            # Example: state_data["parsed_bill"] = {"items": [{"name": "Pizza", "price": 15.99}], "total": 15.99, "restaurant_name": "Pizza Place"}
            logger.info(f"LLM successfully parsed bill for user {user_id}. Items: {len(parsed_bill_data.get('items', []))}")
            
            response = self._format_extracted_bill_for_confirmation(parsed_bill_data)
            return response, STATE_BILL_DETAILS_EXTRACTED_CONFIRM_ITEMS

        except Exception as e:
            logger.error(f"LLM bill parsing failed for user {user_id}: {e}", exc_info=True)
            if llm_generation: llm_generation.end(level="ERROR", status_message=str(e))
            # Fallback if LLM service itself fails
            response_message = f"I encountered an issue while trying to understand the bill details. Here's what I read from the image: '{ocr_text}...'\n\nCould you review this and tell me the items and total, or try sending the image again?"
            return response_message, STATE_BILL_IMAGE_PROCESSED_CONFIRM_OCR

    def _parse_item_assignment_text(self, text: str, items: List[Dict[str, Any]], participants: List[str]) -> Optional[Dict[str, Any]]:
        """Rudimentary parsing of item assignment text. E.g., 'Item 1 for Alice', 'Item 2 for Bob, Me', 'Everyone shares Item 3'"""
        # TODO: Replace with LLM-based intent and slot filling for robustness.
        text_lower = text.lower()
        assignment = {"item_indices": [], "users": [], "shared_by_all": False}

        # Find item numbers/names
        item_indices_found = []
        for i, item_data in enumerate(items):
            item_name_lower = item_data.get("name", "").lower()
            if f"item {i+1}" in text_lower or (item_name_lower and item_name_lower in text_lower):
                item_indices_found.append(i)
        if not item_indices_found:
            # Try to find numbers that might be item indices
            found_numbers = re.findall(r'\d+', text_lower)
            for num_str in found_numbers:
                try:
                    idx = int(num_str) - 1
                    if 0 <= idx < len(items):
                        item_indices_found.append(idx)
                except ValueError:
                    continue
        assignment["item_indices"] = list(set(item_indices_found))
        if not assignment["item_indices"] : return None # No item identified

        # Find users
        assigned_users = []
        if "everyone shares" in text_lower or "shared by all" in text_lower:
            assignment["shared_by_all"] = True
            assigned_users = participants # Assign to all participants
        else:
            for user in participants:
                if user.lower() in text_lower:
                    assigned_users.append(user)
            if not assigned_users and "me" in text_lower and "Me" in participants: # Handle "me" specifically if not typed as participant name
                assigned_users.append("Me")
        
        assignment["users"] = list(set(assigned_users))
        if not assignment["users"] and not assignment["shared_by_all"]: return None # No user identified and not shared by all

        return assignment

    def _format_split_summary(self, calculated_split: Dict[str, float], currency: Optional[str]) -> str:
        summary_lines = ["Here's the split:"]
        currency_symbol = currency if currency else ""
        for user, amount in calculated_split.items():
            summary_lines.append(f"- {user}: {amount:.2f} {currency_symbol}".strip())
        summary_lines.append("\nIs this okay, or do you want to make changes?")
        return "\n".join(summary_lines)

    async def _handle_text_input(self, user_id: str, text: str, state_data: Dict[str, Any]) -> Tuple[str, str]:
        current_step = state_data.get("step", STATE_AWAITING_BILL_IMAGE)
        logger.info(f"Handling text '{text}' for user {user_id} at step: {current_step}")
        next_step = current_step
        response_message = "I'm not sure how to respond to that. Please try again."
        user_name = state_data.get("user_name", "User") # Get user_name from state, default to "User"
                                                    # This assumes user_name might be set during /start or initial interaction

        if current_step == STATE_AWAITING_BILL_IMAGE:
            if hasattr(self.llm_service, 'attempt_text_bill_parse'):
                logger.info(f"Attempting to parse text as bill instruction for user {user_id}")
                # Pass user_name which might have been stored in state_data previously, e.g. from Telegram user profile
                # For simplicity, if not in state, use a generic "User". Ideally, user's first name is better.
                parsed_text_bill = await self.llm_service.attempt_text_bill_parse(text, user_name=user_name)

                if parsed_text_bill and parsed_text_bill.get("is_bill_instruction"):
                    logger.info(f"Text parsed as bill instruction: {parsed_text_bill}")
                    state_data["parsed_bill"] = {
                        "restaurant_name": parsed_text_bill.get("venue"),
                        "items": [{
                            "name": "Total bill (from text input)", 
                            "quantity": 1,
                            "total_price": parsed_text_bill.get("total_amount")
                        }],
                        "total": parsed_text_bill.get("total_amount"),
                        "currency": parsed_text_bill.get("currency")
                    }
                    # Store participants directly if LLM extracts them
                    if parsed_text_bill.get("participants"):
                        state_data["participants"] = list(set(parsed_text_bill.get("participants")))
                    
                    confirmation_text = self._format_extracted_bill_for_confirmation(state_data["parsed_bill"])
                    if state_data.get("participants"):
                        confirmation_text += f"\nParticipants identified: {', '.join(state_data['participants'])}."
                    else:
                        confirmation_text += "\nI couldn't identify all participants clearly from your message."
                    
                    confirmation_text += "\n\nIs this correct? Or you can tell me who participated if that's missing/wrong."
                    response_message = confirmation_text
                    next_step = STATE_BILL_DETAILS_EXTRACTED_CONFIRM_ITEMS
                else:
                    # LLM did not parse it as a bill instruction, or parsing failed
                    response_message = "Hello! I'm ready when you are. Please send me a picture of the bill, or describe the bill you want to split (e.g., 'Split $50 with John and Jane')."
                    next_step = STATE_AWAITING_BILL_IMAGE
            else:
                # Fallback if LLM service does not have the new method
                response_message = "Hello! I'm ready when you are. Please send me a picture of the bill."
                next_step = STATE_AWAITING_BILL_IMAGE
        
        elif current_step == STATE_BILL_IMAGE_PROCESSED_CONFIRM_OCR:
            # User is responding after seeing raw OCR text because structured parsing failed.
            # We need to interpret their confirmation or correction attempt.
            # Simple approach: If they say yes/correct, ask for manual details.
            # More advanced: Use LLM to understand their response to the raw OCR view.

            text_lower = text.lower()
            if text_lower in ["yes", "correct", "ok", "looks good", "this seems correct", "that's right"]:
                logger.info(f"User confirmed OCR text view for user {user_id}, but structured parsing failed. Asking for manual input.")
                # We can't re-process the same OCR text and expect a different structured outcome without new info.
                response_message = "Okay. Since I had a bit of trouble pinpointing all the details automatically, could you please tell me:\n1. Who participated in this bill?\n2. What was the total amount?"
                # We need a new state to await this manual info, or reuse AWAITING_PARTICIPANTS and then figure out total.
                # For now, let's guide them to provide participants, then we can ask for total if needed.
                # Or, create a new state like STATE_AWAITING_MANUAL_BILL_DETAILS
                state_data["ocr_confirmed_but_parsing_failed"] = True # Mark this path
                # Let's try to get participants first.
                next_step = STATE_AWAITING_PARTICIPANTS 
                # We could also add a more specific prompt to the user here like: 
                # "Please list the participants first (e.g., Alice, Bob, Me)."
            elif "new image" in text_lower or "send again" in text_lower or "try again" in text_lower:
                logger.info(f"User wants to send a new image after failed OCR parse for user {user_id}.")
                response_message = "Sure, please send the new bill image."
                next_step = STATE_AWAITING_BILL_IMAGE
            else:
                # User might be trying to correct the OCR text, or give instructions.
                # This is where it gets tricky without a more sophisticated NLU step here.
                # For now, let's assume they might be trying to give participant info or total.
                # This part can be made smarter with another LLM call to interpret this response.
                logger.warning(f"User {user_id} sent ambiguous text '{text}' in STATE_BILL_IMAGE_PROCESSED_CONFIRM_OCR. Defaulting to ask for new image or manual.")
                response_message = "I'm a bit stuck on that last bill. Would you like to send a new, clearer image, or tell me the participants and total manually?"
                # Keep in this state or revert to awaiting image, as re-processing OCR is unlikely to help.
                next_step = STATE_AWAITING_BILL_IMAGE # Or current_step, but awaiting image seems safer.

        elif current_step == STATE_BILL_DETAILS_EXTRACTED_CONFIRM_ITEMS:
            # This is where user confirms items, asks for corrections, or says who participated.
            # This requires more sophisticated NLU with LLM.
            # Example: "Yes", "No, item 2 is wrong", "John, Mary, and I participated"
            # For now, a very simple placeholder:
            if text.lower() in ["yes", "correct", "ok", "looks good"]:
                response_message = "Great! Who participated in this bill? Please list their names separated by commas (e.g., Alice, Bob, Me)."
                next_step = STATE_AWAITING_PARTICIPANTS
            elif "list participants" in text.lower() or "who participated" in text.lower() or "add" in text.lower() or "," in text: # Basic cues for participant listing
                # Attempt to extract participants
                # This needs a proper LLM call for robust extraction.
                # Placeholder extraction:
                raw_participant_text = text.lower().replace("list participants", "").replace("who participated", "").replace("add", "").strip()
                participants = [p.strip().capitalize() for p in raw_participant_text.split(",") if p.strip()]
                if not participants and "me" in raw_participant_text:
                    participants.append("Me") # Add self if mentioned and no others
                
                if participants:
                    state_data["participants"] = list(set(participants)) # Store unique participants
                    logger.info(f"Participants for user {user_id}: {state_data['participants']}")
                    response_message = f"Got it. Participants: {', '.join(state_data['participants'])}. Next, we'll assign items. Or you can say 'everyone shares [item name]' or '[item name] for [person]'."
                    next_step = STATE_AWAITING_ITEM_ASSIGNMENTS # Or a more specific item assignment state
                else:
                    response_message = "I didn't quite catch the names. Please list participants separated by commas (e.g., Alice, Bob, Charlie)."
                    next_step = STATE_AWAITING_PARTICIPANTS # Stay or go back to confirm items if that makes more sense
            # TODO: Add logic for item correction based on user input, e.g., "Correct item 1 to X price Y"
            else:
                response_message = "Sorry, I didn't understand that. You can say 'Yes' if the items are correct, list participants, or tell me what to correct."
                # next_step remains STATE_BILL_DETAILS_EXTRACTED_CONFIRM_ITEMS

        elif current_step == STATE_AWAITING_PARTICIPANTS:
            # User provides names of participants
            # TODO: Use LLM to robustly extract names from text
            raw_participant_text = text.lower().strip()
            participants = [p.strip().capitalize() for p in raw_participant_text.split(",") if p.strip()]
            if not participants and "me" in raw_participant_text: participants.append("Me")
            
            if participants:
                state_data["participants"] = list(set(participants))
                logger.info(f"Participants for user {user_id}: {state_data['participants']}")
                response_message = f"Thanks! Participants: {', '.join(state_data['participants'])}. Now, how should we assign items? (e.g., 'Item 1 for Alice', 'everyone shares Item 3')"
                next_step = STATE_AWAITING_ITEM_ASSIGNMENTS
            else:
                response_message = "Please list the names of participants separated by commas."
                # next_step remains STATE_AWAITING_PARTICIPANTS
        
        elif current_step == STATE_AWAITING_ITEM_ASSIGNMENTS:
            # TODO: LLM to parse item assignments, e.g. "Item 1 for Alice", "Item 2 for Bob and Me", "Everyone shares item 3"
            # For now, just acknowledge and set to a placeholder for calculated split.
            logger.info(f"User {user_id} provided item assignment text: {text}")
            state_data["item_assignment_text"] = state_data.get("item_assignment_text", "") + text + "\n" 
            # This is a placeholder: actual assignment logic is complex.
            # if text.lower() == "done assigning":
            #    response_message = "Okay, calculating the split..."
            #    next_step = STATE_CALCULATING_SPLIT
            # else:
            response_message = f"Got it: '{text}'. Tell me more assignments or say 'Done assigning' when finished."
            next_step = STATE_AWAITING_ITEM_ASSIGNMENTS # Stay until user says done or all items assigned

        elif current_step == STATE_SPLIT_CALCULATED_SHOW_SUMMARY:
            if text.lower() in ["yes", "ok", "correct", "looks good", "finalize", "y"]:
                response_message = "Great! The bill split is finalized. Let me know if you have another bill!"
                next_step = STATE_SPLIT_FINALIZED
                # Optionally, clear state or parts of it for a new bill
                # await self.state_manager.clear_state(user_id) # Or just reset relevant fields
                state_data_new = {"step": STATE_AWAITING_BILL_IMAGE, "history": state_data.get("history",[]), "user_id": user_id} # Keep history
                state_data.clear()
                state_data.update(state_data_new)
            elif text.lower() in ["no", "changes", "adjust", "n"]:
                response_message = "Okay, what would you like to change? You can tell me about item assignments or participants."
                # TODO: More sophisticated jump back to correction states for items or participants
                next_step = STATE_AWAITING_ITEM_ASSIGNMENTS # Go back to assignments for now
            else:
                response_message = "Did you want to finalize this split or make changes? (Yes/No/Make changes)"
        
        else:
            logger.warning(f"Unhandled text input for step '{current_step}' for user {user_id}.")
            # Fallback to LLM for general response if no specific logic for the state
            try:
                logger.info(f"Falling back to LLM for generic response for user {user_id} at step {current_step}")
                # This requires a generic conversational method in LLMService
                # response_message = await self.llm_service.generate_conversational_response(text, state_data)
                # For now, simple fallback:
                response_message = "I'm a bit lost. Could you try sending the bill image again to restart that part?"
                next_step = STATE_AWAITING_BILL_IMAGE # Default to restart
            except Exception as llm_e:
                logger.error(f"LLM fallback failed for user {user_id}: {llm_e}", exc_info=True)
                response_message = "I'm a bit confused. Try sending a bill image?"
                next_step = STATE_AWAITING_BILL_IMAGE

        return response_message, next_step

    # Placeholder for future methods related to splitting logic
    async def _assign_items_to_users(self, user_id: str, assignments: dict, state_data: dict) -> str:
        pass

    async def _calculate_totals(self, user_id: str, state_data: dict) -> dict:
        pass 

    async def _calculate_and_present_split(self, user_id: str, state_data: Dict[str, Any]) -> Tuple[str, str]:
        state_data["step"] = STATE_CALCULATING_SPLIT
        logger.info(f"Calculating split for user {user_id}")
        
        parsed_bill = state_data.get("parsed_bill", {})
        items = parsed_bill.get("items", [])
        participants = state_data.get("participants", [])
        assignments = state_data.get("assignments", {}) # item_index (int) -> list of user names OR "total_amount" -> list of user names
        currency = parsed_bill.get("currency")

        if not participants:
            logger.warning(f"No participants to calculate split for user {user_id}")
            return "I don't have any participants to split the bill for. Please list who participated.", STATE_AWAITING_PARTICIPANTS

        # Initialize individual totals
        individual_totals: Dict[str, float] = {p: 0.0 for p in participants}
        processed_any_assignments = False

        if not items and "total_amount" in assignments and parsed_bill.get("total") is not None:
            # Case: No items, split total amount equally
            total_amount = float(parsed_bill.get("total", 0))
            if assignments["total_amount"] == participants and len(participants) > 0: # Ensure all listed participants share
                share = total_amount / len(participants)
                for p in participants:
                    individual_totals[p] += share
                processed_any_assignments = True
                logger.info(f"Splitting total amount {total_amount} among {len(participants)} participants.")
            else:
                logger.warning(f"Mismatch in participants for total split or no participants for user {user_id}")
                return "Could not split total: participant list mismatch or empty.", STATE_AWAITING_PARTICIPANTS
        elif items:
            # Case: Split based on item assignments
            for item_idx, item_data in enumerate(items):
                item_price = float(item_data.get("price", 0.0))
                assigned_to = assignments.get(item_idx)
                
                if assigned_to and isinstance(assigned_to, list) and len(assigned_to) > 0:
                    share_per_person = item_price / len(assigned_to)
                    for user_assigned in assigned_to:
                        if user_assigned in individual_totals:
                            individual_totals[user_assigned] += share_per_person
                            processed_any_assignments = True
                        else:
                            logger.warning(f"User '{user_assigned}' assigned to item {item_idx} but not in participant list {participants}")
                else:
                    # Item not assigned or invalid assignment - for now, we ignore it or could add to a common pool if desired
                    logger.warning(f"Item {item_idx} ('{item_data.get('name')}') has no valid assignment, price {item_price} not split.")
        
        if not processed_any_assignments and not (not items and "total_amount" in assignments):
            logger.warning(f"No items were assigned or total split for user {user_id}. Cannot calculate split.")
            return "It seems no items were assigned or I couldn't process the assignments. Please review item assignments.", STATE_AWAITING_ITEM_ASSIGNMENTS

        state_data["calculated_split"] = individual_totals
        logger.info(f"Calculated split for user {user_id}: {individual_totals}")
        
        summary_message = self._format_split_summary(individual_totals, currency)
        return summary_message, STATE_SPLIT_CALCULATED_SHOW_SUMMARY 

    def _apply_bill_corrections(self, original_bill: Dict[str, Any], corrections: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Applies corrections to a parsed bill dictionary. Returns updated bill and notes on changes."""
        if not isinstance(original_bill, dict) or not isinstance(corrections, dict):
            return original_bill, ["Invalid input for applying corrections."]
        
        updated_bill = json.loads(json.dumps(original_bill)) # Deep copy
        change_notes = [] # To inform user about what was changed

        # Apply top-level changes
        for key in ["restaurant_name", "transaction_date", "transaction_time", "currency", "subtotal", "tax", "tip", "total"]:
            if key in corrections:
                old_val = updated_bill.get(key)
                new_val = corrections[key]
                updated_bill[key] = new_val
                change_notes.append(f"Changed {key.replace('_', ' ')} from '{old_val}' to '{new_val}'.")
        
        # Apply item modifications/deletions
        if "items" in corrections and isinstance(corrections["items"], list):
            original_items = updated_bill.get("items", [])
            indices_to_delete = []
            for item_change in corrections["items"]:
                if not isinstance(item_change, dict) or "index" not in item_change:
                    continue
                idx = item_change["index"]
                if not (0 <= idx < len(original_items)):
                    change_notes.append(f"Ignored correction for invalid item index {idx+1}.")
                    continue

                if item_change.get("_delete") is True:
                    indices_to_delete.append(idx)
                    change_notes.append(f"Marked item {idx+1} ('{original_items[idx].get('name')}') for deletion.")
                else:
                    item_note = f"For item {idx+1} ('{original_items[idx].get('name')}'): "
                    item_changed_fields = []
                    for field in ["name", "quantity", "price_per_unit", "total_price"]:
                        if field in item_change:
                            old_item_val = original_items[idx].get(field)
                            new_item_val = item_change[field]
                            original_items[idx][field] = new_item_val
                            item_changed_fields.append(f"{field} from '{old_item_val}' to '{new_item_val}'")
                    if item_changed_fields:
                        change_notes.append(item_note + ", ".join(item_changed_fields) + ".")
            
            # Process deletions by rebuilding item list (safer than deleting in place while iterating by index)
            if indices_to_delete:
                updated_items_list = []
                for i, item_data in enumerate(original_items):
                    if i not in indices_to_delete:
                        updated_items_list.append(item_data)
                updated_bill["items"] = updated_items_list
                change_notes.append(f"Removed {len(indices_to_delete)} item(s) as requested.")
            else:
                updated_bill["items"] = original_items # if only modifications, not deletions

        # Add new items
        if "items_to_add" in corrections and isinstance(corrections["items_to_add"], list):
            if "items" not in updated_bill: updated_bill["items"] = []
            for new_item in corrections["items_to_add"]:
                if isinstance(new_item, dict) and new_item.get("name") and new_item.get("total_price") is not None:
                    # Basic validation for new item
                    item_to_add = {
                        "name": new_item["name"],
                        "quantity": new_item.get("quantity", 1),
                        "price_per_unit": new_item.get("price_per_unit"),
                        "total_price": new_item["total_price"]
                    }
                    updated_bill["items"].append(item_to_add)
                    change_notes.append(f"Added new item: {item_to_add['name']} (Price: {item_to_add['total_price']}).")
                else:
                    change_notes.append("Ignored invalid new item structure during addition.")
        
        if not change_notes:
            change_notes.append("No specific changes understood from your command.")
            
        return updated_bill, change_notes 