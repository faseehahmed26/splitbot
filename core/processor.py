import logging
from typing import Optional, Tuple, Any, Dict, List
import re # For parsing item assignments
import json # For serializing current_bill_json for LLM
import time

from core.state_manager import ConversationState
from services.llm.base import LLMService
# Removed OCRService import
from .response_templates import RESPONSE_TEMPLATES # Import from new file
# from monitoring.langfuse_client import LangfuseMonitor # langfuse_client (SDK) is passed

logger = logging.getLogger(__name__)

# Define NEW conversation states
class BillSplitStates:
    INITIAL = "INITIAL"           # After /start, initial welcome
    AWAITING_BILL = "AWAITING_BILL"  # Waiting for bill input (image, text, or audio)
    PROCESSING_BILL = "PROCESSING_BILL"  # Internal state: OCR/LLM processing bill data
    AWAITING_SPLIT_INSTRUCTIONS = "AWAITING_SPLIT_INSTRUCTIONS"  # Waiting for user to tell how to split
    PROCESSING_SPLIT = "PROCESSING_SPLIT" # Internal state: LLM processing split instructions
    CONFIRMING_SPLIT = "CONFIRMING_SPLIT"  # Showing summary, awaiting user confirmation (Yes/No/Adjust)
    AWAITING_ADJUSTMENT = "AWAITING_ADJUSTMENT" # User wants to adjust, awaiting details
    COMPLETED = "COMPLETED"       # Split finalized, summary shown, ready for new /start
    ERROR_STATE = "ERROR_STATE" # Generic error state

# Mapping old states to new ones or removing them - for reference during refactoring
# STATE_AWAITING_BILL_IMAGE -> AWAITING_BILL
# STATE_PROCESSING_BILL_IMAGE -> PROCESSING_BILL
# STATE_BILL_IMAGE_PROCESSED_CONFIRM_OCR -> (Handled within PROCESSING_BILL or specific error recovery)
# STATE_EXTRACTING_BILL_DETAILS -> (Part of PROCESSING_BILL)
# STATE_BILL_DETAILS_EXTRACTED_CONFIRM_ITEMS -> (Replaced by new AWAITING_SPLIT_INSTRUCTIONS)
# STATE_AWAITING_PARTICIPANTS -> (Handled by LLM during AWAITING_SPLIT_INSTRUCTIONS)
# STATE_AWAITING_ITEM_ASSIGNMENTS -> (Handled by LLM during AWAITING_SPLIT_INSTRUCTIONS)
# STATE_CALCULATING_SPLIT -> (Part of PROCESSING_SPLIT)
# STATE_SPLIT_CALCULATED_SHOW_SUMMARY -> CONFIRMING_SPLIT
# STATE_SPLIT_FINALIZED -> COMPLETED
# STATE_AWAITING_CORRECTION_OCR -> (Handled by new error/adjustment flow)
# STATE_AWAITING_CORRECTION_ITEMS -> (Handled by new AWAITING_ADJUSTMENT)


# Keep STATE_DESCRIPTIONS if used for /status, or update/remove
# This might need adjustment based on the new states if /status is to be kept meaningful
STATE_DESCRIPTIONS = {
    BillSplitStates.INITIAL: "I'm ready to start a new bill split.",
    BillSplitStates.AWAITING_BILL: "I'm waiting for you to send the bill details (image, text, or voice).",
    BillSplitStates.PROCESSING_BILL: "I'm currently processing the bill information.",
    BillSplitStates.AWAITING_SPLIT_INSTRUCTIONS: "I have the bill details. How would you like to split it?",
    BillSplitStates.PROCESSING_SPLIT: "I'm calculating the split based on your instructions.",
    BillSplitStates.CONFIRMING_SPLIT: "I have a split summary for you. Please confirm or tell me if you need changes.",
    BillSplitStates.AWAITING_ADJUSTMENT: "Okay, what changes would you like to make to the split?",
    BillSplitStates.COMPLETED: "The bill split is complete! I'm ready for a new one.",
    BillSplitStates.ERROR_STATE: "It seems there was an issue. Please try again or type /start."
}

class BillSplitProcessor:
    def __init__(self, 
                 llm_service: LLMService, 
                 # ocr_service: OCRService, # Removed ocr_service from constructor
                 state_manager: ConversationState,
                 langfuse_client: Optional[Any]):
        self.llm_service = llm_service
        # self.ocr_service = ocr_service # Removed ocr_service assignment
        self.state_manager = state_manager
        self.langfuse_client = langfuse_client
        self.RESPONSE_TEMPLATES = RESPONSE_TEMPLATES # Use imported templates
        logger.info("BillSplitProcessor initialized using response_templates.py (OCR removed).")

    async def process_command(self, user_id: str, interface_type: str, command: str, user_first_name_from_interface: Optional[str] = None) -> str:
        """Handles direct commands like /start, /cancel, /status."""
        from interfaces.models import ConversationStateData # Import here
        
        current_state_tuple = await self.state_manager.get_state(user_id)
        state_dict = current_state_tuple[0] if current_state_tuple and current_state_tuple[0] else {}

        current_state_data_obj: ConversationStateData
        if not state_dict or not isinstance(state_dict, dict) or "user_id" not in state_dict:
            logger.info(f"No valid state found for user {user_id} during command processing. Initializing.")
            current_state_data_obj = ConversationStateData(
                user_id=user_id, current_state=BillSplitStates.INITIAL, step=BillSplitStates.INITIAL,
                user_name=user_first_name_from_interface or "User", conversation_id=f"{user_id}_{int(time.time())}"
            )
        else:
            try:
                if 'current_state' not in state_dict: state_dict['current_state'] = BillSplitStates.INITIAL
                if 'step' not in state_dict: state_dict['step'] = state_dict['current_state']
                current_state_data_obj = ConversationStateData.from_dict(state_dict)
                if not current_state_data_obj.user_name and user_first_name_from_interface:
                    current_state_data_obj.user_name = user_first_name_from_interface
                if not current_state_data_obj.conversation_id:
                     current_state_data_obj.conversation_id = f"{user_id}_{int(time.time())}"
            except Exception as e:
                logger.error(f"Error converting state dict to ConversationStateData for {user_id}: {e}. Re-initializing.")
                current_state_data_obj = ConversationStateData(
                    user_id=user_id, current_state=BillSplitStates.INITIAL, step=BillSplitStates.INITIAL,
                    user_name=user_first_name_from_interface or "User", conversation_id=f"{user_id}_{int(time.time())}"
                )

        current_conv_state = current_state_data_obj.current_state
        response_message = f"Received command: {command}"
        next_conv_state = current_conv_state
        user_name_to_greet = current_state_data_obj.user_name or "there"

        logger.info(f"User {user_id} (Name: {user_name_to_greet}) command '{command}' in state '{current_conv_state}', convo_id: {current_state_data_obj.conversation_id}")
        
        # Langfuse span (commented out for now)

        if command == "/start" or command == "/split":
            current_state_data_obj.conversation_id = f"{user_id}_{int(time.time())}"
            current_state_data_obj.history = [] 
            current_state_data_obj.bill_data = None
            current_state_data_obj.split_data = None
            next_conv_state = BillSplitStates.INITIAL 
            response_message = self.RESPONSE_TEMPLATES["initial"].format(user_name=user_name_to_greet)
            current_state_data_obj.add_message(role="assistant", message_type="text", content=response_message)
            logger.info(f"State reset for {user_id} (Name: {user_name_to_greet}) by {command}. New convo_id: {current_state_data_obj.conversation_id}")
        
        elif command == "/cancel":
            response_message = f"Okay {user_name_to_greet}, operation cancelled. Let's start over." \
                if current_conv_state not in [BillSplitStates.INITIAL, BillSplitStates.COMPLETED] \
                else f"Nothing active to cancel, {user_name_to_greet}. Let's start a new bill!"
            
            current_state_data_obj.add_message(role="user", message_type="text", content=command)
            current_state_data_obj.add_message(role="assistant", message_type="text", content=response_message)
            
            current_state_data_obj.conversation_id = f"{user_id}_{int(time.time())}"
            current_state_data_obj.history = []
            current_state_data_obj.bill_data = None
            current_state_data_obj.split_data = None
            next_conv_state = BillSplitStates.INITIAL
            logger.info(f"Cancelled for {user_id} (Name: {user_name_to_greet}). New convo_id: {current_state_data_obj.conversation_id}")

        elif command == "/status":
            status_desc = STATE_DESCRIPTIONS.get(current_conv_state, "unknown state.")
            response_message = f"Hi {user_name_to_greet}! Status: {status_desc}"
            if current_conv_state == BillSplitStates.AWAITING_SPLIT_INSTRUCTIONS and current_state_data_obj.bill_data:
                response_message += "\nWaiting for split instructions."
            elif current_conv_state == BillSplitStates.CONFIRMING_SPLIT and current_state_data_obj.split_data:
                response_message += "\nWaiting for split confirmation."
            current_state_data_obj.add_message(role="user", message_type="text", content=command)
            current_state_data_obj.add_message(role="assistant", message_type="text", content=response_message)
        
        else:
            response_message = f"Sorry {user_name_to_greet}, unknown command. Try /help or /start."
            current_state_data_obj.add_message(role="user", message_type="text", content=command)
            current_state_data_obj.add_message(role="assistant", message_type="text", content=response_message)
            logger.warning(f"Unknown command '{command}' for {user_id} (Name: {user_name_to_greet}).")

        current_state_data_obj.current_state = next_conv_state
        current_state_data_obj.step = next_conv_state
        await self.state_manager.update_state(user_id, current_state_data_obj.to_dict())
        return response_message

    async def process_message(self, user_id: str, interface_type: str,
                              raw_state_data: Optional[Dict[str, Any]],
                              message_payload: Dict[str, Any]
                              ) -> str:
        from interfaces.models import ConversationStateData

        state_data_obj: ConversationStateData
        user_first_name_from_payload = message_payload.get("user_name")

        if not raw_state_data or not isinstance(raw_state_data, dict) or "user_id" not in raw_state_data:
            state_data_obj = ConversationStateData(
                user_id=user_id, current_state=BillSplitStates.INITIAL, step=BillSplitStates.INITIAL,
                user_name=user_first_name_from_payload or "User", conversation_id=f"{user_id}_{int(time.time())}"
            )
        else:
            try:
                if 'current_state' not in raw_state_data: raw_state_data['current_state'] = BillSplitStates.INITIAL
                if 'step' not in raw_state_data: raw_state_data['step'] = raw_state_data['current_state']
                state_data_obj = ConversationStateData.from_dict(raw_state_data)
                if not state_data_obj.user_name and user_first_name_from_payload:
                    state_data_obj.user_name = user_first_name_from_payload
                if not state_data_obj.conversation_id:
                    state_data_obj.conversation_id = f"{user_id}_{int(time.time())}"
            except Exception as e:
                logger.error(f"Error converting raw_state_data for {user_id}: {e}. Re-initializing.")
                state_data_obj = ConversationStateData(
                    user_id=user_id, current_state=BillSplitStates.INITIAL, step=BillSplitStates.INITIAL,
                    user_name=user_first_name_from_payload or "User", conversation_id=f"{user_id}_{int(time.time())}"
                )

        user_name_to_greet = state_data_obj.user_name or "User"
        state_data_obj.add_message(
            role="user", message_type=message_payload["type"],
            content=message_payload["content"], caption=message_payload.get("caption"),
            transcription=message_payload.get("transcription")
        )

        current_conv_state = state_data_obj.current_state
        logger.info("Processing message", extra={
            "user_id": user_id, "user_name": user_name_to_greet, "current_conv_state": current_conv_state,
            "message_type": message_payload["type"], "conversation_id": state_data_obj.conversation_id
        })
        
        response_text = f"Sorry {user_name_to_greet}, I'm not sure how to handle that."
        next_conv_state = current_conv_state
        
        # Langfuse span (commented out)

        try:
            if current_conv_state == BillSplitStates.INITIAL:
                logger.info(f"User {user_id} sent message in INITIAL. Transitioning to AWAITING_BILL.")
                current_conv_state = BillSplitStates.AWAITING_BILL
            
            if current_conv_state == BillSplitStates.AWAITING_BILL:
                response_text, next_conv_state = await self._handle_bill_input(state_data_obj, message_payload, user_name_to_greet)
            elif current_conv_state == BillSplitStates.AWAITING_SPLIT_INSTRUCTIONS:
                if message_payload["type"] == "text":
                    response_text, next_conv_state = await self._handle_split_instructions(state_data_obj, message_payload["content"], user_name_to_greet)
                else:
                    response_text = f"Expecting split instructions as text, {user_name_to_greet}."
            elif current_conv_state == BillSplitStates.CONFIRMING_SPLIT:
                if message_payload["type"] == "text":
                    response_text, next_conv_state = await self._handle_split_confirmation(state_data_obj, message_payload["content"], user_name_to_greet)
                else:
                    response_text = f"Please confirm with 'Yes', 'No', or 'Adjust', {user_name_to_greet}."
            elif current_conv_state == BillSplitStates.AWAITING_ADJUSTMENT:
                 if message_payload["type"] == "text":
                    response_text, next_conv_state = await self._handle_adjustment_input(state_data_obj, message_payload["content"], user_name_to_greet)
                 else:
                    response_text = f"Please describe adjustments in text, {user_name_to_greet}."
            elif current_conv_state == BillSplitStates.COMPLETED:
                final_summary = self._format_final_summary(state_data_obj.split_data)
                response_text = self.RESPONSE_TEMPLATES["completed_already"].format(user_name=user_name_to_greet, final_summary=final_summary)
            else:
                logger.error(f"Unhandled state '{current_conv_state}' for {user_id}.")
                response_text = f"Unexpected state ({current_conv_state}), {user_name_to_greet}. Try /start."
                next_conv_state = BillSplitStates.ERROR_STATE

            state_data_obj.current_state = next_conv_state
            state_data_obj.step = next_conv_state
            state_data_obj.add_message(role="assistant", message_type="text", content=response_text)
            await self.state_manager.update_state(user_id, state_data_obj.to_dict())

        except Exception as e:
            logger.error(f"Error processing message for {user_id} (Name: {user_name_to_greet}) in {current_conv_state}: {e}", exc_info=True)
            response_text = f"Unexpected error, {user_name_to_greet}. Try /start."
            state_data_obj.current_state = BillSplitStates.ERROR_STATE
            state_data_obj.step = BillSplitStates.ERROR_STATE
            state_data_obj.add_message(role="assistant", message_type="text", content=response_text)
            await self.state_manager.update_state(user_id, state_data_obj.to_dict())
            
        return response_text

    async def _handle_bill_input(self, state_data_obj: 'ConversationStateData', message: Dict[str, Any], user_name: str) -> Tuple[str, str]:
        extracted_bill_info = None
        next_state = BillSplitStates.AWAITING_BILL # Default to AWAITING_BILL on failure to get new input
        input_type_for_llm = "unknown"
        llm_input_content = ""

        if not self.llm_service:
            logger.error(f"LLM service not available for _handle_bill_input for user {state_data_obj.user_id}")
            return f"I'm having trouble connecting to my brain, {user_name}. Please try again later.", BillSplitStates.ERROR_STATE

        try:
            if message["type"] == "image":
                input_type_for_llm = "image"
                image_bytes = message["content"]
                # Assuming Gemini can take image_mime_type from settings or it defaults appropriately in service
                # Ensure llm_service.process_receipt_from_image exists and handles this.
                if not hasattr(self.llm_service, 'process_receipt_from_image'):
                    logger.error(f"LLM service {type(self.llm_service).__name__} missing process_receipt_from_image method.")
                    return f"My image processing ability isn't set up right now, {user_name}. Try text?", next_state
                
                extracted_bill_info = await self.llm_service.process_receipt_from_image(
                    image_bytes=image_bytes,
                    # caption=message.get("caption", "") # Pass caption if model can use it
                )
            
            elif message["type"] == "audio":
                input_type_for_llm = "audio"
                audio_bytes = message["content"]
                # Ensure llm_service.transcribe_audio exists and then process_receipt on transcription.
                if not hasattr(self.llm_service, 'transcribe_audio') or not hasattr(self.llm_service, 'process_receipt'):
                    logger.error(f"LLM service {type(self.llm_service).__name__} missing transcribe_audio or process_receipt method for audio.")
                    return f"My audio processing ability isn't set up right now, {user_name}. Try text or image?", next_state

                transcription = await self.llm_service.transcribe_audio(audio_bytes)
                state_data_obj.add_message(role="user", message_type="transcription", content=transcription) # Log transcription
                if not transcription:
                    return f"I couldn't understand the audio, {user_name}. Could you try again or send a text/image?", next_state
                llm_input_content = transcription # Use transcription for receipt processing
                extracted_bill_info = await self.llm_service.process_receipt(llm_input_content)
            
            elif message["type"] == "text":
                input_type_for_llm = "text"
                text_content = message["content"]
                # Check if it's a quick bill instruction (e.g. "$50 for lunch")
                # This logic might be better suited inside the LLM or a specific parser
                # For now, assume any text in AWAITING_BILL is the bill itself or its description.
                if not hasattr(self.llm_service, 'process_receipt'):
                    logger.error(f"LLM service {type(self.llm_service).__name__} missing process_receipt method for text.")
                    return f"My text processing ability isn't set up right now, {user_name}.", next_state

                llm_input_content = text_content
                extracted_bill_info = await self.llm_service.process_receipt(llm_input_content)
            else:
                logger.warning(f"Unhandled message type '{message['type']}' in _handle_bill_input for {state_data_obj.user_id}")
                return f"I can only understand images, voice messages, or text for the bill, {user_name}.", next_state

        except Exception as e:
            logger.error(f"{input_type_for_llm.capitalize()} processing error for user {state_data_obj.user_id}: {e}", exc_info=True)
            return f"I had trouble understanding the {input_type_for_llm} you sent, {user_name}. Please try again.", next_state

        if not extracted_bill_info or extracted_bill_info.get("error"):
            error_detail = extracted_bill_info.get("error", "unknown reason") if extracted_bill_info else "no details extracted"
            logger.warning(f"Failed to extract bill details for {state_data_obj.user_id} via {input_type_for_llm}. LLM Error: {error_detail}")
            return self.RESPONSE_TEMPLATES["bill_extraction_failed"].format(user_name=user_name, details=error_detail), next_state

        # Successfully extracted bill info
        state_data_obj.bill_data = extracted_bill_info
        # state_data_obj.add_message(role="assistant", message_type="internal_log", content=f"Parsed bill data: {json.dumps(extracted_bill_info)}")
        logger.info(f"Successfully extracted bill for {state_data_obj.user_id} via {input_type_for_llm}. Proceeding to AWAITING_SPLIT_INSTRUCTIONS.")
        return self._format_bill_confirmation_for_user(extracted_bill_info, user_name), BillSplitStates.AWAITING_SPLIT_INSTRUCTIONS

    def _format_bill_confirmation_for_user(self, bill_data: Dict, user_name: str) -> str:
        restaurant = bill_data.get("restaurant_name", "Not specified")
        total_amount = bill_data.get("total", "N/A")
        currency = bill_data.get("currency", "")
        items_list = bill_data.get("items", [])
        item_lines = [f"  - {i.get('name', 'Item')} ({i.get('total_price', i.get('price', 'N/A'))})" for i in items_list[:3]]
        if len(items_list) > 3: item_lines.append(f"  ...and {len(items_list) - 3} more.")
        items_preview = "\n".join(item_lines) if item_lines else "  (No items extracted, only total)"
        return self.RESPONSE_TEMPLATES["bill_summary"].format(
            user_name=user_name, restaurant=restaurant,
            total=f"{total_amount} {currency}".strip(), items_preview=items_preview
        )

    async def _handle_split_instructions(self, state_data_obj: 'ConversationStateData', instruction_text: str, user_name: str) -> Tuple[str, str]:
        if not state_data_obj.bill_data or not state_data_obj.bill_data.get("total"):
            # Ensure bill_data and total exist before proceeding
            logger.warning(f"Missing bill data or total for user {state_data_obj.user_id} when trying to split.")
            return f"I seem to have lost the bill details or the total amount, {user_name}. Could you please send the bill again?", BillSplitStates.AWAITING_BILL
        
        logger.info(f"User {state_data_obj.user_id} (Name: {user_name}) provided split instructions: '{instruction_text}'")

        if not self.llm_service or not hasattr(self.llm_service, 'interpret_split_instructions'):
            logger.error(f"LLM service not available or missing 'interpret_split_instructions' for user {state_data_obj.user_id}")
            return f"I'm having trouble understanding how to split bills right now, {user_name}. Please try again later.", BillSplitStates.AWAITING_SPLIT_INSTRUCTIONS

        try:
            # Prepare context for LLM. This should include relevant parts of the conversation history.
            llm_context_history = state_data_obj.get_llm_context_history(max_messages=5) # Get recent history

            parsed_split_result = await self.llm_service.interpret_split_instructions(
                instruction_text=instruction_text, 
                current_bill_data=state_data_obj.bill_data, 
                conversation_history=llm_context_history,
                user_name=user_name
            )
        except Exception as e:
            logger.error(f"Error calling LLM for split interpretation for user {state_data_obj.user_id}: {e}", exc_info=True)
            return self.RESPONSE_TEMPLATES["split_calculation_error"].format(user_name=user_name), BillSplitStates.AWAITING_SPLIT_INSTRUCTIONS

        if not parsed_split_result or parsed_split_result.get("error"):
            error_detail = parsed_split_result.get("error", "LLM did not return a valid split.") if parsed_split_result else "LLM response was empty."
            logger.warning(f"LLM failed to interpret split for user {state_data_obj.user_id}. Error: {error_detail}")
            return self.RESPONSE_TEMPLATES["split_calculation_failed"].format(user_name=user_name, details=error_detail), BillSplitStates.AWAITING_SPLIT_INSTRUCTIONS
        
        if parsed_split_result.get("is_final") is True and parsed_split_result.get("breakdown"):
            state_data_obj.split_data = parsed_split_result
            logger.info(f"Split calculated for {state_data_obj.user_id}. Split data: {json.dumps(parsed_split_result)}")
            return self._format_split_confirmation_for_user(parsed_split_result, user_name), BillSplitStates.CONFIRMING_SPLIT
        elif parsed_split_result.get("clarification_needed"):
            clarification = parsed_split_result["clarification_needed"]
            logger.info(f"LLM needs clarification for split from {state_data_obj.user_id}: {clarification}")
            return self.RESPONSE_TEMPLATES["split_calculation_needs_clarification"].format(user_name=user_name, details=clarification), BillSplitStates.AWAITING_SPLIT_INSTRUCTIONS
        else:
            logger.warning(f"LLM returned non-final split without clarification for {state_data_obj.user_id}. Response: {parsed_split_result}")
            return self.RESPONSE_TEMPLATES["split_calculation_failed"].format(user_name=user_name, details="I couldn't finalize the split with that information."), BillSplitStates.AWAITING_SPLIT_INSTRUCTIONS

    def _format_split_confirmation_for_user(self, split_data: Dict, user_name: str) -> str:
        total = split_data.get("total", "N/A"); currency = split_data.get("currency", "")
        breakdown_dict = split_data.get("breakdown", {})
        person_lines = [f"  👤 {p}: {a} {currency}".strip() for p, a in breakdown_dict.items()] if breakdown_dict else ["  (Could not determine amounts)"]
        return self.RESPONSE_TEMPLATES["split_confirmation"].format(
            user_name=user_name, total=f"{total} {currency}".strip(), person_breakdown="\n".join(person_lines)
        )
        
    async def _handle_split_confirmation(self, state_data_obj: 'ConversationStateData', text: str, user_name: str) -> Tuple[str, str]:
        text_lower = text.lower()
        if any(w in text_lower for w in ["yes", "correct", "ok", "yep", "confirm"]):
            if not state_data_obj.split_data:
                 logger.error(f"Split confirmation 'yes' but no split_data for user {state_data_obj.user_id}")
                 return f"Something went wrong, I don't have the split details to confirm, {user_name}. Please try starting the split process again.", BillSplitStates.AWAITING_BILL
            final_summary = self._format_final_summary(state_data_obj.split_data)
            return self.RESPONSE_TEMPLATES["completed"].format(user_name=user_name, final_summary=final_summary), BillSplitStates.COMPLETED
        elif any(w in text_lower for w in ["no", "wrong", "change", "adjust"]):
            return self.RESPONSE_TEMPLATES["ask_for_adjustment_details"].format(user_name=user_name), BillSplitStates.AWAITING_ADJUSTMENT
        else:
            return self.RESPONSE_TEMPLATES["confirmation_not_understood"].format(user_name=user_name), BillSplitStates.CONFIRMING_SPLIT

    async def _handle_adjustment_input(self, state_data_obj: 'ConversationStateData', adjustment_text: str, user_name: str) -> Tuple[str, str]:
        logger.info(f"User {state_data_obj.user_id} (Name: {user_name}) requested adjustment: '{adjustment_text}'")

        if not self.llm_service or not hasattr(self.llm_service, 'apply_split_adjustment'):
            logger.error(f"LLM service not available or missing 'apply_split_adjustment' for user {state_data_obj.user_id}")
            return f"I'm having trouble with adjustments right now, {user_name}. Please try again later.", BillSplitStates.AWAITING_ADJUSTMENT

        if not state_data_obj.split_data or not state_data_obj.bill_data:
            logger.warning(f"Missing split_data or bill_data for adjustment by {state_data_obj.user_id}")
            return f"I don't have the current split or bill details to adjust, {user_name}. Let's try splitting the bill again.", BillSplitStates.AWAITING_BILL

        try:
            llm_context_history = state_data_obj.get_llm_context_history(max_messages=5)
            updated_split_data = await self.llm_service.apply_split_adjustment(
                adjustment_text=adjustment_text,
                current_split_data=state_data_obj.split_data,
                current_bill_data=state_data_obj.bill_data, # Pass original bill data for context
                conversation_history=llm_context_history,
                user_name=user_name
            )
        except Exception as e:
            logger.error(f"Error calling LLM for split adjustment for user {state_data_obj.user_id}: {e}", exc_info=True)
            return self.RESPONSE_TEMPLATES["adjustment_error"].format(user_name=user_name), BillSplitStates.AWAITING_ADJUSTMENT

        if not updated_split_data or updated_split_data.get("error"):
            error_detail = updated_split_data.get("error", "LLM could not apply adjustment.") if updated_split_data else "LLM response was empty."
            logger.warning(f"LLM failed to apply adjustment for {state_data_obj.user_id}. Error: {error_detail}")
            return self.RESPONSE_TEMPLATES["adjustment_failed"].format(user_name=user_name, details=error_detail), BillSplitStates.AWAITING_ADJUSTMENT

        if updated_split_data.get("is_final") is True and updated_split_data.get("breakdown"):
            state_data_obj.split_data = updated_split_data
            logger.info(f"Split adjusted for {state_data_obj.user_id}. New split: {json.dumps(updated_split_data)}")
            return self._format_split_confirmation_for_user(updated_split_data, user_name), BillSplitStates.CONFIRMING_SPLIT
        elif updated_split_data.get("clarification_needed"):
            clarification = updated_split_data["clarification_needed"]
            logger.info(f"LLM needs clarification for adjustment from {state_data_obj.user_id}: {clarification}")
            # Stay in AWAITING_ADJUSTMENT or ask for specific clarification
            return self.RESPONSE_TEMPLATES["adjustment_needs_clarification"].format(user_name=user_name, details=clarification), BillSplitStates.AWAITING_ADJUSTMENT
        else:
            # If LLM could not make a final adjustment and didn't ask for clarification, re-prompt for split instructions
            logger.warning(f"LLM adjustment non-final for {state_data_obj.user_id}. Response: {updated_split_data}. Re-prompting.")
            # This might be too abrupt; consider a more nuanced response or state.
            # For simplicity, we revert to asking for split instructions again with the original bill.
            state_data_obj.split_data = None # Clear previous attempt
            return self.RESPONSE_TEMPLATES["adjustment_re_prompt_split"].format(user_name=user_name), BillSplitStates.AWAITING_SPLIT_INSTRUCTIONS
        
    def _format_final_summary(self, split_data: Optional[Dict]) -> str:
        if not split_data: return "No split details available to summarize."
        total = split_data.get("total", "N/A"); currency = split_data.get("currency", "")
        breakdown = split_data.get("breakdown", {})
        lines = [f"💰 Total Bill: {total} {currency}".strip(), "\nHere's how it's split:"]
        lines.extend([f"  👤 {p}: {a} {currency}".strip() for p, a in breakdown.items()] if breakdown else ["  (Could not determine individual amounts)"])
        return "\n".join(lines)

    # Removed old/unused methods like _assign_items_to_users, _calculate_totals, _calculate_and_present_split, _apply_bill_corrections
    # as their logic is now primarily handled by the LLM service through more abstract interface methods.

    # Ensure all RESPONSE_TEMPLATES are defined in core/response_templates.py
    # Example content for core/response_templates.py:
    # RESPONSE_TEMPLATES = {
    # "initial": "👋 Hi {user_name}! I'm SplitBot...",
    # "bill_summary": "📋 Bill Summary for {user_name}:...",
    # ...etc
    # }

    # Ensure LLMService interface has methods like:
    # - process_receipt(text_input_with_caption_and_ocr) -> Dict (existing, ensure it handles combined input)
    # - attempt_text_bill_parse(text_input, user_name) -> Dict (existing, ensure it fits)
    # - interpret_split_instructions(instruction_text, bill_data, llm_context) -> Dict (NEW, crucial for core logic)
    # - apply_split_adjustment(adjustment_text, current_split_data, bill_data) -> Dict (NEW, for corrections)

    # OCRService should have:
    # - extract_text_from_bytes(image_bytes) -> str (NEW or ensure existing method matches)

    # AudioService (conceptual) would need:
    # - transcribe_audio_bytes(audio_bytes) -> str

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