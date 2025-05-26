from telegram import Update, Bot, PhotoSize, Document
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import logging
import json # For pretty printing Update object
import asyncio # Added for create_task
from typing import Optional, Any, Tuple, Dict, List
import time # Added for time
import io # For BytesIO

from .base import MessagingInterface
from .models import ConversationStateData # Ensure ConversationStateData is imported
from config.settings import settings 
from core.processor import BillSplitProcessor, BillSplitStates # Import new states
from core.state_manager import ConversationState 
from monitoring.langfuse_client import LangfuseMonitor

logger = logging.getLogger(__name__)

class TelegramInterface(MessagingInterface):
    def __init__(self, token: str = settings.TELEGRAM_BOT_TOKEN, 
                 processor: Optional[BillSplitProcessor] = None,
                 langfuse_monitor: Optional[LangfuseMonitor] = None):
        self.token = token
        if not self.token:
            raise ValueError("Telegram bot token is not configured.")
        self.bot = Bot(token=self.token)
        self.processor = processor 
        self.langfuse_monitor = langfuse_monitor
        if not self.processor:
            logger.warning("TelegramInterface initialized without a BillSplitProcessor.")
        if self.langfuse_monitor:
            logger.info("TelegramInterface initialized with LangfuseMonitor.")

        self.application = Application.builder().token(self.token).build()
        self._setup_handlers()
        self._is_initialized = False

    def _setup_handlers(self) -> None:
        """Setup a broader range of handlers."""
        self.application.add_handler(CommandHandler("start", self._start_command))
        self.application.add_handler(CommandHandler("split", self._start_command)) # Alias for /start
        self.application.add_handler(CommandHandler("help", self._help_command))
        self.application.add_handler(CommandHandler("status", self._status_command))
        self.application.add_handler(CommandHandler("cancel", self._cancel_command))
        
        # Message handlers
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text_message))
        self.application.add_handler(MessageHandler(filters.PHOTO, self._handle_photo_message))
        # Handle documents that are images
        self.application.add_handler(MessageHandler(filters.Document.IMAGE, self._handle_document_message))
        # Add a handler for other document types if needed, or a generic non-image document handler
        self.application.add_handler(MessageHandler(filters.Document.ALL & (~filters.Document.IMAGE), self._handle_other_document_types))
        # self.application.add_handler(MessageHandler(filters.VOICE, self._handle_voice_message)) # TODO: Add voice handling
        
        # Fallback for unknown commands or messages not caught by other handlers
        self.application.add_handler(MessageHandler(filters.COMMAND, self._unknown_command))
        self.application.add_handler(MessageHandler(filters.ALL, self._unhandled_message_type))
        logger.info("Telegram command and message handlers set up.")

    async def initialize_application(self) -> None:
        """Initializes and pre-populates the Application. Needs to be called in an async context."""
        if self._is_initialized:
            logger.info("Telegram application already initialized.")
            return
        await self.application.initialize()
        self._is_initialized = True
        logger.info("Telegram application initialized.")

    async def run(self) -> None:
        if not self._is_initialized:
            await self.application.initialize()
            self._is_initialized = True # Ensure flag is set

        if not self.application.updater:
             await self.application.run_polling(allowed_updates=Update.ALL_TYPES)
             logger.info("Telegram bot started polling.")
        else:
            logger.info("Telegram bot is already running or polling setup is managed externally.")

    def _create_trace_if_configured(self, update: Update, handler_name: str) -> Optional[Any]:
        if not self.langfuse_monitor or not self.langfuse_monitor.get_client():
            return None
        
        user_id = str(update.effective_user.id) if update.effective_user else "unknown_user"
        chat_id = str(update.effective_chat.id) if update.effective_chat else "unknown_chat"
        # conversation_id will be set later from state_data_obj.conversation_id
        # For now, we create a temporary ID or use chat_id for grouping before state is fetched.
        # Using a unique ID per interaction initially, can be linked later via metadata or if Langfuse supports trace updates.
        temp_trace_id = f"telegram_interaction_{update.update_id}"
        
        trace_name = f"telegram_handler_{handler_name}"
        trace_metadata = {
            "user_id": user_id,
            "chat_id": chat_id,
            "handler": handler_name,
            "update_id": update.update_id,
            "message_type": update.message.chat.type if update.message and update.message.chat else "N/A",
            # actual_conversation_id will be added once known
        }
        if update.message and update.message.text:
            trace_metadata["text_preview"] = update.message.text[:50]

        # Create a trace. Note: trace_id here becomes the ID in Langfuse.
        # If we want to group by conversation_id from state, we need to fetch state first
        # or use user_id/chat_id as session_id if Langfuse supports that concept for grouping.
        # For now, let's assume each handler interaction is a trace, and we can add conversation_id to metadata.
        return self.langfuse_monitor.trace_conversation(
            conversation_id=temp_trace_id, # This is the langfuse trace ID
            name=trace_name,
            user_id=user_id, 
            metadata=trace_metadata
        )

    async def _process_input(self, update: Update, context: ContextTypes.DEFAULT_TYPE, 
                             image_bytes: Optional[bytes] = None, 
                             document_bytes: Optional[bytes] = None, 
                             text_input: Optional[str] = None,
                             # voice_bytes: Optional[bytes] = None # TODO: For voice
                             ):
        if not update.message:
            logger.warning("Update received without a message in _process_input.")
            return

        user_id = str(update.effective_chat.id)
        user_first_name = update.effective_user.first_name if update.effective_user else "User"
        
        # Trace creation is now deferred until after state is fetched to use the correct conversation_id
        trace = None 
        state_data_obj: Optional[ConversationStateData] = None # To ensure it's defined for finally block

        try:
            if not self.processor or not self.processor.state_manager:
                logger.error(f"BillSplitProcessor or StateManager N/A for user {user_id}.")
                await update.message.reply_text("Service unavailable.")
                return

            raw_state_data, _, error_msg = await self.processor.state_manager.get_state(user_id)
            
            if error_msg or not raw_state_data or not isinstance(raw_state_data, dict):
                logger.warning(f"State error for {user_id}: {error_msg if error_msg else 'No/Invalid state'}. Initializing.")
                state_data_obj = ConversationStateData(user_id=user_id, current_state=BillSplitStates.INITIAL, user_name=user_first_name)
                state_data_obj.conversation_id = f"telegram_{user_id}_{int(time.time())}" # Create new conversation ID
            else:
                try:
                    state_data_obj = ConversationStateData.from_dict(raw_state_data)
                    if not state_data_obj.user_name: state_data_obj.user_name = user_first_name
                    if not state_data_obj.conversation_id: 
                        state_data_obj.conversation_id = f"telegram_{user_id}_{int(time.time())}"
                        logger.info(f"Generated new conversation_id for existing state for user {user_id}: {state_data_obj.conversation_id}")
                except Exception as e_conv:
                    logger.error(f"Corrupted state for {user_id}: {e_conv}. Re-initializing.")
                    state_data_obj = ConversationStateData(user_id=user_id, current_state=BillSplitStates.INITIAL, user_name=user_first_name)
                    state_data_obj.conversation_id = f"telegram_{user_id}_{int(time.time())}"

            # Now create trace with the correct conversation_id
            if self.langfuse_monitor and self.langfuse_monitor.get_client() and state_data_obj.conversation_id:
                 trace = self.langfuse_monitor.trace_conversation(
                    conversation_id=state_data_obj.conversation_id, # Use state's conversation_id as Langfuse trace ID
                    name=f"telegram_message_handling", # Generic name, can be more specific
                    user_id=user_id,
                    metadata={"handler_trigger": update.message.text or update.message.caption or "media_input", "initial_update_id": update.update_id}
                )

            message_payload: Dict[str, Any] = {"user_name": user_first_name}
            processed_as_type = "unknown"

            if image_bytes:
                message_payload["type"] = "image"; message_payload["content"] = image_bytes
                message_payload["caption"] = update.message.caption or ""
                processed_as_type = "image"
            elif document_bytes: 
                message_payload["type"] = "image"; message_payload["content"] = document_bytes
                message_payload["caption"] = update.message.caption or ""
                processed_as_type = "document_image"
            elif text_input:
                message_payload["type"] = "text"; message_payload["content"] = text_input
                processed_as_type = "text"
            # elif voice_bytes: # TODO
            #     message_payload["type"] = "audio"; message_payload["content"] = voice_bytes
            #     processed_as_type = "audio"
            else:
                logger.warning(f"No processable input in _process_input for {user_id}.")
                await update.message.reply_text("Didn't get a clear message. Try text, image, or voice.")
                return
            
            logger.info(f"Passing to processor for user {user_id}, type: {processed_as_type}, convo_id: {state_data_obj.conversation_id}")
            
            # Pass the raw_state_data dict, processor will convert it internally
            response_text = await self.processor.process_message(user_id, "telegram", raw_state_data, message_payload)
            
            if response_text:
                await update.message.reply_text(response_text)
                if trace: trace.update(output={"response_preview": response_text[:100]})
            else:
                await update.message.reply_text("Not sure how to respond. Try again.")
                if trace: trace.update(level="ERROR", status_message="Processor empty response")

        except Exception as e:
            logger.error(f"Error in _process_input for {user_id}: {e}", exc_info=True)
            await update.message.reply_text("Sorry, an error occurred.")
            if trace: trace.end(level="ERROR", status_message=str(e))
        finally:
            if trace and hasattr(trace, 'ended') and not trace.ended:
                 trace.end()

    async def _handle_text_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.text: return
        logger.info(f"Text from {update.effective_chat.id}: '{update.message.text[:50]}...'")
        await self._process_input(update, context, text_input=update.message.text)

    async def _handle_photo_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.photo: return        
        photo_file = await update.message.photo[-1].get_file()
        image_stream = io.BytesIO()
        await photo_file.download_to_memory(image_stream)
        logger.info(f"Photo from {update.effective_chat.id}. Caption: '{update.message.caption or "N/A"}'")
        await self._process_input(update, context, image_bytes=image_stream.getvalue())

    async def _handle_document_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.document: return
        document: Document = update.message.document
        if document.mime_type and document.mime_type.startswith("image/"):
            doc_file = await document.get_file()
            doc_stream = io.BytesIO()
            await doc_file.download_to_memory(doc_stream)
            logger.info(f"Doc (image) from {update.effective_chat.id}. Caption: '{update.message.caption or "N/A"}'")
            await self._process_input(update, context, document_bytes=doc_stream.getvalue())
        else:
            await update.message.reply_text("I can only process image files as bills.")
            logger.info(f"Doc type {document.mime_type} not processed as bill image.")

    async def _handle_other_document_types(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.document: return
        logger.info(f"User {update.effective_chat.id} sent unhandled document type: {update.message.document.mime_type}")
        await update.message.reply_text("I can only process images sent as photos or image files. Please try sending your bill as an image.")

    # async def _handle_voice_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    #     # TODO: Implement voice handling: download voice, transcribe, pass to _process_input
    #     logger.info(f"Voice message received from user {update.effective_chat.id}")
    #     await update.message.reply_text("Voice processing is coming soon!")

    async def _start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message: return
        user_id = str(update.effective_chat.id)
        user_first_name = update.effective_user.first_name if update.effective_user else "User"
        logger.info(f"/start from {user_id} (Name: {user_first_name})")
        
        trace = None # Trace will be created by process_command if needed, using conversation_id from state
        try:
            if self.processor:
                response_message = await self.processor.process_command(user_id, "telegram", "/start", user_first_name_from_interface=user_first_name)
                await update.message.reply_text(response_message)
            else:
                await update.message.reply_text(f"Hello {user_first_name}! Service offline.")
        except Exception as e:
            logger.error(f"Error in _start_command for {user_id}: {e}", exc_info=True)
            await update.message.reply_text("Problem starting. Try /start again.")

    async def _status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message: return
        user_id = str(update.effective_chat.id)
        logger.info(f"/status from {user_id}")
        try:
            if self.processor:
                response_message = await self.processor.process_command(user_id, "telegram", "/status")
                await update.message.reply_text(response_message)
            else: await update.message.reply_text("Service offline for status.")
        except Exception as e:
            logger.error(f"Error in _status_command for {user_id}: {e}", exc_info=True)
            await update.message.reply_text("Could not fetch status. Try /start.")

    async def _cancel_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message: return
        user_id = str(update.effective_chat.id)
        logger.info(f"/cancel from {user_id}")
        try:
            if self.processor:
                response_message = await self.processor.process_command(user_id, "telegram", "/cancel")
                await update.message.reply_text(response_message)
            else: await update.message.reply_text("Service offline for cancel.")
        except Exception as e:
            logger.error(f"Error in _cancel_command for {user_id}: {e}", exc_info=True)
            await update.message.reply_text("Trouble cancelling. Try /start.")

    async def _help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message: return
        # Help text from strategy
        user_name = update.effective_user.first_name if update.effective_user else "there"
        help_text = f"""
        Hi {user_name}! I'm SplitBot. Here's how I can help you split bills in 3 simple steps:
        
        1.  **Send Bill**: Send me a photo of your bill, an image file, or type out bill details (e.g., "Split $50 with John and Jane at Cafe Example").
        2.  **Instructions**: Tell me how to split it (e.g., "John pays for item 1, rest split equally", or "Alice pays 60%, Bob 40%").
        3.  **Confirm**: I'll show you the breakdown for you to confirm!
        
        Available commands:
        /start or /split - Start a new bill splitting session.
        /cancel - Cancel current operation.
        /status - Check current progress.
        /help - Show this message.
        """
        await update.message.reply_text(help_text)
        trace = self._create_trace_if_configured(update, "help_command") # Create trace for help too
        if trace: trace.end()

    async def _unknown_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.text: return
        logger.info(f"Unknown command from {update.effective_chat.id}: {update.message.text}")
        await update.message.reply_text("Sorry, I didn't understand that command. Try /help for a list of commands.")

    async def _unhandled_message_type(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message: return
        logger.info(f"Unhandled message type received from {update.effective_chat.id}. Message: {update.message}")
        await update.message.reply_text("I can only process text, images, or image documents for bill splitting. Please try one of those or use /help.")