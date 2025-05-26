from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import logging
import json # For pretty printing Update object
import asyncio # Added for create_task
from typing import Optional, Any, Tuple, Dict, List
import time # Added for time
# from dataclasses import dataclass, field # No longer needed here

from .base import MessagingInterface
from .models import Message, ReceiptItem, ReceiptData, ConversationStateData # Import ConversationStateData
from config.settings import settings # For Telegram token and other settings
from core.processor import BillSplitProcessor, STATE_AWAITING_BILL_IMAGE # Import state
from core.state_manager import ConversationState # To be integrated
# from services.llm.base import LLMService # To be integrated
# from services.ocr.base import OCRService # To be integrated
from monitoring.langfuse_client import LangfuseMonitor # Import LangfuseMonitor

logger = logging.getLogger(__name__)

# ConversationStateData is now imported from .models
# @dataclass
# class ConversationStateData:
#     step: str
#     user_id: str
#     history: List[Dict[str, Any]] = field(default_factory=list)
#     receipt_data: Optional[ReceiptData] = None
#     participants: Optional[List[str]] = None
#     splits: Optional[Dict[str, float]] = None
#     created_at: float = field(default_factory=time.time)
#     last_updated: float = field(default_factory=time.time)

#     def to_dict(self) -> Dict[str, Any]:
#         """Convert state to dictionary for storage"""
#         return {
#             "step": self.step,
#             "user_id": self.user_id,
#             "history": self.history,
#             "receipt_data": self.receipt_data.__dict__ if self.receipt_data else None,
#             "participants": self.participants,
#             "splits": self.splits,
#             "created_at": self.created_at,
#             "last_updated": self.last_updated
#         }

#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> 'ConversationStateData':
#         """Create state from dictionary"""
#         if data.get("receipt_data"):
#             data["receipt_data"] = ReceiptData(**data["receipt_data"])
#         return cls(**data)

class TelegramInterface(MessagingInterface):
    def __init__(self, token: str = settings.TELEGRAM_BOT_TOKEN, 
                 processor: Optional[BillSplitProcessor] = None,
                 langfuse_monitor: Optional[LangfuseMonitor] = None): # Add langfuse_monitor
        self.token = token
        if not self.token:
            raise ValueError("Telegram bot token is not configured.")
        self.bot = Bot(token=self.token)
        self.processor = processor # Store the processor instance
        self.langfuse_monitor = langfuse_monitor # Store it
        if not self.processor:
            logger.warning("TelegramInterface initialized without a BillSplitProcessor. Message processing will be limited.")
        if self.langfuse_monitor:
            logger.info("TelegramInterface initialized with LangfuseMonitor.")

        # Build the application but DO NOT initialize it here.
        # Initialization will be handled by an explicit async method called from lifespan.
        self.application = Application.builder().token(self.token).build()
        self._setup_handlers()
        self._is_initialized = False # Flag to track initialization

        # In interfaces/telegram.py, modify the initialize_application method:
    async def initialize_application(self):
        """Initializes the PTB application. Should be called once during startup."""
        if not self._is_initialized:
            try:
                logger.info("Initializing PTB Application...")
                # Initialize the bot first
                await self.bot.initialize()
                # Then initialize the application
                await self.application.initialize()
                # Start the application
                await self.application.start()
                self._is_initialized = True
                logger.info("PTB Application initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize PTB Application: {e}", exc_info=True)
                raise
        else:
            logger.info("PTB Application already initialized.")

        # In interfaces/telegram.py, modify the _setup_handlers method:
    def _setup_handlers(self):
        """Sets up the command and message handlers for the bot."""
        # Add command handlers
        self.application.add_handler(CommandHandler("start", self._start_command))
        self.application.add_handler(CommandHandler("help", self._help_command))
        self.application.add_handler(CommandHandler("split", self._split_command))
        self.application.add_handler(CommandHandler("cancel", self._cancel_command))
        self.application.add_handler(CommandHandler("status", self._status_command))
        
        # Add message handlers
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text_message))
        self.application.add_handler(MessageHandler(filters.PHOTO, self._handle_photo_message))
        self.application.add_handler(MessageHandler(filters.Document.IMAGE, self._handle_document_image_message))
        
        # Add error handler
        self.application.add_error_handler(self._error_handler)
        
        logger.info("Telegram bot handlers setup completed")
    def _create_trace_if_configured(self, update: Update, handler_name: str) -> Optional[Any]:
        """Creates a Langfuse trace if monitor is available."""
        if not self.langfuse_monitor or not self.langfuse_monitor.get_client():
            return None
        
        user_id = str(update.effective_user.id) if update.effective_user else "unknown_user"
        chat_id = str(update.effective_chat.id) if update.effective_chat else "unknown_chat"
        # conversation_id could be chat_id or a custom session ID if available
        conversation_id = f"telegram_{chat_id}"
        
        trace_name = f"telegram_handler_{handler_name}"
        trace_metadata = {
            "user_id": user_id,
            "chat_id": chat_id,
            "handler": handler_name,
            "update_id": update.update_id,
            "message_type": update.message.chat.type if update.message and update.message.chat else "N/A"
        }
        if update.message and update.message.text:
            trace_metadata["text_preview"] = update.message.text[:50]

        # Use conversation_id as trace_id for Langfuse trace
        return self.langfuse_monitor.trace_conversation(
            conversation_id=conversation_id, # This will be the trace_id in Langfuse
            name=trace_name,
            user_id=user_id, # Langfuse specific user_id field
            metadata=trace_metadata
        )

    async def _process_input(self, update: Update, context: ContextTypes.DEFAULT_TYPE, image_bytes: Optional[bytes] = None):
        """Common method to handle processing for text, photo, or document image."""
        # This method will now be the one calling self.processor.process_message
        # It replaces the direct call from _handle_text_message, _handle_photo_message
        trace = self._create_trace_if_configured(update, "process_input_unified") # New trace name
        user_id = str(update.effective_chat.id)
        message_text = update.message.text if update.message and update.message.text else None
        user_first_name = update.effective_user.first_name if update.effective_user else "User"

        # If image_bytes are not passed directly (e.g. from document handler),
        # try to get them from photo (for _handle_photo_message path)
        if image_bytes is None and update.message and update.message.photo:
            photo_file = await update.message.photo[-1].get_file()
            image_bytearray_content = await photo_file.download_as_bytearray()
            image_bytes = bytes(image_bytearray_content)

        try:
            if not self.processor:
                logger.error(f"BillSplitProcessor not available for user {user_id}.")
                await update.message.reply_text("Sorry, the bill processing service is not available right now.")
                if trace: trace.update(level="ERROR", status_message="Processor not available")
                return

            current_state_data, source, error_msg = await self.processor.state_manager.get_state(user_id)
            if error_msg:
                logger.warning(f"Error getting state for {user_id} from {source}: {error_msg}")
            
            if not current_state_data:
                logger.info(f"No existing state for user {user_id}, initializing new state.")
                initial_state_dict = ConversationStateData(
                    user_id=user_id,
                    step=STATE_AWAITING_BILL_IMAGE,
                    history=[],
                    # Store user's first name in state if available
                    # This field needs to be added to ConversationStateData model or handled dynamically
                    # For now, we'll pass it to the processor if needed, but ideally it becomes part of the state model.
                    # Let's add a temporary "user_profile" key to state if not too complex for now.
                ).to_dict()
                # Add user_name to the initial state dictionary
                initial_state_dict["user_name"] = user_first_name 

                success, update_source, update_error_msg = await self.processor.state_manager.update_state(user_id, initial_state_dict)
                if not success:
                    logger.error(f"Failed to initialize state for user {user_id} via {update_source}: {update_error_msg}")
                    await update.message.reply_text("Sorry, I couldn't prepare your session. Please try /start again.")
                    if trace: trace.update(level="ERROR", status_message="State init failed")
                    return
                current_state_data = initial_state_dict
                logger.info(f"Successfully initialized state for user {user_id} via {update_source}")

            response_text = await self.processor.process_message(
                user_id=user_id,
                interface_type="telegram",
                current_state_data=current_state_data,
                message_text=message_text,
                image_bytes=image_bytes
            )
            await update.message.reply_text(response_text)
            if trace: trace.update(output={"status": "success", "response_length": len(response_text)})

        except Exception as e:
            logger.error(f"Error in _process_input for user {user_id}: {e}", exc_info=True)
            await update.message.reply_text("Sorry, an unexpected error occurred while processing your request.")
            if trace: trace.update(level="ERROR", status_message=str(e))

    async def _start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Sends a welcome message when the /start command is issued."""
        trace = self._create_trace_if_configured(update, "start_command")
        user_id = str(update.effective_chat.id)
        user_first_name = update.effective_user.first_name if update.effective_user else "User"
        logger.info(f"/start command received from user_id: {user_id} (Name: {user_first_name})")
        try:
            if self.processor and self.processor.state_manager:
                initial_state_dict = ConversationStateData(
                    user_id=user_id,
                    step=STATE_AWAITING_BILL_IMAGE,
                    history=[]
                ).to_dict()
                initial_state_dict["user_name"] = user_first_name # Store user_name

                success, source, error_msg = await self.processor.state_manager.update_state(user_id, initial_state_dict)
                if not success:
                    logger.error(f"Failed to set initial state for /start command for user {user_id} via {source}: {error_msg}")
                    # Inform user about the issue
                    await update.message.reply_text("Sorry, I had a little trouble getting things ready. Could you try /start again?")
                    if trace: trace.update(level="ERROR", status_message="Failed to set initial state for /start")
                    return
                logger.info(f"Initial state set for user {user_id} via /start command using {source}.")
            
            welcome_message = (
                "Hello! I am your bill splitting bot. 👋\n\n"
                "I can help you split bills with your friends. Here's how:\n"
                "1. Send me a photo of your bill 📷 (or attach it as a file)\n"
                "2. I'll extract the items and amounts\n"
                "3. Tell me who participated\n"
                "4. I'll calculate the split!\n\n"
                "Type /help for more commands or just send me a bill photo to get started!"
            )
            
            await update.message.reply_text(welcome_message)
            if trace: trace.update(output={"status": "success", "reply": "Welcome message sent"})
        except Exception as e:
            if trace: trace.update(level="ERROR", status_message=str(e), output={"status": "error"})
            logger.error(f"Error in start command for user {user_id}: {e}", exc_info=True)
            await update.message.reply_text("Sorry, I encountered an error. Please try again.")

    async def _help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        trace = self._create_trace_if_configured(update, "help_command")
        try:
            user_id = str(update.effective_chat.id)
            logger.info(f"/help command received from user_id: {user_id}")
            help_text = (
                "I can help you split bills! Here's how:\n\n"
                "1. Send me a clear photo of your bill.\n"
                "2. I'll try to read it and show you what I found.\n"
                "3. Confirm or correct the details.\n"
                "4. Tell me who participated in the bill.\n"
                "5. Assign items to participants or mark items as shared.\n"
                "6. I'll calculate the split!\n\n"
                "Commands:\n"
                "- /start: Start a new bill splitting process (or reset current one).\n"
                "- /split: Same as /start, useful to begin.\n"
                "- /help: Show this help message.\n"
                "- /cancel: Cancel the current bill splitting operation and start over.\n"
                "- /status: Check the current step of your bill splitting process.\n\n"
                "Just send an image to get started!"
            )
            await update.message.reply_text(help_text)
            if trace: trace.update(output={"status": "success", "reply": "Help message sent"})
        except Exception as e:
            if trace: trace.update(level="ERROR", status_message=str(e))
            raise

    async def _split_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        trace = self._create_trace_if_configured(update, "split_command")
        try:
            user_id = str(update.effective_chat.id)
            logger.info(f"/split command received from user_id: {user_id}")
            # /split should behave like /start, resetting the state.
            await self._start_command(update, context) # Re-use start logic for reset and welcome
            # No need to call _process_input here as _start_command handles state and reply.
            if trace: trace.update(output={"status": "success", "reply": "Split command processed by invoking start"})
        except Exception as e:
            if trace: trace.update(level="ERROR", status_message=str(e))
            logger.error(f"Error in /split command for user {user_id}: {e}", exc_info=True)
            await update.message.reply_text("Sorry, there was an issue with the /split command. Please try /start.")
            # Do not re-raise, let Telegram handler complete.

    async def _cancel_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        trace = self._create_trace_if_configured(update, "cancel_command")
        user_id = str(update.effective_chat.id)
        logger.info(f"/cancel command received from user_id: {user_id}")
        try:
            if self.processor and self.processor.state_manager:
                # Use the processor's command handling logic for /cancel
                response_text = await self.processor.process_command(user_id, "telegram", "/cancel")
                await update.message.reply_text(response_text)
                if trace: trace.update(output={"status": "success", "response": response_text})
            else:
                await update.message.reply_text("Sorry, the bill processor is not available to cancel.")
                if trace: trace.update(level="ERROR", status_message="Processor not available for /cancel")
        except Exception as e:
            if trace: trace.update(level="ERROR", status_message=str(e))
            logger.error(f"Error in /cancel command for user {user_id}: {e}", exc_info=True)
            await update.message.reply_text("Sorry, there was an issue cancelling. You can try /start to begin fresh.")

    async def _status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        trace = self._create_trace_if_configured(update, "status_command")
        user_id = str(update.effective_chat.id)
        logger.info(f"/status command received from user_id: {user_id}")
        try:
            if self.processor and self.processor.state_manager:
                 # Use the processor's command handling logic for /status
                response_text = await self.processor.process_command(user_id, "telegram", "/status")
                await update.message.reply_text(response_text)
                if trace: trace.update(output={"status": "success", "response": response_text})
            else:
                await update.message.reply_text("Sorry, the bill processor is not available to check status.")
                if trace: trace.update(level="ERROR", status_message="Processor not available for /status")
        except Exception as e:
            if trace: trace.update(level="ERROR", status_message=str(e))
            logger.error(f"Error in /status command for user {user_id}: {e}", exc_info=True)
            await update.message.reply_text("Sorry, there was an issue checking the status.")

    async def _handle_text_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handles incoming text messages."""
        # Remove trace from here as it's now in _process_input
        # trace = self._create_trace_if_configured(update, "handle_text_message")
        try:
            user_id = str(update.effective_chat.id)
            text = update.message.text
            logger.info(f"Text message received from user_id {user_id}: {text:.50}...") # Log snippet
            logger.debug(f"Full Update object for text message: {update.to_json()}")
            
            await self._process_input(update, context) # No image_bytes for text message
            # if trace: trace.update(output={"status": "success", "processed_text_length": len(text)}) # Trace in _process_input
        except Exception as e:
            # if trace: trace.update(level="ERROR", status_message=str(e), output={"status": "error"})
            logger.error(f"Error handling text message for user {user_id}: {e}", exc_info=True)
            # _process_input should handle replying to the user on error
            # Do not re-raise to prevent duplicate error messages or Telegram retries.

    async def _handle_photo_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handles incoming photo messages (compressed images)."""
        # trace = self._create_trace_if_configured(update, "handle_photo_message") # Trace in _process_input
        try:
            user_id = str(update.effective_chat.id)
            logger.info(f"Photo message received from user_id {user_id}")
            logger.debug(f"Full Update object for photo message: {update.to_json()}")
            
            # Get the largest photo (best quality)
            photo_file = await update.message.photo[-1].get_file()
            image_bytearray_content = await photo_file.download_as_bytearray()
            # image_bytes = bytes(image_bytearray_content) # This will be done in _process_input
            
            await self._process_input(update, context, image_bytes=bytes(image_bytearray_content))
            # if trace: trace.update(output={"status": "success", "image_size_bytes": len(image_bytes)}) # Trace in _process_input
        except Exception as e:
            # if trace: trace.update(level="ERROR", status_message=str(e), output={"status": "error"})
            logger.error(f"Error handling photo message for user {user_id}: {e}", exc_info=True)
            # _process_input should handle replying to the user on error

    async def _handle_document_image_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handles image sent as a document (uncompressed)."""
        trace = self._create_trace_if_configured(update, "handle_document_image_message")
        try:
            user_id = str(update.effective_chat.id)
            document = update.message.document
            logger.info(f"Document received from user_id {user_id}: Name: {document.file_name}, MIME Type: {document.mime_type}, Size: {document.file_size} bytes") # Log more info
            logger.debug(f"Full Update object for document message: {update.to_json()}")

            # Check for common image MIME types
            image_mime_types = ["image/jpeg", "image/png", "image/gif", "image/bmp", "image/webp"]
            
            if document.mime_type and document.mime_type.lower() in image_mime_types:
                logger.info(f"Image document ({document.mime_type}) received from user_id {user_id}: {document.file_name}")
                doc_file = await document.get_file()
                image_bytearray = await doc_file.download_as_bytearray()
                # image_bytes = bytes(image_bytearray) # Already passed to _process_input
                
                # Call _process_with_processor directly, passing the image_bytes
                # This centralizes the logic of state checking and calling the actual BillSplitProcessor
                # We need to ensure _process_with_processor can be called with image_bytes directly
                # (It's designed to get them from update.message.photo OR be passed them)
                
                # Create a pseudo-update or directly call a modified _process_with_processor if needed.
                # For now, let's assume the structure of _process_with_processor correctly handles
                # being called when update.message.photo is None but image_bytes are provided.
                
                # The _process_with_processor method is designed to get image_bytes if update.message.photo exists.
                # If we call it from here, update.message.photo will be None. We need to ensure it uses the passed image_bytes.
                # Let's adapt _process_with_processor slightly or ensure this call path is handled correctly.
                # For now, directly calling the processor as it was, but ideally, the _process_with_processor itself
                # would just take image_bytes as an optional argument directly if it's cleaner.

                # Simplified: Call a common processing method that expects image_bytes if available
                await self._process_input(update, context, image_bytes=bytes(image_bytearray))
                if trace: trace.update(output={"status": "success", "image_size_bytes": len(image_bytearray), "filename": document.file_name})
            elif document.mime_type and "image" in document.mime_type.lower():
                # Catching other image/* types that might not be in the explicit list
                logger.warning(f"Received document from {user_id} with a less common image MIME type: {document.mime_type}. Attempting to process.")
                doc_file = await document.get_file()
                image_bytearray = await doc_file.download_as_bytearray()
                await self._process_input(update, context, image_bytes=bytes(image_bytearray))
                if trace: trace.update(output={"status": "success_uncommon_mime", "image_size_bytes": len(image_bytearray), "filename": document.file_name})
            else:
                logger.warning(f"Received document from {user_id} that is not a recognized image type: Name: {document.file_name}, MIME: {document.mime_type}")
                await update.message.reply_text("It looks like you sent a file, but I can only process images (like JPG, PNG). Please send an image of your bill.")
                if trace: trace.update(output={"status": "rejected", "reason": "Not an image document", "mime_type": document.mime_type})
        except Exception as e:
            if trace: trace.update(level="ERROR", status_message=str(e), output={"status": "error"})
            logger.error(f"Error handling document image for user {user_id}: {e}", exc_info=True)
            await update.message.reply_text("Sorry, I had trouble handling the file you sent. Please try sending it as a photo or ensure it's a standard image file.")
            # Do not re-raise, let Telegram handler complete to avoid retries from Telegram.

    async def _error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Log Errors caused by Updates."""
        logger.error(f"Update {update} caused error {context.error}", exc_info=context.error)
        # Optionally, inform the user that an error occurred
        if isinstance(update, Update) and update.effective_chat:
            try:
                await self.send_message(str(update.effective_chat.id), Message(text="Sorry, an unexpected error occurred."))
            except Exception as e:
                logger.error(f"Failed to send error message to user {update.effective_chat.id}: {e}")

    async def send_message(self, user_id: str, message: Message):
        """Sends a text message to a user via Telegram."""
        try:
            await self.bot.send_message(chat_id=user_id, text=message.text)
            logger.info(f"Message sent to Telegram user_id {user_id}: {message.text}")
        except Exception as e:
            logger.error(f"Error sending Telegram message to {user_id}: {e}")
            # Implement retry mechanisms or specific error handling if needed

    async def send_image(self, user_id: str, image_path: str, caption: str = None): # Changed image: bytes to image_path for simplicity
        """Sends an image to a user via Telegram."""
        # For sending bytes, use InputFile(image_bytes)
        # For this example, assuming image_path. OCR service would provide bytes.
        try:
            await self.bot.send_photo(chat_id=user_id, photo=open(image_path, 'rb'), caption=caption)
            logger.info(f"Image sent to Telegram user_id {user_id} from path: {image_path}")
        except Exception as e:
            logger.error(f"Error sending Telegram image to {user_id}: {e}")
    
    # --- Interface Methods Implementation ---
    
    async def handle_incoming(self, data: dict):
        """
        Handles incoming webhook data from Telegram.
        This method will parse the data and trigger appropriate actions via the Application.
        """
        if not self._is_initialized:
            logger.error("Telegram application is not initialized. Cannot process update.")
            # Potentially re-raise an exception or return an error response if this path is hit
            # This shouldn't happen if lifespan initialization is correct.
            raise RuntimeError("TelegramInterface.handle_incoming called before application was initialized.")

        try:
            update = Update.de_json(data, self.bot)
            # Log the deserialized Update object for debugging
            logger.debug(f"Processing deserialized Telegram Update object: {update.to_json()}") 
            await self.application.process_update(update)
        except Exception as e:
            logger.error(f"Error processing incoming Telegram webhook data: {e} - Data: {data}", exc_info=True)
            # Do not re-raise here if FastAPI should return 200 to Telegram to prevent retries.
            # Or, re-raise if the error should result in a 500.

    async def run_polling(self, drop_pending_updates=True):
        """Starts the bot in polling mode."""
        logger.info("Starting Telegram bot in polling mode...")
        if not self._is_initialized:
            await self.initialize_application() # Ensure initialized for polling too
        if not self.processor: # Added check for polling mode
            logger.error("BillSplitProcessor not available. Polling mode cannot fully function.")
            # Decide if polling should even start without a processor
            # For now, it will start but handlers will fail to process logically.
        await self.application.start() 
        await self.application.run_polling(drop_pending_updates=drop_pending_updates)

    async def run_webhook(self, listen: str = "0.0.0.0", port: int = settings.TELEGRAM_WEBHOOK_PORT, 
                          webhook_url: str = settings.TELEGRAM_WEBHOOK_URL):
        """Starts the bot in webhook mode. Ensure the webhook is set on Telegram's side."""
        logger.info(f"Starting Telegram bot in webhook mode. Listening on {listen}:{port}. URL: {webhook_url}")
        await self.application.bot.set_webhook(url=webhook_url)
        # The PTB Application's run_webhook is for development, not recommended for production.
        # For production, integrate with a web server like Uvicorn/Gunicorn
        # This simplified run_webhook is for development. A proper setup would involve
        # self.application.initialize()
        # self.application.start()
        # And then the web server (FastAPI) would forward requests to self.application.process_update
        # For now, we'll assume the FastAPI endpoint will call `handle_incoming`.
        # This method `run_webhook` is more of a setup for the webhook URL.
        logger.info(f"Telegram webhook set to {webhook_url}. Ensure your FastAPI app handles POSTs to this URL.")
        # For actual webhook server, PTB recommends integrating with a web framework.
        # Our FastAPI app will serve this role. This function primarily sets the webhook.

    # Placeholder for webhook verification, if Telegram uses a similar mechanism to WhatsApp's GET request.
    # Telegram webhooks are simpler; they just POST to your URL.
    # The "secret_token" parameter in set_webhook can be used for verification.
    def verify_webhook_request(self, request_headers: dict, request_body: bytes) -> bool:
        """
        Verifies if the incoming webhook request is genuinely from Telegram.
        This can be done using a secret token.
        """
        # Example: Check for a secret token in headers
        # if 'X-Telegram-Bot-Api-Secret-Token' in request_headers:
        #     if request_headers['X-Telegram-Bot-Api-Secret-Token'] == settings.TELEGRAM_WEBHOOK_SECRET:
        #         return True
        # logger.warning("Webhook verification failed: Secret token mismatch or not provided.")
        # return False
        # For now, assuming verification is handled if a secret token is set via set_webhook
        # or if the URL is secret enough.
        logger.info("Telegram webhook request received. Basic verification (e.g., secret token) should be implemented if configured.")
        return True # Placeholder

    async def set_telegram_webhook(self, webhook_url: str, secret_token: str | None = None):
        if not self._is_initialized:
            logger.warning("Attempting to set webhook, but PTB application not yet initialized by lifespan. Initializing now...")
            await self.initialize_application() # Ensure initialized before setting webhook

        logger.info(f"Setting Telegram webhook to: {webhook_url}")
        try:
            await self.application.bot.set_webhook(
                url=webhook_url,
                secret_token=secret_token
            )
            logger.info(f"Telegram webhook set to {webhook_url} successfully.")
        except Exception as e:
            logger.error(f"Failed to set Telegram webhook: {e}", exc_info=True)
            raise

    async def cleanup(self):
        try:
            if self.application:
                await self.application.stop()
                await self.application.shutdown()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        finally:
            self.application = None

# # Example of how to use it (typically in main.py)
# if __name__ == '__main__':
#     # This is for standalone testing of the interface
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
#     # Mock settings if not running within the full app context
#     class MockSettings:
#         TELEGRAM_BOT_TOKEN = "YOUR_ACTUAL_BOT_TOKEN" # Replace with your token for testing
#         TELEGRAM_WEBHOOK_PORT = 8443
#         TELEGRAM_WEBHOOK_URL = "https://your.domain.com/webhooks/telegram" # Replace
#         # Add other settings if your interface uses them directly for testing

#     # settings = MockSettings() # Uncomment for standalone testing
    
#     if not settings.TELEGRAM_BOT_TOKEN or settings.TELEGRAM_BOT_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN_HERE":
#         logger.error("Telegram Bot Token is not configured. Please set TELEGRAM_BOT_TOKEN in your .env or config.")
#     else:
#         async def main_test(): # Wrapped in async main for await
#             telegram_interface = TelegramInterface(token=settings.TELEGRAM_BOT_TOKEN)
#             await telegram_interface.initialize_application() # Initialize before polling
#             logger.info("TelegramInterface initialized. Starting polling for testing...")
#             await telegram_interface.run_polling() # run_polling is async
        
#         asyncio.run(main_test()) 