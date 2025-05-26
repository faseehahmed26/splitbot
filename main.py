from fastapi import FastAPI, Request, Response, HTTPException
# from fastapi.responses import PlainTextResponse # No longer needed for WhatsApp GET
import logging
import asyncio # Added for startup event
import json # For pretty printing JSON
from contextlib import asynccontextmanager # Import asynccontextmanager
from typing import Optional
from pythonjsonlogger import jsonlogger # Import jsonlogger
from logging.handlers import TimedRotatingFileHandler # Import TimedRotatingFileHandler
import os # For creating log directory
import time # Added for timestamp

# from interfaces.whatsapp import WhatsAppInterface # Commented out
from interfaces.telegram import TelegramInterface
from config.settings import settings
from core.processor import BillSplitProcessor # Import BillSplitProcessor
from core.state_manager import ConversationState
from core.fallback_state_manager import InMemoryStateStore # Import InMemoryStateStore
from services.llm.base import LLMService # For type hinting
from services.ocr.base import OCRService # For type hinting
from monitoring.langfuse_client import LangfuseMonitor

import redis # type: ignore
from redis.asyncio import Redis as AsyncRedis  # Import AsyncRedis
from redis.asyncio.connection import ConnectionPool as AsyncConnectionPool # Import AsyncConnectionPool
from redis.retry import Retry
from redis.backoff import ExponentialBackoff

# --- Logging Setup ---
# Remove basicConfig if it was there, and set up handlers directly.
# logging.basicConfig(
#     level=logging.DEBUG, 
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S'
# )

# Get the root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG) # Set root logger level

# Remove any existing handlers to avoid duplicate logs if re-running this part of script
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Add a new handler with JsonFormatter
log_handler = logging.StreamHandler() # Outputs to stderr by default
formatter = jsonlogger.JsonFormatter(
    '%(asctime)s %(levelname)s %(name)s %(module)s %(funcName)s %(lineno)d %(message)s'
)
log_handler.setFormatter(formatter)
root_logger.addHandler(log_handler)

# Add TimedRotatingFileHandler if LOG_FILE_PATH is set
if settings.LOG_FILE_PATH:
    try:
        log_dir = os.path.dirname(settings.LOG_FILE_PATH)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            root_logger.info(f"Created log directory: {log_dir}")
        
        file_handler = TimedRotatingFileHandler(
            filename=settings.LOG_FILE_PATH,
            when=settings.LOG_ROTATION_WHEN,
            interval=settings.LOG_ROTATION_INTERVAL,
            backupCount=settings.LOG_ROTATION_BACKUP_COUNT,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter) # Use the same JSON formatter
        file_handler.setLevel(logging.DEBUG) # Set level for file handler, can be different from console
        root_logger.addHandler(file_handler)
        root_logger.info(f"Logging to file: {settings.LOG_FILE_PATH} with rotation.")
    except Exception as e:
        root_logger.error(f"Failed to configure file logging: {e}", exc_info=True)

logger = logging.getLogger(__name__) # Get logger for the current module
# You might want to set specific log levels for libraries if they are too noisy
# logging.getLogger("httpx").setLevel(logging.WARNING)
# logging.getLogger("telegram.ext").setLevel(logging.INFO)

# --- Global instances (to be initialized in lifespan or before) ---
telegram_interface: TelegramInterface | None = None
langfuse_client: any = None # Actual Langfuse client type
bill_processor: BillSplitProcessor | None = None # Add bill_processor global

# --- Service instances (initialized before lifespan uses them in processor) ---
# These are initialized once and then used by the processor.
# It's important they are ready before bill_processor is created.

# Initialize Langfuse client (used by services and processor)
langfuse_monitor_instance = LangfuseMonitor()

# Initialize services (LLM, OCR, StateManager)
llm_service_instance = None
settings.LLM_PROVIDER="gemini"
if settings.LLM_PROVIDER:
    logger.info(f"Attempting to initialize LLM provider: {settings.LLM_PROVIDER}")
    if settings.LLM_PROVIDER.lower() == "groq":
        from services.llm.groq import GroqService
        if settings.GROQ_API_KEY:
            try:
                llm_service_instance = GroqService(langfuse_monitor=langfuse_monitor_instance)
                logger.info(f"GroqService initialized with model {settings.GROQ_MODEL_NAME}.")
            except Exception as e:
                logger.error(f"Failed to initialize GroqService: {e}", exc_info=True)
        else:
            logger.error("GROQ_API_KEY not set, cannot initialize GroqService.")
    elif settings.LLM_PROVIDER.lower() == "gemini":
        from services.llm.gemini import GeminiService # Import GeminiService
        if settings.GEMINI_API_KEY_1: # Gemini uses GEMINI_API_KEY as per GeminiService __init__
            try:
                llm_service_instance = GeminiService(langfuse_monitor=langfuse_monitor_instance)
                logger.info(f"GeminiService initialized with model {settings.GEMINI_MODEL_NAME}.")
            except Exception as e:
                logger.error(f"Failed to initialize GeminiService: {e}", exc_info=True)
        else:
            logger.error("GEMINI_API_KEY not set (required for Gemini), cannot initialize GeminiService.")
    else:
        logger.error(f"Unsupported LLM_PROVIDER specified: {settings.LLM_PROVIDER}. Please choose 'groq' or 'gemini'.")
else:
    logger.warning("LLM_PROVIDER not set. LLM features will be unavailable.")

ocr_service_instance = None
if settings.OCR_PROVIDER:
    from services.ocr.google_vision import GoogleVisionOCR # Example
    try:
        ocr_service_instance = GoogleVisionOCR(langfuse_monitor=langfuse_monitor_instance)
        logger.info(f"OCRService ({settings.OCR_PROVIDER}) initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize OCRService ({settings.OCR_PROVIDER}): {e}")
else:
    logger.warning("OCR_PROVIDER not set. OCR features will be unavailable.")

# Initialize Async Redis Client
redis_client_instance: Optional[AsyncRedis] = None
if settings.REDIS_URL:
    try:
        # Configure retry strategy
        # Retry up to 3 times with exponential backoff (0.1s, 0.2s, 0.4s approx)
        # Retries on ConnectionError and TimeoutError by default.
        # We can add more errors to retry_on_error list if needed.
        retry_strategy = Retry(ExponentialBackoff(cap=0.5, base=0.1), 3)
        
        # Create an async connection pool
        # from_url will create a connection pool implicitly, but if we want to pass
        # retry to the pool's connections, it's better to create pool explicitly
        # or pass retry to from_url if it supports it directly for connections.
        # As of redis-py 5.0, `from_url` passes kwargs to the Redis client, which then uses them for its pool.
        
        # It's simpler to pass retry directly to from_url if that correctly configures
        # the underlying connections/pool. Let's assume from_url handles this.
        # If not, explicit pool creation would be:
        # pool = AsyncConnectionPool.from_url(settings.REDIS_URL, decode_responses=True)
        # redis_client_instance = AsyncRedis(connection_pool=pool, retry=retry_strategy)

        redis_client_instance = AsyncRedis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            retry=retry_strategy,
            # Add other relevant parameters like health_check_interval if desired
            # health_check_interval=30 # seconds
        )
        # As ping is async now, we'd need an async context to call it here.
        # This check will be moved to the lifespan's startup.
        # await redis_client_instance.ping() # This needs to be in an async function
        logger.info("Async Redis client configured, connection will be tested in lifespan.")
    except Exception as e:
        logger.error(f"Could not configure Async Redis client: {e}. State management will be impaired.")
        redis_client_instance = None # Ensure it's None on failure
else:
    logger.warning("REDIS_URL not set. Redis client will not be initialized.")
    redis_client_instance = None

# Initialize Fallback State Store
fallback_store_instance = InMemoryStateStore(backup_filepath=settings.FALLBACK_STATE_FILEPATH)

# Initialize ConversationState with Redis client (if available), Fallback store, and Langfuse monitor
state_manager_instance = ConversationState(
    redis_client=redis_client_instance, 
    fallback_store=fallback_store_instance,
    langfuse_monitor=langfuse_monitor_instance
)

if not state_manager_instance.redis and not state_manager_instance.fallback_store:
    # This case should ideally not happen if InMemoryStateStore is always initialized.
    # Logging in ConversationState __init__ covers individual store failures.
    logger.critical("CRITICAL: StateManager initialized with NO actual stores (Redis or Fallback). This should not happen.")
elif not state_manager_instance.redis:
    logger.info("StateManager initialized to use Fallback store ONLY (Redis unavailable or not configured).")
else:
    logger.info("StateManager initialized (potentially with Redis and Fallback).")

@asynccontextmanager
async def lifespan(app_param: FastAPI):
    global telegram_interface, langfuse_client, bill_processor # Add bill_processor to globals
    
    # Initialize BillSplitProcessor with pre-initialized services
    # BillSplitProcessor needs the raw langfuse_client if it makes direct SDK calls, 
    # or langfuse_monitor if it uses decorators internally.
    # Current BillSplitProcessor constructor takes langfuse_client (SDK client).
    raw_langfuse_sdk_client = langfuse_monitor_instance.get_client() # Get for processor

    if llm_service_instance and ocr_service_instance and state_manager_instance and raw_langfuse_sdk_client:
        bill_processor = BillSplitProcessor(
            llm_service=llm_service_instance,
            ocr_service=ocr_service_instance,
            state_manager=state_manager_instance,
            langfuse_client=raw_langfuse_sdk_client # Pass the SDK client here
        )
        logger.info("BillSplitProcessor initialized successfully.")
    else:
        # Log detailed reasons for failure
        missing_deps = []
        if not llm_service_instance: missing_deps.append("LLMService")
        if not ocr_service_instance: missing_deps.append("OCRService")
        if not state_manager_instance: missing_deps.append("StateManager")
        if not raw_langfuse_sdk_client: missing_deps.append("LangfuseClient (SDK)")
        logger.error(f"BillSplitProcessor could not be initialized. Missing: {', '.join(missing_deps)}.")
        bill_processor = None

    # Initialize TelegramInterface instance, now passing the processor and langfuse_monitor
    if settings.TELEGRAM_BOT_TOKEN:
        if bill_processor: 
            telegram_interface = TelegramInterface(
                token=settings.TELEGRAM_BOT_TOKEN,
                processor=bill_processor,
                langfuse_monitor=langfuse_monitor_instance # Pass monitor
            )
            try:
                await telegram_interface.initialize_application()
                if settings.TELEGRAM_WEBHOOK_URL:
                    await telegram_interface.set_telegram_webhook(
                        webhook_url=settings.TELEGRAM_WEBHOOK_URL,
                        secret_token=settings.TELEGRAM_WEBHOOK_SECRET
                    )
                else:
                    logger.warning("TELEGRAM_WEBHOOK_URL not set. Webhook for Telegram bot will not be configured.")
            except Exception as e:
                logger.error(f"Error during Telegram interface setup: {e}", exc_info=True)
                telegram_interface = None # Failed to initialize or set webhook
        else:
            logger.error("Telegram interface cannot be initialized because BillSplitProcessor failed to initialize.")
            telegram_interface = None
    else:
        logger.error("TELEGRAM_BOT_TOKEN not set. Telegram interface cannot be initialized.")
        telegram_interface = None

    # Verify Google credentials (can remain here)
    if settings.OCR_PROVIDER == "google_vision" and settings.GOOGLE_APPLICATION_CREDENTIALS:
        credentials_path = settings.GOOGLE_APPLICATION_CREDENTIALS
        abs_credentials_path = os.path.abspath(credentials_path)
        if not os.path.exists(credentials_path):
            logger.error(f"Google credentials file NOT FOUND at configured path: {credentials_path} (Absolute: {abs_credentials_path})")
            logger.error("Please ensure GOOGLE_APPLICATION_CREDENTIALS in your .env file points to the correct JSON key file.")
        else:
            logger.info(f"Google credentials file found at: {credentials_path} (Absolute: {abs_credentials_path})")
    elif settings.OCR_PROVIDER == "google_vision" and not settings.GOOGLE_APPLICATION_CREDENTIALS:
        logger.warning("OCR_PROVIDER is 'google_vision' but GOOGLE_APPLICATION_CREDENTIALS is not set in .env. Google Vision OCR might not work.")
    
    # Assign pre-initialized langfuse client to the global name used by lifespan shutdown
    # langfuse_client = langfuse_client_instance # This global is the SDK client for shutdown. Monitor has it.
    langfuse_sdk_client_for_shutdown = langfuse_monitor_instance.get_client() # Get SDK client for shutdown
    logger.info("Application startup...")

    # Load in-memory state from file if configured
    if fallback_store_instance:
        await fallback_store_instance.load_from_file_on_startup()

    # Test Redis connection at startup
    if redis_client_instance:
        try:
            await redis_client_instance.ping()
            logger.info("Successfully connected to Async Redis.")
        except Exception as e:
            logger.error(f"Async Redis ping failed on startup: {e}. State management may be impaired.")
            # Depending on policy, we might choose to not start or set state_manager to a fallback here.
            # For now, we'll rely on ConversationState methods to handle runtime errors.
    else:
        logger.warning("Async Redis client not initialized. State management will be unavailable or use fallback if configured.")

    yield # FastAPI app runs after this yield

    logger.info("Application shutdown...")
    if telegram_interface:
        logger.info("Cleaning up Telegram interface...")
        await telegram_interface.cleanup()
    
    # Save in-memory state to file if configured
    if fallback_store_instance:
        await fallback_store_instance.save_to_file_on_shutdown()
    
    # Re-assign langfuse_client if it was initialized within a local scope 
    # or ensure it's accessible here if initialized globally earlier.
    # For this example, assuming langfuse_client was assigned globally.
    if langfuse_sdk_client_for_shutdown and hasattr(langfuse_sdk_client_for_shutdown, 'shutdown') and callable(langfuse_sdk_client_for_shutdown.shutdown):
        logger.info("Shutting down Langfuse client...")
        try:
            langfuse_sdk_client_for_shutdown.shutdown()
        except Exception as e:
            logger.error(f"Error shutting down Langfuse client: {e}", exc_info=True)

# Pass the lifespan context manager to FastAPI
app = FastAPI(debug=True, lifespan=lifespan)

# --- Remove standalone initializations of services and interfaces here ---
# They are now handled before or within the lifespan function.
# Specifically, langfuse_monitor, llm_service, ocr_service, redis_client, state_manager
# are initialized before lifespan. BillProcessor and TelegramInterface are initialized within lifespan.

# --- Webhook Endpoints ---

# @app.get("/webhooks/whatsapp", status_code=200) # Commented out
# async def whatsapp_webhook_get(request: Request):
#     logger.info(f"GET /webhooks/whatsapp verification request. Params: {request.query_params}")
#     challenge, status_code = whatsapp_interface.verify_webhook(request.query_params)
#     if isinstance(challenge, str):
#         return PlainTextResponse(content=challenge, status_code=status_code)
#     else:
#         raise HTTPException(status_code=status_code, detail="Webhook verification failed.")

# @app.post("/webhooks/whatsapp") # Commented out
# async def whatsapp_webhook_post(request: Request):
#     try:
#         data = await request.json()
#         logger.info(f"POST /webhooks/whatsapp received data: {data}")
#         await whatsapp_interface.handle_incoming(data)
#         return {"status": "ok"}
#     except Exception as e:
#         logger.error(f"Error processing WhatsApp webhook POST request: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/webhooks/telegram")
async def telegram_webhook_post(request: Request):
    """Telegram webhook endpoint for receiving messages."""
    if not telegram_interface:
        logger.error("Telegram interface not initialized. Cannot process webhook.")
        raise HTTPException(status_code=500, detail="Telegram interface not available.")
    try:
        data = await request.json()
        # Log the raw incoming data for debugging
        logger.debug(f"POST /webhooks/telegram received raw data: {json.dumps(data, indent=2)}")
        
        # --- Webhook Secret Verification ---
        if settings.TELEGRAM_WEBHOOK_SECRET:
            secret_token_header = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
            if secret_token_header != settings.TELEGRAM_WEBHOOK_SECRET:
                logger.warning(f"Telegram webhook secret token mismatch. Header: '{secret_token_header}', Expected: '{settings.TELEGRAM_WEBHOOK_SECRET}'")
                raise HTTPException(status_code=403, detail="Invalid secret token.")
            else:
                logger.debug("Telegram webhook secret token verified successfully.")
        # --- End Webhook Secret Verification ---

        await telegram_interface.handle_incoming(data)
        return {"status": "ok"}
    except json.JSONDecodeError:
        body = await request.body()
        logger.error(f"Failed to decode JSON from Telegram webhook. Body: {body.decode()}", exc_info=True)
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")
    except Exception as e:
        logger.error(f"Error processing Telegram webhook POST request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")
# In main.py, modify the health endpoint:
@app.get("/health")
async def health():
    redis_ok = False
    if redis_client_instance:
        try:
            await redis_client_instance.ping()
            redis_ok = True 
        except Exception as e:
            logger.debug(f"Health check: Redis ping failed: {e}")
            pass
    
    langfuse_ok = langfuse_monitor_instance is not None and langfuse_monitor_instance.get_client() is not None
    
    ocr_ready = ocr_service_instance is not None
    if ocr_service_instance and hasattr(ocr_service_instance, 'client') and ocr_service_instance.client is None:
        ocr_ready = False 

    llm_ready = llm_service_instance is not None
    if llm_service_instance and hasattr(llm_service_instance, 'client') and llm_service_instance.client is None:
        llm_ready = False

    # Fix: Use time.time() directly instead of asyncio.to_thread
    return {
        "status": "healthy",
        "timestamp": time.time(),  # Changed this line
        "services": {
            "llm_service": {
                "initialized": llm_service_instance is not None,
                "provider": settings.LLM_PROVIDER if llm_service_instance else "Not Configured",
                "client_ready": llm_ready
            },
            "ocr_service": {
                "initialized": ocr_service_instance is not None,
                "provider": settings.OCR_PROVIDER if ocr_service_instance else "Not Configured",
                "client_ready": ocr_ready
            },
            "redis": {
                "configured": settings.REDIS_URL is not None,
                "connected": redis_ok
            },
            "langfuse": {
                "configured": langfuse_monitor_instance is not None,
                "client_initialized": langfuse_ok
            },
            "state_manager": {
                "initialized": state_manager_instance is not None,
                "redis_in_use": state_manager_instance.redis is not None if state_manager_instance else False,
                "fallback_in_use": state_manager_instance.fallback_store is not None if state_manager_instance else False
            },
            "bill_processor": {
                "initialized": bill_processor is not None
            },
            "telegram_interface": {
                "initialized": telegram_interface is not None and telegram_interface._is_initialized
            }
        }
    }
# To run this app (after installing FastAPI and Uvicorn):
# uvicorn main:app --reload --port 8000 (or your configured port)
# Ensure .env file has TELEGRAM_BOT_TOKEN and optionally TELEGRAM_WEBHOOK_URL & TELEGRAM_WEBHOOK_SECRET 