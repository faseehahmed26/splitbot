from typing import Dict, Any, Optional, Tuple # Import Tuple
import redis # type: ignore
from redis.asyncio import Redis as AsyncRedis
from core.fallback_state_manager import InMemoryStateStore # Import fallback store
from monitoring.langfuse_client import LangfuseMonitor # Import LangfuseMonitor
import json
import logging
import time # Added for last_updated timestamp

logger = logging.getLogger(__name__)

class ConversationState:
    def __init__(self, 
                 redis_client: Optional[AsyncRedis], 
                 fallback_store: Optional[InMemoryStateStore] = None, # Add fallback_store parameter
                 langfuse_monitor: Optional[LangfuseMonitor] = None, # Add langfuse_monitor
                 ttl: int = 3600):
        """
        Initialize ConversationState.
        redis_client: An instance of redis.asyncio.Redis.
        fallback_store: An instance of a fallback state store (e.g., InMemoryStateStore).
        langfuse_monitor: An instance of LangfuseMonitor.
        ttl: Time-to-live for state keys in seconds (default 1 hour).
        """
        self.redis = redis_client
        self.fallback_store = fallback_store
        self.langfuse_monitor = langfuse_monitor # Store it
        self.ttl = ttl
        if self.redis:
            logger.info(f"ConversationState initialized with Redis client. TTL: {ttl}s")
        if self.fallback_store:
            logger.info(f"ConversationState initialized with Fallback store. TTL: {ttl}s")
        if self.langfuse_monitor:
            logger.info(f"ConversationState initialized with LangfuseMonitor.")
        if not self.redis and not self.fallback_store:
            logger.warning("ConversationState initialized with NO Redis client AND NO Fallback store. State will not be managed.")
        elif not self.redis:
            logger.warning(f"ConversationState initialized WITHOUT Redis client. Using Fallback store ONLY. TTL: {ttl}s")

    @property
    def observe(self):
        if self.langfuse_monitor and self.langfuse_monitor.initialized: # Check if initialized
            return self.langfuse_monitor.observe_function
        def dummy_decorator(metadata=None):
            def decorator(func):
                return func
            return decorator
        return dummy_decorator
        
    async def get_state(self, user_id: str) -> Tuple[Optional[Dict[str, Any]], str, Optional[str]]:
        """
        Get user state from available stores.
        Returns: (state_data, source, error_message)
        """
        # @self.observe(metadata={"category": "state_management", "operation": "get_state"}) # Ignoring langfuse for now
        async def _decorated_get_state():
            state_data: Optional[Dict[str, Any]] = None
            source: str = "unknown"
            error_message: Optional[str] = None

            if self.redis:
                state_key = f"state:{user_id}"
                redis_data_bytes = None
                try:
                    redis_data_bytes = await self.redis.get(state_key)
                    if redis_data_bytes:
                        state_data = json.loads(redis_data_bytes)
                        source = "Redis"
                        logger.debug(f"State retrieved from Redis for user {user_id}")
                        if self.fallback_store and state_data is not None:
                            await self.fallback_store.clear_state(user_id)
                            logger.debug(f"Cleared fallback state for {user_id} after Redis read.")
                except redis.exceptions.ConnectionError as e: # More specific error for connection issues
                    error_message = f"Redis connection error: {str(e)}"
                    logger.error(f"Redis connection error getting state for {user_id}: {e}. Attempting fallback.")
                except redis.exceptions.RedisError as e:
                    error_message = f"Redis error: {str(e)}"
                    logger.error(f"Redis error getting state for {user_id}: {e}. Attempting fallback.")
                except json.JSONDecodeError as e:
                    error_message = f"JSON decode error: {str(e)}"
                    logger.error(f"JSON decode error for Redis state of {user_id}: {e}. Data: {redis_data_bytes}. Attempting fallback.")
            
            if state_data is None and self.fallback_store:
                logger.debug(f"Attempting to get state from fallback for user {user_id}")
                try:
                    state_data = await self.fallback_store.get_state(user_id)
                    if state_data:
                        source = "Fallback"
                        error_message = None # Clear previous Redis error if fallback succeeded
                        logger.info(f"State retrieved from Fallback for user {user_id}")
                except Exception as e:
                    error_message = f"Fallback store error: {str(e)}" # Keep error if fallback also fails
                    logger.error(f"Fallback store error getting state for {user_id}: {e}")

            if state_data is None and not error_message: # If no data and no specific error yet
                source = "None"
                logger.debug(f"No state found in Redis or Fallback for user {user_id}")
            elif state_data is None and error_message:
                 logger.warning(f"Failed to retrieve state for user {user_id} from any source. Error: {error_message}")
            
            return state_data, source, error_message
        return await _decorated_get_state()

    async def update_state(self, user_id: str, state: Dict[str, Any]) -> Tuple[bool, str, Optional[str]]:
        """
        Update user state in available stores.
        Returns: (success, source, error_message)
        """
        # @self.observe(metadata={"category": "state_management", "operation": "update_state"}) # Ignoring langfuse for now
        async def _decorated_update_state():
            success = False
            source = "unknown"
            error_message = None
            state["last_updated"] = time.time() # Ensure last_updated is set

            if self.redis:
                state_key = f"state:{user_id}"
                try:
                    await self.redis.setex(
                        state_key,
                        self.ttl,
                        json.dumps(state)
                    )
                    logger.debug(f"State updated in Redis for user {user_id}")
                    success = True
                    source = "Redis"
                    if self.fallback_store:
                        await self.fallback_store.clear_state(user_id)
                        logger.debug(f"Cleared fallback state for {user_id} after Redis write.")
                except redis.exceptions.ConnectionError as e:
                    error_message = f"Redis connection error: {str(e)}"
                    logger.error(f"Redis connection error updating state for {user_id}: {e}. Attempting fallback write.")
                except redis.exceptions.RedisError as e:
                    error_message = f"Redis error: {str(e)}"
                    logger.error(f"Redis error updating state for {user_id}: {e}. Attempting fallback write.")
                except TypeError as e: 
                     error_message = f"TypeError serializing state: {str(e)}"
                     logger.error(f"TypeError serializing state for Redis {user_id}: {e}. State (first 200 chars): {str(state)[:200]}. Attempting fallback write.")
            
            if not success and self.fallback_store:
                logger.info(f"Updating state in Fallback for user {user_id} (Redis failed or unavailable)")
                try:
                    await self.fallback_store.update_state(user_id, state, self.ttl)
                    success = True
                    source = "Fallback"
                    error_message = None # Clear Redis error if fallback succeeded
                except Exception as e:
                    error_message = f"Fallback store error: {str(e)}" # Keep or set error if fallback also fails
                    logger.error(f"Fallback store error updating state for {user_id}: {e}")
            elif not self.redis and not self.fallback_store:
                source = "None"
                error_message = "No storage available"
                logger.error(f"Cannot update state for {user_id}: No Redis and No Fallback store available.")
            
            return success, source, error_message
        return await _decorated_update_state()

    async def clear_state(self, user_id: str) -> Tuple[bool, str, Optional[str]]:
        """
        Clears user state from available stores.
        Returns: (success, source, error_message)
        """
        # @self.observe(metadata={"category": "state_management", "operation": "clear_state"}) # Ignoring langfuse for now
        async def _decorated_clear_state():
            success = False # Overall success
            source = "unknown"
            error_message = None
            
            redis_cleared_successfully = False
            if self.redis:
                state_key = f"state:{user_id}"
                try:
                    await self.redis.delete(state_key)
                    logger.info(f"State cleared from Redis for user {user_id}")
                    redis_cleared_successfully = True
                    source = "Redis"
                except redis.exceptions.ConnectionError as e:
                    error_message = f"Redis connection error: {str(e)}"
                    logger.error(f"Redis connection error clearing state for {user_id}: {e}. Will still attempt fallback clear.")
                except redis.exceptions.RedisError as e:
                    error_message = f"Redis error: {str(e)}"
                    logger.error(f"Redis error clearing state for {user_id}: {e}. Will still attempt fallback clear.")
            
            fallback_cleared_successfully = False
            if self.fallback_store:
                logger.debug(f"Attempting to clear state from fallback for user {user_id}")
                try:
                    await self.fallback_store.clear_state(user_id)
                    fallback_cleared_successfully = True
                    if source == "unknown" or not redis_cleared_successfully: # If Redis didn't run or failed
                        source = "Fallback"
                        error_message = None # Clear Redis error if fallback succeeded
                except Exception as e:
                    if not error_message: # Don't overwrite a Redis error
                        error_message = f"Fallback store error: {str(e)}"
                    logger.error(f"Fallback store error clearing state for {user_id}: {e}")
            
            success = redis_cleared_successfully or fallback_cleared_successfully
            if not self.redis and not self.fallback_store:
                source = "None"
                error_message = "No storage available"
                logger.warning(f"Cannot clear state for {user_id}: No Redis and No Fallback store available.")
            
            return success, source, error_message
        return await _decorated_clear_state()


# Example state structure (from PRD)
# state = {
#     "step": "awaiting_participants",
#     "receipt_data": {
#         "total": 67.89,
#         "items": [...],
#         "currency": "USD"
#     },
#     "participants": ["John", "Sarah"],
#     "splits": {},
#     "conversation_id": "uuid",
#     "platform": "whatsapp"
# } 