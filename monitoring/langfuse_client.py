from langfuse import Langfuse # type: ignore
from functools import wraps
import time
import logging
from typing import Optional, TypeVar, Any, Callable, Dict
from inspect import signature

import json
from config.settings import settings # For default values if not passed in init

logger = logging.getLogger(__name__)

# Type variable for decorator
T = TypeVar('T')

class LangfuseMonitor:
    _instance = None

    # Making LangfuseMonitor a singleton as typically one instance is used per application.
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LangfuseMonitor, cls).__new__(cls)
            # Initialization logic that should only run once.
            try:
                # Prioritize kwargs if provided, otherwise use settings
                public_key = kwargs.get('public_key', settings.LANGFUSE_PUBLIC_KEY_1)
                secret_key = kwargs.get('secret_key', settings.LANGFUSE_SECRET_KEY_1)
                host = kwargs.get('host', settings.LANGFUSE_HOST)
                
                cls._instance.client = Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host,
                    # debug=True # Uncomment for verbose Langfuse SDK logging
                )
                logger.info(f"Langfuse client initialized. Host: {host}")
                # Perform a simple check, like flushing, to see if client is operational
                # This is a synchronous operation, consider if it's okay at startup.
                cls._instance.client.flush() 
                logger.info("Langfuse client seems operational.")

            except Exception as e:
                logger.error(f"Failed to initialize Langfuse client: {e}", exc_info=True)
                # Handle failed initialization, e.g. set client to None or use a mock/dummy client
                cls._instance.client = None # No monitoring if this fails
        return cls._instance

    # This __init__ will be called every time LangfuseMonitor() is invoked,
    # but the actual client init happens in __new__ only once.
    def __init__(self, public_key: str = None, secret_key: str = None, host: str = None):
        # Parameters are optional here; if not provided, __new__ uses settings.
        # If self.client is None here, it means initialization failed in __new__.
        if not hasattr(self, 'client') or self.client is None:
            # This ensures that if __new__ failed, we re-attempt or log issue.
            # However, with the current __new__, self.client should always be set (even to None).
            logger.warning("Langfuse client was not properly initialized during __new__.")
            # Potentially try to re-initialize or ensure it's None
            # For safety, ensure client attribute exists, even if None
            if not hasattr(self, 'client'): 
                self.client = None

    def get_client(self) -> Optional[Langfuse]:
        """Returns the active Langfuse client, or None if initialization failed."""
        return self.client

    def trace_conversation(self, conversation_id: str, user_id: Optional[str] = None, **kwargs):
        """Create a trace for an entire conversation."""
        if not self.client:
            logger.warning("Langfuse client not available. Cannot trace conversation.")
            return None # Or a dummy trace object if your code expects one
        
        # PRD shows name="bill_split_conversation" as hardcoded, allow override via kwargs
        trace_name = kwargs.pop("name", "bill_split_conversation")
        
        try:
            # Ensure trace_id is used if provided, matching Langfuse SDK behavior for get_trace
            trace_id = kwargs.pop("id", conversation_id) # Use conversation_id if id not in kwargs

            # Check if a trace with this ID already exists
            # The Python SDK's get_trace creates if not exists, or gets if exists.
            # So, we can directly use it.
            trace = self.client.trace(
                id=trace_id, 
                name=trace_name, 
                user_id=user_id, # Add user_id if available
                # metadata=kwargs.pop("metadata", {}), # Pass other kwargs as metadata
                **kwargs # Pass remaining kwargs
            )
            logger.info(f"Langfuse trace created/retrieved for ID: {trace_id}, Name: {trace_name}, User: {user_id}")
            return trace
        except Exception as e:
            logger.error(f"Error creating/retrieving Langfuse trace for ID {conversation_id}: {e}", exc_info=True)
            return None

    def _prepare_input_for_langfuse(self, func: Callable, args: tuple, kwargs: dict) -> Any:
        """Prepare input for Langfuse, attempting direct serialization, then named dict, then stringified."""
        input_data: Any = None
        try:
            # Try to create a dictionary with named arguments if possible
            sig = signature(func)
            bound_args = sig.bind_partial(*args, **kwargs)
            input_data = {name: val for name, val in bound_args.arguments.items()}
            json.dumps(input_data) # Test serializability
        except (TypeError, ValueError):
            # If direct dict of args fails (e.g. non-serializable), fallback
            logger.debug(f"Could not directly serialize input args/kwargs for {func.__name__} as dict. Trying simplified.")
            simple_input = {}
            if args:
                simple_input["args"] = [str(arg) for arg in args]
            if kwargs:
                simple_input["kwargs"] = {k: str(v) for k, v in kwargs.items()}
            
            try:
                json.dumps(simple_input)
                input_data = simple_input
            except TypeError:
                logger.warning(f"Could not serialize simplified inputs for Langfuse span {func.__name__}. Fallback to string of args.")
                input_data = f"args: {str(args)}, kwargs: {str(kwargs)}"
        return input_data

    def _prepare_output_for_langfuse(self, output: Any, func_name: str) -> Any:
        """Prepare output for Langfuse, attempting direct serialization, then stringified."""
        try:
            json.dumps(output) # Test serializability
            return output
        except TypeError:
            logger.warning(f"Output of {func_name} is not directly JSON serializable for Langfuse. Storing as string.")
            return str(output)

    def observe_function(self, metadata: Optional[Dict[str, Any]] = None):
        """Decorator to monitor function calls, creating spans within an active trace."""
        def decorator(func: Callable[..., Any]):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                if not self.client:
                    logger.warning(f"Langfuse client not available. Cannot observe {func.__name__}. Calling original function.")
                    return await func(*args, **kwargs)

                span = None
                start_time = time.time()
                try:
                    prepared_input = self._prepare_input_for_langfuse(func, args, kwargs)
                    
                    # Merge decorator metadata with any runtime metadata if passed
                    # For now, just use decorator metadata
                    span_metadata = metadata or {}

                    span = self.client.span(
                        name=func.__name__,
                        input=prepared_input,
                        metadata=span_metadata
                        # session_id, user_id can also be set on span if available and relevant
                    )
                    
                    result = await func(*args, **kwargs)
                    
                    prepared_output = self._prepare_output_for_langfuse(result, func.__name__)
                    
                    if span: span.end(output=prepared_output)
                    return result
                except Exception as e:
                    logger.error(f"Exception in observed function {func.__name__}: {e}", exc_info=True)
                    if span: span.end(level="ERROR", status_message=str(e))
                    raise
                finally:
                    if span:
                        duration = time.time() - start_time
                        try:
                            self.client.score(
                                trace_id=span.trace_id, 
                                name="latency_seconds", # More specific name
                                value=duration,
                                comment=f"Latency for {func.__name__}"
                            )
                        except Exception as e_score:
                            logger.error(f"Failed to score latency for {func.__name__} on trace {span.trace_id}: {e_score}")
            return wrapper
        return decorator

    def flush(self):
        if self.client:
            try:
                self.client.flush()
                logger.info("Langfuse client flushed.")
            except Exception as e:
                logger.error(f"Error flushing Langfuse client: {e}")

# Global instance, initialized using settings by default
# langfuse_monitor = LangfuseMonitor() 