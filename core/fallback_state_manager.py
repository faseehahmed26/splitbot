import asyncio
import time
from typing import Dict, Any, Optional, Tuple
import logging
import json # For file operations
import os # For checking file existence

logger = logging.getLogger(__name__)

class InMemoryStateStore:
    def __init__(self, backup_filepath: Optional[str] = None):
        """
        Initialize InMemoryStateStore.
        Uses a simple dictionary to store states with their expiry times.
        Optionally loads state from a backup file if backup_filepath is provided.
        """
        self.store: Dict[str, Tuple[Dict[str, Any], float]] = {} # {user_id: (state_data, expiry_timestamp)}
        self.backup_filepath = backup_filepath
        self._lock = asyncio.Lock() # To protect access to self.store during file ops
        
        if self.backup_filepath:
            # Run load_from_file in a way that doesn't block __init__ if it were async
            # For a synchronous __init__, direct call is fine, but we make it async
            # and usually, you'd schedule this with asyncio.create_task if __init__ itself couldn't be async.
            # However, since our lifespan context in main.py is async, we can await it there
            # or make __init__ itself async (which is not standard for constructors).
            # For simplicity, let's make a separate async method to call post-initialization.
            pass # Loading will be handled by an explicit call to load_from_file_on_startup

        logger.info(f"InMemoryStateStore initialized. Backup filepath: {self.backup_filepath if self.backup_filepath else 'Not configured'}")

    async def load_from_file_on_startup(self):
        if not self.backup_filepath or not await asyncio.to_thread(os.path.exists, self.backup_filepath):
            logger.info("No backup state file found or filepath not configured. Starting with empty in-memory store.")
            return

        async with self._lock:
            try:
                logger.info(f"Attempting to load in-memory state from {self.backup_filepath}")
                content = await asyncio.to_thread(self._read_file_sync, self.backup_filepath)
                if not content:
                    logger.warning(f"Backup file {self.backup_filepath} is empty.")
                    return
                
                backed_up_store = json.loads(content)
                loaded_states = 0
                expired_states = 0
                current_time = time.time()
                
                for user_id, (state_data, expiry_timestamp) in backed_up_store.items():
                    if expiry_timestamp > current_time:
                        self.store[user_id] = (state_data, expiry_timestamp)
                        loaded_states += 1
                    else:
                        expired_states += 1
                logger.info(f"Loaded {loaded_states} states from backup. Discarded {expired_states} expired states.")
            except FileNotFoundError:
                logger.info(f"Backup file {self.backup_filepath} not found. Starting fresh.")
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON from backup file {self.backup_filepath}: {e}. Starting fresh.")
                self.store = {} # Ensure store is empty on decode error
            except Exception as e:
                logger.error(f"Failed to load state from file {self.backup_filepath}: {e}. Starting fresh.")
                self.store = {} # Ensure store is empty on other errors

    def _read_file_sync(self, filepath: str) -> Optional[str]:
        # Helper synchronous method for reading file content
        try:
            with open(filepath, 'r') as f:
                return f.read()
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.error(f"Error reading file {filepath} synchronously: {e}")
            return None

    async def save_to_file_on_shutdown(self):
        if not self.backup_filepath:
            logger.info("Backup filepath not configured. Skipping save of in-memory state.")
            return

        async with self._lock:
            if not self.store: # If store is empty, no need to write an empty file or overwrite a good one
                logger.info("In-memory store is empty. Skipping save to file.")
                # Optionally, one might want to write an empty JSON ({}) to signify an empty state.
                # For now, we skip to avoid overwriting a potentially valid older backup with an empty one
                # if the app ran very briefly without interactions.
                return

            logger.info(f"Attempting to save in-memory state to {self.backup_filepath}")
            try:
                # Filter out expired states before saving
                current_time = time.time()
                valid_store_to_save = {
                    user_id: (data, expiry)
                    for user_id, (data, expiry) in self.store.items()
                    if expiry > current_time
                }

                if not valid_store_to_save:
                    logger.info("No valid (non-expired) states in memory. Saving an empty state to backup file.")
                    # To signify an intentional empty state, we write {}. 
                    # This helps differentiate from a corrupted file or an app that never saved.
                    await asyncio.to_thread(self._write_file_sync, self.backup_filepath, json.dumps({}))
                    return

                json_data = json.dumps(valid_store_to_save, indent=2)
                await asyncio.to_thread(self._write_file_sync, self.backup_filepath, json_data)
                logger.info(f"Successfully saved {len(valid_store_to_save)} in-memory states to {self.backup_filepath}")
            except Exception as e:
                logger.error(f"Failed to save state to file {self.backup_filepath}: {e}")
    
    def _write_file_sync(self, filepath: str, data: str):
        # Helper synchronous method for writing file content
        with open(filepath, 'w') as f:
            f.write(data)

    async def get_state(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get current conversation state for a user_id from in-memory store."""
        async with self._lock:
            if user_id in self.store:
                state_data, expiry_timestamp = self.store[user_id]
                if time.time() < expiry_timestamp:
                    logger.debug(f"In-memory state found for user {user_id}")
                    return state_data
                else:
                    logger.debug(f"In-memory state expired for user {user_id}")
                    del self.store[user_id] # Clean up expired state
            logger.debug(f"No valid in-memory state found for user {user_id}")
            return None

    async def update_state(self, user_id: str, state: Dict[str, Any], ttl: int):
        """Update conversation state for a user_id in in-memory store with TTL."""
        async with self._lock:
            expiry_timestamp = time.time() + ttl
            self.store[user_id] = (state, expiry_timestamp)
            logger.debug(f"In-memory state updated for user {user_id} with TTL: {ttl}s")

    async def clear_state(self, user_id: str):
        """Clear conversation state for a user_id from in-memory store."""
        async with self._lock:
            if user_id in self.store:
                del self.store[user_id]
                logger.debug(f"In-memory state cleared for user {user_id}")
            else:
                logger.debug(f"No in-memory state to clear for user {user_id}")

    async def Ls(self): # For debugging, to see the store. Not part of a real interface.
        async with self._lock:
            logger.info(f"Current InMemoryStore: {self.store}") 