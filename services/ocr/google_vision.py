from google.cloud import vision # type: ignore
from .base import OCRService
from monitoring.langfuse_client import LangfuseMonitor # Import LangfuseMonitor
from typing import Dict, Any, Optional
import logging
from config.settings import settings  # Add this import
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type # Import tenacity
import asyncio # For running sync retryable in thread
from google.api_core import exceptions as google_exceptions

logger = logging.getLogger(__name__)


# Update the retryable exceptions
RETRYABLE_GOOGLE_EXCEPTIONS = (
    google_exceptions.ServiceUnavailable,
    google_exceptions.DeadlineExceeded,
    google_exceptions.InternalServerError
)
class GoogleVisionOCR(OCRService):
    def __init__(self, langfuse_monitor: Optional[LangfuseMonitor] = None): # Accept LangfuseMonitor
        self.langfuse_monitor = langfuse_monitor
        # Get the actual client if monitor is available, for decorator usage
        self.lf_client = langfuse_monitor.get_client() if langfuse_monitor else None
        try:
            # Use credentials from settings
            if settings.GOOGLE_APPLICATION_CREDENTIALS:
                self.client = vision.ImageAnnotatorClient.from_service_account_file(
                    settings.GOOGLE_APPLICATION_CREDENTIALS
                )
            else:
                self.client = vision.ImageAnnotatorClient()
            logger.info("Google Vision API client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Google Vision API client: {e}", exc_info=True)
            raise
        
    @property # Make it a property so the decorator can be applied easily
    def observe(self):
        # Helper to get the decorator if langfuse_monitor is available
        if self.langfuse_monitor:
            return self.langfuse_monitor.observe_function
        # Return a dummy decorator that does nothing if langfuse is not configured
        def dummy_decorator(metadata=None):
            def decorator(func):
                return func
            return decorator
        return dummy_decorator

    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=1, max=5),
           retry=retry_if_exception_type(RETRYABLE_GOOGLE_EXCEPTIONS))
    def _text_detection_with_retry(self, image: vision.Image) -> vision.AnnotateImageResponse:
        logger.debug("Calling Google Vision API: text_detection")
        return self.client.text_detection(image=image)

    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=1, max=5),
           retry=retry_if_exception_type(RETRYABLE_GOOGLE_EXCEPTIONS))
    def _document_text_detection_with_retry(self, image: vision.Image) -> vision.AnnotateImageResponse:
        logger.debug("Calling Google Vision API: document_text_detection")
        return self.client.document_text_detection(image=image)

    async def extract_text(self, image_bytes: bytes) -> str:
        """Extracts all text from an image using Google Vision API's text_detection."""
        # Apply decorator dynamically
        @self.observe(metadata={"category": "ocr_processing", "ocr_method": "text_detection"})
        async def _decorated_extract_text():
            if not self.client:
                logger.error("GoogleVisionOCR client not initialized. Cannot extract text.")
                raise ConnectionRefusedError("OCR client not initialized.")
            try:
                image = vision.Image(content=image_bytes)
                # Run the synchronous, retryable method in a thread
                response = await asyncio.to_thread(self._text_detection_with_retry, image)
                
                if response.error.message:
                    logger.error(f"Google Vision API error during text_detection: {response.error.message}")
                    raise Exception(f"Google Vision API Error: {response.error.message}")

                if response.text_annotations:
                    # The first annotation is usually the full text
                    full_text = response.text_annotations[0].description
                    logger.info(f"Text extracted successfully. Length: {len(full_text)}")
                    return full_text
                else:
                    logger.info("No text found in image by text_detection.")
                    return ""
            except RETRYABLE_GOOGLE_EXCEPTIONS as e: # Catch retryable exceptions if all retries fail
                logger.error(f"Google Vision text_detection failed after retries: {e}", exc_info=True)
                raise # Re-raise for decorator and caller to handle
            except Exception as e:
                logger.error(f"Exception during Google Vision text_detection: {e}", exc_info=True)
                raise
        return await _decorated_extract_text()
        
    async def extract_structured_data(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Extracts structured data from a receipt image using Google Vision API's document_text_detection.
        This is a placeholder for the more complex parsing logic mentioned in PRD (_parse_receipt_structure).
        For now, it will return the raw response or a simplified structure.
        """
        @self.observe(metadata={"category": "ocr_processing", "ocr_method": "document_text_detection"})
        async def _decorated_extract_structured_data():
            if not self.client:
                logger.error("GoogleVisionOCR client not initialized. Cannot extract structured data.")
                raise ConnectionRefusedError("OCR client not initialized.") 
            try:
                image = vision.Image(content=image_bytes)
                # Run the synchronous, retryable method in a thread
                response = await asyncio.to_thread(self._document_text_detection_with_retry, image)

                if response.error.message:
                    logger.error(f"Google Vision API error during document_text_detection: {response.error.message}")
                    raise Exception(f"Google Vision API Error: {response.error.message}")

                # The PRD mentions: "Process blocks, paragraphs, and words. Extract line items, amounts, etc.
                # return self._parse_receipt_structure(response)"
                # This _parse_receipt_structure would be a complex method.
                # For now, let's return a simplified version or a marker that it needs implementation.
                
                # For demonstration, we can try to return the full text annotation similar to extract_text
                # or a more structured representation if easily accessible.
                if response.full_text_annotation:
                    logger.info("Structured data (full_text_annotation) extracted using document_text_detection.")
                    # This is not the final structured data (line items, total etc.) but the raw material.
                    # The actual parsing logic will be complex and is TBD.
                    return {
                        "full_text": response.full_text_annotation.text,
                        "pages": len(response.full_text_annotation.pages),
                        "status": "Full text extracted, further parsing needed."
                    }
                else:
                    logger.info("No structured text found in image by document_text_detection.")
                    return {"full_text": "", "pages": 0, "status": "No text found."}

            except RETRYABLE_GOOGLE_EXCEPTIONS as e: # Catch retryable exceptions if all retries fail
                logger.error(f"Google Vision document_text_detection failed after retries: {e}", exc_info=True)
                raise
            except Exception as e:
                logger.error(f"Exception during Google Vision document_text_detection: {e}", exc_info=True)
                raise 
        return await _decorated_extract_structured_data()

    def _parse_receipt_structure(self, response: vision.AnnotateImageResponse) -> Dict[str, Any]:
        """
        Parses the complex response from document_text_detection into a structured receipt format.
        This is a placeholder for the actual implementation which will involve iterating through
        blocks, paragraphs, words, and symbols to identify receipt elements.
        """
        # Placeholder for complex parsing logic
        # This would involve looking at geometry (bounding boxes) of words, 
        # pattern matching for prices, item descriptions, totals, tax, etc.
        logger.warning("_parse_receipt_structure is not fully implemented. Returning raw text for now.")
        if response and response.full_text_annotation:
            return {"raw_text": response.full_text_annotation.text, "parsed_items": []} # Example structure
        return {"error": "No data to parse"} 