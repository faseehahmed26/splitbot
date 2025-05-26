from .base import MessagingInterface
from .models import Message
from config.settings import settings
import httpx  # For making HTTP requests, will be added to requirements.txt
import logging

logger = logging.getLogger(__name__)

class WhatsAppInterface(MessagingInterface):
    def __init__(self, token: str = settings.WHATSAPP_API_TOKEN, verify_token: str = settings.WHATSAPP_WEBHOOK_VERIFY):
        self.token = token
        self.verify_token = verify_token
        # The PRD test uses token="test_token", but the main app init uses settings.WHATSAPP_API_TOKEN.
        # For consistency and to load from settings, I'll use settings here.
        # The actual WhatsApp API URL would depend on the provider (e.g., Meta Graph API).
        # This is a placeholder.
        self.api_url = "https_YOUR_WHATSAPP_PROVIDER_API_URL/v1/messages" 

    async def send_message(self, user_id: str, message: Message):
        """
        Sends a message to a user via WhatsApp.
        user_id: The recipient's WhatsApp ID (phone number).
        message: A Message object containing the text to send.
        """
        payload = {
            "messaging_product": "whatsapp",
            "to": user_id,
            "type": "text",
            "text": {
                "body": message.text
            }
        }
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(self.api_url, json=payload, headers=headers)
                response.raise_for_status()  # Raises an exception for 4XX/5XX responses
                logger.info(f"Message sent to {user_id}: {response.json()}")
            except httpx.HTTPStatusError as e:
                logger.error(f"Error sending WhatsApp message to {user_id}: {e.response.status_code} - {e.response.text}")
            except httpx.RequestError as e:
                logger.error(f"Request error sending WhatsApp message to {user_id}: {e}")


    async def send_image(self, user_id: str, image: bytes):
        """
        Sends an image to a user via WhatsApp.
        This is a placeholder and would require using the WhatsApp API for media uploads.
        """
        # Implementation for sending an image would involve:
        # 1. Uploading the image to WhatsApp servers to get a media ID.
        # 2. Sending a message with that media ID.
        logger.info(f"Placeholder: Sending image to {user_id} via WhatsApp. Length: {len(image)} bytes")
        # Example text message indicating an image was supposed to be sent
        await self.send_message(user_id, Message(text="[Image received - actual image sending not yet implemented]"))

    async def handle_incoming(self, data: dict):
        """
        Handles incoming webhook data from WhatsApp.
        This method will parse the data and trigger appropriate actions.
        """
        logger.info(f"Received WhatsApp webhook data: {data}")
        # Basic parsing (highly dependent on actual WhatsApp API payload structure)
        # This is a simplified example. Real payload is more complex.
        # See: https://developers.facebook.com/docs/whatsapp/cloud-api/webhooks/payload-examples#text-messages
        
        try:
            if data.get("object") == "whatsapp_business_account":
                for entry in data.get("entry", []):
                    for change in entry.get("changes", []):
                        if change.get("field") == "messages":
                            message_object = change.get("value", {}).get("messages", [{}])[0]
                            user_id = message_object.get("from")
                            
                            if message_object.get("type") == "text":
                                text_body = message_object.get("text", {}).get("body")
                                logger.info(f"Incoming text from {user_id}: {text_body}")
                                # Here you would route to core.processor
                                # For now, let's just echo back
                                if text_body:
                                     await self.send_message(user_id, Message(text=f"You said: {text_body}"))

                            elif message_object.get("type") == "image":
                                image_id = message_object.get("image", {}).get("id")
                                logger.info(f"Incoming image from {user_id}, media ID: {image_id}")
                                # Here you would:
                                # 1. Fetch the image using the image_id and self.token
                                # 2. Pass the image_bytes to core.processor.process_image
                                # For now, just acknowledge
                                await self.send_message(user_id, Message(text="I received an image! Processing not yet implemented."))
                            
                            # Add handling for other message types (voice, location, etc.)
                            else:
                                logger.warning(f"Received unhandled message type from {user_id}: {message_object.get('type')}")
                                await self.send_message(user_id, Message(text=f"I received a {message_object.get('type')}, but I can't handle that yet."))
            else:
                logger.warning(f"Received non-WhatsApp webhook data or malformed data: {data}")

        except Exception as e:
            logger.error(f"Error processing incoming WhatsApp message: {e}", exc_info=True)
            # Potentially send an error message back to the user if possible and appropriate
            # For example, if user_id is available:
            # await self.send_message(user_id, Message(text="Sorry, I encountered an error processing your message."))

    def verify_webhook(self, params: dict) -> tuple[str, int] | tuple[bool, int]:
        """
        Verifies the webhook subscription with WhatsApp.
        params: A dictionary containing the query parameters from the webhook verification request.
        Expected params: 'hub.mode', 'hub.challenge', 'hub.verify_token'
        """
        mode = params.get("hub.mode")
        challenge = params.get("hub.challenge")
        token = params.get("hub.verify_token")

        if mode == "subscribe" and token == self.verify_token:
            logger.info(f"WhatsApp webhook verified successfully. Challenge: {challenge}")
            return challenge, 200  # Return the challenge string and HTTP 200 OK
        else:
            logger.warning(f"WhatsApp webhook verification failed. Mode: {mode}, Token: {token}, Expected Token: {self.verify_token}")
            return False, 403 # Return False and HTTP 403 Forbidden 