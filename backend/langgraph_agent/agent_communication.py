"""
Agent communication module for handling interactions with external API agents.
Extracted from test.py to be reusable for agent-to-agent communication.
"""
import os
import requests
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()


class ExternalAPIAgent:
    """
    Handles communication with the external negotiation bot API.
    Manages conversation initialization and message exchange.
    """

    def __init__(
        self,
        api_base: Optional[str] = None,
        team_id: Optional[int] = None,
        vendor_id: Optional[int] = None,
    ):
        """
        Initialize the external API agent.

        Args:
            api_base: Base URL for the external API. Defaults to env var EXTERNAL_API_BASE.
            team_id: Team ID for the API. Defaults to env var EXTERNAL_API_TEAM_ID.
            vendor_id: Vendor ID for the API. Defaults to env var EXTERNAL_API_VENDOR_ID.
        """
        self.api_base = api_base or os.getenv(
            "EXTERNAL_API_BASE",
            "https://negbot-backend-ajdxh9axb0ddb0e9.westeurope-01.azurewebsites.net/api",
        )
        self.team_id = team_id or int(os.getenv("EXTERNAL_API_TEAM_ID", "851996"))
        self.vendor_id = vendor_id or int(os.getenv("EXTERNAL_API_VENDOR_ID", "44"))
        self.conversation_id: Optional[str] = None

    async def initialize_conversation(self) -> str:
        """
        Initialize a new conversation with the external API.

        Returns:
            The conversation ID.

        Raises:
            Exception: If conversation creation fails.
        """
        try:
            response = requests.post(
                f"{self.api_base}/conversations/",
                params={"team_id": self.team_id},
                json={"vendor_id": self.vendor_id, "title": "Agent-to-Agent Session"},
            )

            if response.status_code not in [200, 201]:
                raise Exception(
                    f"Failed to create conversation: {response.status_code} - {response.text}"
                )

            self.conversation_id = response.json()["id"]
            return self.conversation_id
        except Exception as e:
            raise Exception(f"Error initializing conversation: {str(e)}")

    async def send_message(self, message: str) -> str:
        """
        Send a message to the external API and get the response.

        Args:
            message: The message to send.

        Returns:
            The response content from the external API.

        Raises:
            Exception: If the conversation is not initialized or if sending fails.
        """
        if not self.conversation_id:
            raise Exception("Conversation not initialized. Call initialize_conversation() first.")

        try:
            response = requests.post(
                f"{self.api_base}/messages/{self.conversation_id}",
                data={"content": message},
            )

            if response.status_code in [200, 201]:
                ai_response = response.json()
                return ai_response.get("content", "No response")
            elif response.status_code == 429:
                raise Exception("Rate limit reached. Please wait a moment and try again.")
            else:
                raise Exception(f"Error sending message: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Network error sending message: {str(e)}")
        except Exception as e:
            raise Exception(f"Error sending message: {str(e)}")

    def is_initialized(self) -> bool:
        """Check if the conversation has been initialized."""
        return self.conversation_id is not None

