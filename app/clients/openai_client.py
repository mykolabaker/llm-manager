import httpx
import asyncio
from typing import Dict, Any, Optional
from .base import BaseLLMClient


class OpenAIClient(BaseLLMClient):
    """OpenAI API client implementation"""

    def __init__(self, name: str = "openai", api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        super().__init__(name)
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.openai.com/v1"

        if not self.api_key:
            raise ValueError("OpenAI API key is required")

    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate text using OpenAI API

        Args:
            prompt: The input text prompt
            **kwargs: Additional parameters like temperature, max_tokens, etc.

        Returns:
            Dict containing the generated response

        Raises:
            Exception: If the API request fails
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Default parameters
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 150)
        }

        # Override with any additional parameters
        for key, value in kwargs.items():
            if key in ["temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"]:
                payload[key] = value

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                )

                if response.status_code == 200:
                    data = response.json()
                    generated_text = data["choices"][0]["message"]["content"]

                    return {
                        "response": generated_text.strip(),
                        "client": self.name,
                        "model": self.model,
                        "usage": data.get("usage", {}),
                        "simulated": False
                    }
                else:
                    raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")

        except httpx.TimeoutException:
            raise Exception("OpenAI API request timed out")
        except httpx.RequestError as e:
            raise Exception(f"OpenAI API request failed: {str(e)}")
        except Exception as e:
            raise Exception(f"OpenAI client error: {str(e)}")

    def get_client_info(self) -> Dict[str, str]:
        """Get information about the OpenAI client"""
        return {
            "name": self.name,
            "type": "openai",
            "description": "OpenAI API client using GPT models",
            "model": self.model,
            "status": "active" if self.api_key else "inactive"
        }