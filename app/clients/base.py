from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseLLMClient(ABC):
    """Abstract base class for all LLM clients"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate text using the LLM API

        Args:
            prompt: The input text prompt
            **kwargs: Additional parameters specific to the client

        Returns:
            Dict containing the generated response

        Raises:
            Exception: If the API request fails
        """
        pass

    @abstractmethod
    def get_client_info(self) -> Dict[str, str]:
        """
        Get information about the client

        Returns:
            Dict containing client information
        """
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __repr__(self) -> str:
        return self.__str__()