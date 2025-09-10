from typing import Dict, List, Optional, Type
from ..clients.base import BaseLLMClient
from ..clients.mock_client import MockLLMClient
from ..clients.openai_client import OpenAIClient


class ClientManager:
    """
    Client manager implementing Factory and Registry patterns
    Manages different LLM clients and provides a unified interface
    """

    def __init__(self):
        self._clients: Dict[str, BaseLLMClient] = {}
        self._client_types: Dict[str, Type[BaseLLMClient]] = {
            "mock": MockLLMClient,
            "openai": OpenAIClient
        }

    def register_client(self, client: BaseLLMClient) -> None:
        """
        Register a new client instance

        Args:
            client: An instance of BaseLLMClient
        """
        self._clients[client.name] = client

    def register_client_type(self, name: str, client_class: Type[BaseLLMClient]) -> None:
        """
        Register a new client type for factory creation

        Args:
            name: The name identifier for the client type
            client_class: The client class to register
        """
        self._client_types[name] = client_class

    def create_client(self, client_type: str, name: str, **kwargs) -> BaseLLMClient:
        """
        Factory method to create a new client instance

        Args:
            client_type: The type of client to create
            name: The name for the client instance
            **kwargs: Additional parameters for client initialization

        Returns:
            A new client instance

        Raises:
            ValueError: If client_type is not registered
        """
        if client_type not in self._client_types:
            raise ValueError(f"Unknown client type: {client_type}")

        client_class = self._client_types[client_type]
        return client_class(name=name, **kwargs)

    def get_client(self, name: str) -> BaseLLMClient:
        """
        Get a client by name

        Args:
            name: The name of the client

        Returns:
            The requested client instance

        Raises:
            KeyError: If client with given name is not found
        """
        if name not in self._clients:
            raise KeyError(f"Client '{name}' not found. Available clients: {list(self._clients.keys())}")

        return self._clients[name]

    def list_clients(self) -> List[str]:
        """
        Get list of available client names

        Returns:
            List of registered client names
        """
        return list(self._clients.keys())

    def list_client_types(self) -> List[str]:
        """
        Get list of available client types

        Returns:
            List of registered client types
        """
        return list(self._client_types.keys())

    def get_client_info(self, name: str) -> Dict[str, str]:
        """
        Get information about a specific client

        Args:
            name: The name of the client

        Returns:
            Dict containing client information
        """
        client = self.get_client(name)
        return client.get_client_info()

    def get_all_clients_info(self) -> Dict[str, Dict[str, str]]:
        """
        Get information about all registered clients

        Returns:
            Dict mapping client names to their information
        """
        return {name: client.get_client_info() for name, client in self._clients.items()}

    async def generate(self, client_name: str, prompt: str, **kwargs) -> Dict[str, any]:
        """
        Generate text using a specific client

        Args:
            client_name: The name of the client to use
            prompt: The input text prompt
            **kwargs: Additional parameters for generation

        Returns:
            Dict containing the generated response

        Raises:
            KeyError: If client is not found
        """
        client = self.get_client(client_name)
        return await client.generate(prompt, **kwargs)

    def remove_client(self, name: str) -> bool:
        """
        Remove a client from the manager

        Args:
            name: The name of the client to remove

        Returns:
            True if client was removed, False if client was not found
        """
        if name in self._clients:
            del self._clients[name]
            return True
        return False

    def __len__(self) -> int:
        """Return the number of registered clients"""
        return len(self._clients)

    def __contains__(self, name: str) -> bool:
        """Check if a client is registered"""
        return name in self._clients

    def __str__(self) -> str:
        return f"ClientManager(clients={list(self._clients.keys())})"

    def __repr__(self) -> str:
        return self.__str__()