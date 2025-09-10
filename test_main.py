import pytest
import asyncio
import os
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from httpx import AsyncClient
import json

# Import all the modules from the app
from app.main import app
from app.clients.base import BaseLLMClient
from app.clients.mock_client import MockLLMClient
from app.clients.openai_client import OpenAIClient
from app.managers.client_manager import ClientManager
from app.config import get_settings, get_client_configs


class TestBaseLLMClient:
    """Test the abstract base class"""

    def test_base_client_cannot_be_instantiated(self):
        """Test that abstract base class cannot be instantiated directly"""
        with pytest.raises(TypeError):
            BaseLLMClient("test")


class TestMockClient:
    """Test suite for MockLLMClient"""

    @pytest.fixture
    def mock_client(self):
        return MockLLMClient()

    def test_mock_client_initialization(self, mock_client):
        """Test mock client initialization"""
        assert mock_client.name == "mock"
        assert len(mock_client.responses) > 0

    @pytest.mark.asyncio
    async def test_mock_client_generate(self, mock_client):
        """Test mock client text generation"""
        result = await mock_client.generate("Hello, world!")

        assert "response" in result
        assert "client" in result
        assert "prompt_length" in result
        assert "simulated" in result
        assert result["client"] == "mock"
        assert result["simulated"] is True
        assert len(result["response"]) > 0

    @pytest.mark.asyncio
    async def test_mock_client_short_prompt(self, mock_client):
        """Test mock client with short prompt"""
        result = await mock_client.generate("Hi")
        assert "brief mock response" in result["response"].lower()

    @pytest.mark.asyncio
    async def test_mock_client_long_prompt(self, mock_client):
        """Test mock client with long prompt"""
        long_prompt = "A" * 150
        result = await mock_client.generate(long_prompt)
        assert "extended mock response" in result["response"].lower()

    @pytest.mark.asyncio
    async def test_mock_client_hello_prompt(self, mock_client):
        """Test mock client with hello prompt"""
        result = await mock_client.generate("Hello there!")
        assert "Hello there!" in result["response"]

    def test_mock_client_info(self, mock_client):
        """Test getting mock client info"""
        info = mock_client.get_client_info()
        assert info["name"] == "mock"
        assert info["type"] == "mock"
        assert "status" in info


class TestOpenAIClient:
    """Test suite for OpenAIClient"""

    def test_openai_client_initialization_no_key(self):
        """Test OpenAI client initialization without API key"""
        with pytest.raises(ValueError, match="OpenAI API key is required"):
            OpenAIClient(api_key=None)

    def test_openai_client_initialization_with_key(self):
        """Test OpenAI client initialization with API key"""
        client = OpenAIClient(api_key="test-key")
        assert client.name == "openai"
        assert client.api_key == "test-key"
        assert client.model == "gpt-3.5-turbo"

    def test_openai_client_custom_model(self):
        """Test OpenAI client with custom model"""
        client = OpenAIClient(api_key="test-key", model="gpt-4")
        assert client.model == "gpt-4"

    @pytest.mark.asyncio
    async def test_openai_client_generate_success(self):
        """Test successful OpenAI API call"""
        client = OpenAIClient(api_key="test-key")

        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": "This is a test response from OpenAI"
                    }
                }
            ],
            "usage": {"total_tokens": 10}
        }

        with patch('httpx.AsyncClient.post') as mock_post:
            mock_resp = Mock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = mock_response
            mock_post.return_value = mock_resp

            result = await client.generate("Test prompt")

            assert result["response"] == "This is a test response from OpenAI"
            assert result["client"] == "openai"
            assert result["simulated"] is False
            assert "usage" in result

    @pytest.mark.asyncio
    async def test_openai_client_generate_error(self):
        """Test OpenAI API error handling"""
        client = OpenAIClient(api_key="test-key")

        with patch('httpx.AsyncClient.post') as mock_post:
            mock_resp = Mock()
            mock_resp.status_code = 500
            mock_resp.text = "Internal Server Error"
            mock_post.return_value = mock_resp

            with pytest.raises(Exception, match="OpenAI API error"):
                await client.generate("Test prompt")

    @pytest.mark.asyncio
    async def test_openai_client_timeout(self):
        """Test OpenAI API timeout handling"""
        client = OpenAIClient(api_key="test-key")

        with patch('httpx.AsyncClient.post', side_effect=asyncio.TimeoutError()):
            with pytest.raises(Exception, match="timed out"):
                await client.generate("Test prompt")

    def test_openai_client_info(self):
        """Test getting OpenAI client info"""
        client = OpenAIClient(api_key="test-key", model="gpt-4")
        info = client.get_client_info()

        assert info["name"] == "openai"
        assert info["type"] == "openai"
        assert info["model"] == "gpt-4"
        assert info["status"] == "active"


class TestClientManager:
    """Test suite for ClientManager"""

    @pytest.fixture
    def manager(self):
        return ClientManager()

    @pytest.fixture
    def mock_client(self):
        return MockLLMClient("test-mock")

    def test_client_manager_initialization(self, manager):
        """Test client manager initialization"""
        assert len(manager) == 0
        assert manager.list_clients() == []

    def test_register_client(self, manager, mock_client):
        """Test registering a client"""
        manager.register_client(mock_client)
        assert len(manager) == 1
        assert "test-mock" in manager
        assert manager.list_clients() == ["test-mock"]

    def test_get_existing_client(self, manager, mock_client):
        """Test getting an existing client"""
        manager.register_client(mock_client)
        retrieved = manager.get_client("test-mock")
        assert retrieved is mock_client

    def test_get_nonexistent_client(self, manager):
        """Test getting a non-existent client"""
        with pytest.raises(KeyError, match="Client 'nonexistent' not found"):
            manager.get_client("nonexistent")

    def test_create_client_mock(self, manager):
        """Test creating a mock client through factory"""
        client = manager.create_client("mock", "test-mock")
        assert isinstance(client, MockLLMClient)
        assert client.name == "test-mock"

    def test_create_client_openai(self, manager):
        """Test creating an OpenAI client through factory"""
        client = manager.create_client("openai", "test-openai", api_key="test-key")
        assert isinstance(client, OpenAIClient)
        assert client.name == "test-openai"

    def test_create_unknown_client_type(self, manager):
        """Test creating unknown client type"""
        with pytest.raises(ValueError, match="Unknown client type: unknown"):
            manager.create_client("unknown", "test")

    def test_list_client_types(self, manager):
        """Test listing available client types"""
        types = manager.list_client_types()
        assert "mock" in types
        assert "openai" in types

    def test_get_client_info(self, manager, mock_client):
        """Test getting client info"""
        manager.register_client(mock_client)
        info = manager.get_client_info("test-mock")
        assert info["name"] == "test-mock"
        assert info["type"] == "mock"

    def test_get_all_clients_info(self, manager):
        """Test getting info for all clients"""
        mock_client = MockLLMClient("mock1")
        manager.register_client(mock_client)

        all_info = manager.get_all_clients_info()
        assert len(all_info) == 1
        assert "mock1" in all_info

    @pytest.mark.asyncio
    async def test_generate_text(self, manager):
        """Test text generation through manager"""
        mock_client = MockLLMClient("test-mock")
        manager.register_client(mock_client)

        result = await manager.generate("test-mock", "Hello!")
        assert "response" in result
        assert result["client"] == "test-mock"

    def test_remove_client(self, manager, mock_client):
        """Test removing a client"""
        manager.register_client(mock_client)
        assert len(manager) == 1

        removed = manager.remove_client("test-mock")
        assert removed is True
        assert len(manager) == 0

    def test_remove_nonexistent_client(self, manager):
        """Test removing a non-existent client"""
        removed = manager.remove_client("nonexistent")
        assert removed is False

    def test_register_new_client_type(self, manager):
        """Test registering a new client type"""

        class CustomClient(BaseLLMClient):
            async def generate(self, prompt, **kwargs):
                return {"response": "custom", "client": self.name}

            def get_client_info(self):
                return {"name": self.name, "type": "custom"}

        manager.register_client_type("custom", CustomClient)
        assert "custom" in manager.list_client_types()

        client = manager.create_client("custom", "test-custom")
        assert isinstance(client, CustomClient)


class TestConfiguration:
    """Test suite for configuration management"""

    def test_settings_creation(self):
        """Test settings creation"""
        settings = get_settings()
        assert hasattr(settings, 'openai_api_key')
        assert hasattr(settings, 'app_name')
        assert hasattr(settings, 'debug')

    def test_client_configs_mock_always_available(self):
        """Test that mock client is always available"""
        configs = get_client_configs()
        assert "mock" in configs
        assert configs["mock"]["enabled"] is True

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_client_configs_with_openai_key(self):
        """Test client configs when OpenAI key is available"""
        configs = get_client_configs()
        assert "openai" in configs
        assert configs["openai"]["enabled"] is True


class TestFastAPIEndpoints:
    """Test suite for FastAPI endpoints"""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "clients_count" in data

    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_list_clients_endpoint(self, client):
        """Test clients listing endpoint"""
        response = client.get("/clients")
        assert response.status_code == 200
        data = response.json()
        assert "clients" in data
        assert "total" in data

    def test_generate_endpoint_success(self, client):
        """Test successful generation request"""
        test_data = {
            "prompt": "Hello, world!",
            "client_name": "mock"
        }
        response = client.post("/generate", json=test_data)
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert len(data) == 1  # Should only return response field

    def test_generate_endpoint_invalid_client(self, client):
        """Test generation with invalid client"""
        test_data = {
            "prompt": "Hello, world!",
            "client_name": "nonexistent"
        }
        response = client.post("/generate", json=test_data)
        assert response.status_code == 404

    def test_generate_endpoint_missing_prompt(self, client):
        """Test generation with missing prompt"""
        test_data = {
            "client_name": "mock"
        }
        response = client.post("/generate", json=test_data)
        assert response.status_code == 422

    def test_generate_endpoint_empty_prompt(self, client):
        """Test generation with empty prompt"""
        test_data = {
            "prompt": "",
            "client_name": "mock"
        }
        response = client.post("/generate", json=test_data)
        assert response.status_code == 422

    def test_generate_detailed_endpoint(self, client):
        """Test detailed generation endpoint"""
        test_data = {
            "prompt": "Hello, world!",
            "client_name": "mock"
        }
        response = client.post("/generate/detailed", json=test_data)
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "client" in data

    def test_get_specific_client_info(self, client):
        """Test getting specific client info"""
        response = client.get("/clients/mock")
        assert response.status_code in [200, 404]  # Depends on whether mock is registered

    def test_get_nonexistent_client_info(self, client):
        """Test getting info for nonexistent client"""
        response = client.get("/clients/nonexistent")
        assert response.status_code == 404

    def test_invalid_json_request(self, client):
        """Test invalid JSON request"""
        response = client.post("/generate", data="invalid json")
        assert response.status_code == 422

    def test_validation_errors_long_prompt(self, client):
        """Test validation error with overly long prompt"""
        test_data = {
            "prompt": "A" * 15000,  # Exceeds max length
            "client_name": "mock"
        }
        response = client.post("/generate", json=test_data)
        assert response.status_code == 422

    def test_validation_errors_invalid_temperature(self, client):
        """Test validation error with invalid temperature"""
        test_data = {
            "prompt": "Hello",
            "client_name": "mock",
            "temperature": 5.0  # Exceeds max value
        }
        response = client.post("/generate", json=test_data)
        assert response.status_code == 422

    def test_validation_errors_invalid_max_tokens(self, client):
        """Test validation error with invalid max_tokens"""
        test_data = {
            "prompt": "Hello",
            "client_name": "mock",
            "max_tokens": 0  # Below minimum
        }
        response = client.post("/generate", json=test_data)
        assert response.status_code == 422


class TestIntegration:
    """Integration tests"""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_full_workflow_mock_client(self, client):
        """Test complete workflow with mock client"""
        # Test health check
        health_response = client.get("/health")
        assert health_response.status_code == 200

        # List clients
        clients_response = client.get("/clients")
        assert clients_response.status_code == 200
        clients_data = clients_response.json()

        # Generate text if mock client is available
        if clients_data["total"] > 0:
            generate_data = {
                "prompt": "Write a haiku about programming",
                "client_name": "mock"
            }
            generate_response = client.post("/generate", json=generate_data)
            # Should succeed if mock client is registered
            assert generate_response.status_code in [200, 404]

    def test_concurrent_requests(self, client):
        """Test concurrent requests handling"""
        import concurrent.futures
        import threading

        def make_request():
            test_data = {
                "prompt": f"Request from thread {threading.current_thread().ident}",
                "client_name": "mock"
            }
            response = client.post("/generate", json=test_data)
            return response.status_code

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # All requests should return valid HTTP status codes
        for status_code in results:
            assert status_code in [200, 404, 422]  # Valid status codes


# Fixtures for testing
@pytest.fixture
def mock_openai_response():
    """Fixture for mock OpenAI response"""
    return {
        "choices": [
            {
                "message": {
                    "content": "This is a test response from OpenAI API"
                }
            }
        ],
        "usage": {"total_tokens": 10}
    }


@pytest.fixture
def sample_request_data():
    """Fixture for sample request data"""
    return {
        "prompt": "Write a short story about a robot learning to paint",
        "client_name": "mock"
    }


@pytest.fixture
def invalid_request_data():
    """Fixture for invalid request data"""
    return {
        "prompt": "Test prompt",
        "client_name": "invalid_client"
    }


# Test runner configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
