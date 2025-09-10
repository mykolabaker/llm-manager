# LLM Client Manager

A FastAPI service that implements a client manager for accessing various LLM APIs using the Factory and Strategy design
patterns.

## Features

- **Multiple LLM Clients**: Support for OpenAI and mock clients (easily extensible)
- **Design Patterns**: Implementation of Factory and Strategy patterns for clean architecture
- **SOLID Principles**: Easy addition of new clients without modifying existing code
- **Comprehensive Testing**: Full test suite with pytest
- **Docker Support**: Containerized deployment
- **Configuration Management**: Environment-based configuration
- **Error Handling**: Robust error handling and validation

## Architecture

The project follows clean architecture principles with:

- **Abstract Base Client**: `BaseLLMClient` defines the interface for all LLM clients
- **Concrete Clients**: `MockLLMClient`, `OpenAIClient` implement specific providers
- **Client Manager**: `ClientManager` implements Factory and Registry patterns
- **Request/Response Models**: Pydantic models for data validation
- **FastAPI Integration**: RESTful API with automatic documentation

## Project Structure

```
llm-manager/
├── app/
│   ├── clients/
│   │   ├── base.py          # Abstract base client
│   │   ├── mock_client.py   # Mock client for testing
│   │   └── openai_client.py # OpenAI API client
│   ├── managers/
│   │   └── client_manager.py # Client factory and registry
│   ├── models/
│   │   └── schemas.py       # Pydantic models
│   ├── config.py           # Configuration management
│   └── main.py             # FastAPI application
├── tests/
│   └── test_main.py        # Comprehensive test suite
├── .env                    # Environment variables
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker configuration
└── README.md              # This file
```

## Environment Variables

Create a `.env` file in the project root:

```bash
# API Keys (at least one is recommended)
OPENAI_API_KEY=your_openai_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_key_here

# OpenAI Configuration
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_TEMPERATURE=0.7
OPENAI_MAX_TOKENS=150

# Application Configuration
APP_NAME=LLM Client Manager
APP_VERSION=1.0.0
DEBUG=false

# Server Configuration
HOST=0.0.0.0
PORT=8000

# Request Configuration
REQUEST_TIMEOUT=30
MAX_PROMPT_LENGTH=10000

# CORS Configuration (comma-separated)
CORS_ORIGINS=*
```

## Installation

### Local Development

1. **Clone the repository**:

```bash
git clone <repository-url>
cd llm-manager
```

2. **Create virtual environment**:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

4. **Set up environment**:

```bash
cp .env.example .env
# Edit .env with your API keys
```

5. **Run the application**:

```bash
uvicorn app.main:app --reload
```

### Docker Deployment

1. **Build the image**:

```bash
docker build -t llm-manager .
```

2. **Run the container**:

```bash
docker run -p 8000:8000 --env-file .env llm-manager
```

## Usage

### API Endpoints

The service provides the following endpoints:

#### Health Check

- **GET** `/health` - Service health status
- **GET** `/` - Root endpoint with health info

#### Client Management

- **GET** `/clients` - List all available clients
- **GET** `/clients/{client_name}` - Get specific client info

#### Text Generation

- **POST** `/generate` - Generate text (returns single response field)
- **POST** `/generate/detailed` - Generate text with detailed response

### Example Requests

#### Simple Text Generation

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a haiku about programming",
    "client_name": "mock"
  }'
```

Response:

```json
{
    "response": "Mock response about programming haiku..."
}
```

#### Detailed Text Generation

```bash
curl -X POST "http://localhost:8000/generate/detailed" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum computing",
    "client_name": "openai",
    "temperature": 0.7,
    "max_tokens": 200
  }'
```

Response:

```json
{
    "response": "Quantum computing is...",
    "client": "openai",
    "model": "gpt-3.5-turbo",
    "usage": {"total_tokens": 150},
    "simulated": false
}
```

#### List Available Clients

```bash
curl -X GET "http://localhost:8000/clients"
```

Response:

```json
{
    "clients": {
        "mock": {
            "name": "mock",
            "type": "mock",
            "description": "Mock LLM client for testing and development",
            "status": "active"
        },
        "openai": {
            "name": "openai",
            "type": "openai",
            "description": "OpenAI API client using GPT models",
            "model": "gpt-3.5-turbo",
            "status": "active"
        }
    },
    "total": 2
}
```

## Testing

The project includes comprehensive tests covering all components:

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test file
pytest tests/test_main.py

# Run with verbose output
pytest -v

# Run specific test class
pytest tests/test_main.py::TestMockClient -v
```

### Test Categories

1. **Unit Tests**: Individual component testing
    - `TestMockClient`: Mock client functionality
    - `TestOpenAIClient`: OpenAI client with mocked API calls
    - `TestClientManager`: Factory and registry pattern testing
    - `TestConfiguration`: Configuration management

2. **Integration Tests**: End-to-end API testing
    - `TestFastAPIEndpoints`: All REST endpoints
    - `TestIntegration`: Full workflow testing
    - `TestErrorHandling`: Error scenarios

3. **Concurrent Testing**: Multi-threaded request handling

## Adding New Clients

The architecture supports easy addition of new LLM clients:

### 1. Create New Client Class

```python
# app/clients/new_provider_client.py
from .base import BaseLLMClient
from typing import Dict, Any


class NewProviderClient(BaseLLMClient):
    def __init__(self, name: str = "new_provider", api_key: str = None):
        super().__init__(name)
        self.api_key = api_key
        # Initialize provider-specific settings

    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        # Implement provider-specific API call
        # Return standardized response format
        return {
            "response": "Generated text...",
            "client": self.name,
            "simulated": False
        }

    def get_client_info(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "type": "new_provider",
            "description": "New Provider LLM client",
            "status": "active"
        }
```

### 2. Register in Client Manager

```python
# In client_manager.py constructor
self._client_types: Dict[str, Type[BaseLLMClient]] = {
    "mock": MockLLMClient,
    "openai": OpenAIClient,
    "new_provider": NewProviderClient  # Add new client type
}
```

### 3. Add Configuration Support

```python
# In config.py get_client_configs()
if settings.new_provider_api_key:
    configs["new_provider"] = {
        "enabled": True,
        "name": "new_provider",
        "type": "new_provider",
        "api_key": settings.new_provider_api_key
    }
```

### 4. Write Tests

Create comprehensive tests following the existing patterns in `test_main.py`.

## API Documentation

When the service is running, visit:

- **Interactive API docs**: http://localhost:8000/docs
- **Alternative docs**: http://localhost:8000/redoc

## Design Patterns Used

### 1. Factory Pattern

The `ClientManager` acts as a factory for creating different types of LLM clients:

```python
client = manager.create_client("openai", "my-openai", api_key="...")
```

### 2. Strategy Pattern

Different LLM clients implement the same `BaseLLMClient` interface, allowing interchangeable use:

```python
# Can use any registered client
await manager.generate("mock", "Hello world")
await manager.generate("openai", "Hello world")
```

### 3. Registry Pattern

The `ClientManager` maintains a registry of active clients:

```python
manager.register_client(client)
manager.list_clients()  # ["mock", "openai"]
```

## Error Handling

The service includes comprehensive error handling:

- **404**: Client not found
- **422**: Validation errors (invalid prompts, parameters)
- **500**: Internal server errors (API failures, timeouts)
- **Timeout handling**: For external API calls
- **Validation**: Input validation with Pydantic models

## Performance Considerations

- **Async/await**: Non-blocking API calls
- **Connection pooling**: HTTP client reuse
- **Timeout management**: Configurable request timeouts
- **Error recovery**: Graceful handling of API failures

## Security Notes

- **Environment variables**: Keep API keys in `.env` file
- **Input validation**: All inputs validated with Pydantic
- **Error responses**: No sensitive information in error messages
- **CORS configuration**: Configurable allowed origins

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License.