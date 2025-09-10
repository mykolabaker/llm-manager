from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from app.config import settings, get_client_configs
from app.managers.client_manager import ClientManager
from app.models.schemas import (
    GenerateRequest,
    GenerateResponse,
    ClientListResponse,
    ClientInfo,
    ErrorResponse,
    HealthResponse
)

# Set up logging
logging.basicConfig(level=logging.INFO if not settings.debug else logging.DEBUG)
logger = logging.getLogger(__name__)

# Global client manager instance
client_manager = ClientManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting LLM Client Manager service...")

    # Initialize clients based on configuration
    client_configs = get_client_configs()

    for client_name, config in client_configs.items():
        if config.get("enabled", False):
            try:
                client = client_manager.create_client(
                    client_type=config["type"],
                    name=config["name"],
                    **{k: v for k, v in config.items() if k not in ["enabled", "type", "name"]}
                )
                client_manager.register_client(client)
                logger.info(f"Registered client: {client_name}")
            except Exception as e:
                logger.warning(f"Failed to register client {client_name}: {e}")

    logger.info(f"Service started with {len(client_manager)} clients")

    yield

    # Shutdown
    logger.info("Shutting down LLM Client Manager service...")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="A FastAPI service for managing multiple LLM clients",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_client_manager() -> ClientManager:
    """Dependency to get the client manager"""
    return client_manager


@app.exception_handler(KeyError)
async def key_error_handler(request, exc):
    """Handle KeyError exceptions (client not found)"""
    return JSONResponse(
        status_code=404,
        content=ErrorResponse(
            error="Client not found",
            detail=str(exc)
        ).dict()
    )


@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle ValueError exceptions"""
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error="Invalid request",
            detail=str(exc)
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail="An unexpected error occurred"
        ).dict()
    )


@app.get("/", response_model=HealthResponse)
async def root(manager: ClientManager = Depends(get_client_manager)):
    """Root endpoint with service health information"""
    return HealthResponse(
        status="healthy",
        clients_count=len(manager),
        available_clients=manager.list_clients()
    )


@app.get("/health", response_model=HealthResponse)
async def health_check(manager: ClientManager = Depends(get_client_manager)):
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        clients_count=len(manager),
        available_clients=manager.list_clients()
    )


@app.get("/clients", response_model=ClientListResponse)
async def list_clients(manager: ClientManager = Depends(get_client_manager)):
    """List all available clients"""
    clients_info = manager.get_all_clients_info()
    clients = {
        name: ClientInfo(**info)
        for name, info in clients_info.items()
    }

    return ClientListResponse(
        clients=clients,
        total=len(clients)
    )


@app.get("/clients/{client_name}", response_model=ClientInfo)
async def get_client_info(
        client_name: str,
        manager: ClientManager = Depends(get_client_manager)
):
    """Get information about a specific client"""
    try:
        info = manager.get_client_info(client_name)
        return ClientInfo(**info)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Client '{client_name}' not found"
        )


@app.post("/generate", response_model=Dict[str, str])
async def generate_text(
        request: GenerateRequest,
        manager: ClientManager = Depends(get_client_manager)
):
    """
    Generate text using the specified LLM client

    This endpoint accepts a prompt and client name, then returns
    a JSON response with a single field containing the generated text.
    """
    try:
        # Generate text using the specified client
        result = await manager.generate(
            client_name=request.client_name,
            prompt=request.prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        # Return only the response field as required
        return {"response": result["response"]}

    except KeyError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Client '{request.client_name}' not found. Available clients: {manager.list_clients()}"
        )
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Text generation failed: {str(e)}"
        )


@app.post("/generate/detailed", response_model=GenerateResponse)
async def generate_text_detailed(
        request: GenerateRequest,
        manager: ClientManager = Depends(get_client_manager)
):
    """
    Generate text with detailed response information

    This endpoint returns full details about the generation process,
    including client info, model used, and usage statistics.
    """
    try:
        result = await manager.generate(
            client_name=request.client_name,
            prompt=request.prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        return GenerateResponse(**result)

    except KeyError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Client '{request.client_name}' not found. Available clients: {manager.list_clients()}"
        )
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Text generation failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )