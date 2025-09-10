from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional


class GenerateRequest(BaseModel):
    """Request model for text generation"""

    prompt: str = Field(..., min_length=1, max_length=10000, description="The input text prompt")
    client_name: str = Field(..., min_length=1, max_length=50, description="The name of the LLM client to use")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(150, ge=1, le=4000, description="Maximum number of tokens to generate")

    @validator('prompt')
    def validate_prompt(cls, v):
        """Validate that prompt is not just whitespace"""
        if not v.strip():
            raise ValueError('Prompt cannot be empty or just whitespace')
        return v.strip()

    @validator('client_name')
    def validate_client_name(cls, v):
        """Validate client name format"""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Client name can only contain alphanumeric characters, hyphens, and underscores')
        return v.lower()


class GenerateResponse(BaseModel):
    """Response model for text generation"""

    response: str = Field(..., description="The generated text response")
    client: Optional[str] = Field(None, description="The client that generated the response")
    model: Optional[str] = Field(None, description="The model used for generation")
    usage: Optional[Dict[str, Any]] = Field(None, description="Usage statistics from the API")
    simulated: Optional[bool] = Field(None, description="Whether this is a simulated response")


class ClientInfo(BaseModel):
    """Model for client information"""

    name: str = Field(..., description="Client name")
    type: str = Field(..., description="Client type")
    description: str = Field(..., description="Client description")
    status: str = Field(..., description="Client status")
    model: Optional[str] = Field(None, description="Model name if applicable")
    version: Optional[str] = Field(None, description="Client version if applicable")


class ErrorResponse(BaseModel):
    """Model for error responses"""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    client: Optional[str] = Field(None, description="Client that caused the error")


class ClientListResponse(BaseModel):
    """Response model for listing clients"""

    clients: Dict[str, ClientInfo] = Field(..., description="Dictionary of available clients")
    total: int = Field(..., description="Total number of clients")


class HealthResponse(BaseModel):
    """Response model for health check"""

    status: str = Field(..., description="Service status")
    clients_count: int = Field(..., description="Number of registered clients")
    available_clients: list[str] = Field(..., description="List of available client names")