import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings and configuration"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    # API Configuration
    openai_api_key: Optional[str] = Field(None, alias="OPENAI_API_KEY")
    huggingface_api_key: Optional[str] = Field(None, alias="HUGGINGFACE_API_KEY")

    # OpenAI Configuration
    openai_model: str = Field("gpt-3.5-turbo", alias="OPENAI_MODEL")
    openai_temperature: float = Field(0.7, alias="OPENAI_TEMPERATURE")
    openai_max_tokens: int = Field(150, alias="OPENAI_MAX_TOKENS")

    # Application Configuration
    app_name: str = Field("LLM Client Manager", alias="APP_NAME")
    app_version: str = Field("1.0.0", alias="APP_VERSION")
    debug: bool = Field(False, alias="DEBUG")

    # Server Configuration
    host: str = Field("0.0.0.0", alias="HOST")
    port: int = Field(8010, alias="PORT")

    # CORS Configuration
    cors_origins: str = Field(default="*", alias="CORS_ORIGINS")

    # Request Configuration
    request_timeout: int = Field(30, alias="REQUEST_TIMEOUT")
    max_prompt_length: int = Field(10000, alias="MAX_PROMPT_LENGTH")
    
    def get_cors_origins_list(self) -> list[str]:
        """Convert CORS_ORIGINS string to list"""
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


def get_settings() -> Settings:
    """Get application settings instance"""
    return Settings()


# Global settings instance
settings = get_settings()


def get_openai_config() -> dict:
    """Get OpenAI configuration"""
    return {
        "api_key": settings.openai_api_key,
        "model": settings.openai_model,
        "temperature": settings.openai_temperature,
        "max_tokens": settings.openai_max_tokens
    }


def validate_required_env_vars():
    """Validate that required environment variables are set"""
    errors = []

    # Check if at least one API key is configured
    if not settings.openai_api_key and not settings.huggingface_api_key:
        errors.append("At least one API key must be configured (OPENAI_API_KEY or HUGGINGFACE_API_KEY)")

    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors))


def get_client_configs() -> dict:
    """Get configuration for all supported clients"""
    configs = {}

    # Mock client (always available)
    configs["mock"] = {
        "enabled": True,
        "name": "mock",
        "type": "mock"
    }

    # OpenAI client
    if settings.openai_api_key:
        configs["openai"] = {
            "enabled": True,
            "name": "openai",
            "type": "openai",
            "api_key": settings.openai_api_key,
            "model": settings.openai_model
        }

    return configs