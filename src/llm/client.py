"""
OpenAI-compatible client for OpenRouter.

Provides a singleton client instance via get_client().
"""

import logging

from openai import OpenAI
from src.config import settings

logger = logging.getLogger(__name__)

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Return (and cache) an OpenAI client pointed at OpenRouter."""
    global _client
    if _client is None:
        api_key = settings.openrouter_api_key
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. "
                "Please set it in your .env file or environment variables."
            )
        _client = OpenAI(
            base_url=settings.openrouter_base_url,
            api_key=api_key,
        )
        logger.info("OpenAI client initialised (base_url=%s)", settings.openrouter_base_url)
    return _client
