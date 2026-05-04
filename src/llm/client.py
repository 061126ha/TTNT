"""
OpenAI-compatible clients for OpenRouter.

Provides:
  - get_client()      → raw OpenAI client (for embeddings)
  - get_chat_model()  → ChatOpenAI (for tool-calling, structured output)
"""

import logging

from openai import OpenAI
from langchain_openai import ChatOpenAI
from src.config import settings

logger = logging.getLogger(__name__)

_client: OpenAI | None = None
_chat_model: ChatOpenAI | None = None


def get_client() -> OpenAI:
    """Return (and cache) a raw OpenAI client for embeddings."""
    global _client
    if _client is None:
        api_key = settings.openrouter_api_key
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set.")
        _client = OpenAI(
            base_url=settings.openrouter_base_url,
            api_key=api_key,
        )
        logger.info("OpenAI client initialised (base_url=%s)", settings.openrouter_base_url)
    return _client


def get_chat_model() -> ChatOpenAI:
    """Return (and cache) a ChatOpenAI model for tool-calling and structured output."""
    global _chat_model
    if _chat_model is None:
        api_key = settings.openrouter_api_key
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set.")
        _chat_model = ChatOpenAI(
            model=settings.chat_model,
            openai_api_key=api_key,
            openai_api_base=settings.openrouter_base_url,
            temperature=0,
        )
        logger.info("ChatOpenAI model initialised (model=%s)", settings.chat_model)
    return _chat_model
