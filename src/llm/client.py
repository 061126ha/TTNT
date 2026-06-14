"""
OpenAI-compatible clients for OpenRouter.
"""

import logging

from openai import OpenAI
from langchain_openai import ChatOpenAI
from src.config import settings

logger = logging.getLogger(__name__)

_client: OpenAI | None = None
_chat_model: ChatOpenAI | None = None


def get_client() -> OpenAI:
    """Raw OpenAI client for embeddings."""

    global _client

    if _client is None:
        api_key = settings.openrouter_api_key
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set.")

        _client = OpenAI(
            base_url=settings.openrouter_base_url,
            api_key=api_key,
            default_headers={
                "HTTP-Referer": "http://localhost",
                "X-Title": "HAUI-Agent",
            },
        )

        logger.info("OpenAI client initialized (base_url=%s)", settings.openrouter_base_url)

    return _client


def get_chat_model() -> ChatOpenAI:
    """Chat model for tool calling + reasoning."""

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

        logger.info("Chat model initialized (model=%s)", settings.chat_model)

    return _chat_model