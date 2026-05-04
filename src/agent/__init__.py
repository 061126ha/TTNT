"""
HAUI Agent package — logging configuration.

Configures logging for all src.agent.* modules so that
node execution steps are visible in the terminal.
"""

import logging

# Configure logging for the agent package
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)

# Set our modules to INFO, suppress noisy third-party loggers
logging.getLogger("src.agent").setLevel(logging.INFO)
logging.getLogger("src.service").setLevel(logging.INFO)
logging.getLogger("src.llm").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
