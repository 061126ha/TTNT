"""
HAUI Agent package — logging configuration.

Configures logging for all src.agent.* modules so that
node execution steps are visible in the terminal.
"""

import logging

# Prevent duplicate handlers when Streamlit / reload imports multiple times
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)s │ %(levelname)s │ %(message)s",
    datefmt="%H:%M:%S",
    force=True,  # IMPORTANT: overrides existing Streamlit logging config
)

# Our modules
logging.getLogger("src.agent").setLevel(logging.INFO)
logging.getLogger("src.service").setLevel(logging.INFO)
logging.getLogger("src.llm").setLevel(logging.INFO)

# Third-party noise suppression
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)