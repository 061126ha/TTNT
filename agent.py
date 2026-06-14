"""
Bridge file for backward compatibility.
Imports HAUIAgent from the new modular structure in src/
"""

from src.agent.haui_agent import HAUIAgent

__all__ = ["HAUIAgent"]