"""
Bridge file for backward compatibility.
Imports HAUIAgent from the new modular structure in src/
"""

import sys
import os

# Thêm thư mục gốc vào path để đảm bảo import src hoạt động
root_path = os.path.dirname(os.path.abspath(__file__))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.agent.haui_agent import HAUIAgent
__all__ = ["HAUIAgent"]