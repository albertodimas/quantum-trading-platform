"""
Core module for Quantum Trading Platform.

This module contains the fundamental components and utilities
that are used throughout the application.
"""

from src.core.config import settings
from src.core.logging import get_logger

__all__ = ["settings", "get_logger"]