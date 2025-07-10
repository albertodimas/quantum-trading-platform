"""
API module for Quantum Trading Platform.

This module provides REST and WebSocket APIs for external
integrations and frontend applications.
"""

from src.api.app import create_app
from src.api.websocket import WebSocketManager

__all__ = ["create_app", "WebSocketManager"]