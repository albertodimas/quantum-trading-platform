"""
WebSocket manager for real-time communications.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Set

from fastapi import WebSocket, WebSocketDisconnect
from redis.asyncio import Redis

from src.core.logging import get_logger

logger = get_logger(__name__)


class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        """Initialize connection manager."""
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, Set[str]] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept and store WebSocket connection."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.subscriptions[client_id] = set()
        logger.info(f"Client {client_id} connected")
        
    def disconnect(self, client_id: str):
        """Remove WebSocket connection."""
        self.active_connections.pop(client_id, None)
        self.subscriptions.pop(client_id, None)
        logger.info(f"Client {client_id} disconnected")
        
    async def send_personal_message(self, message: str, client_id: str):
        """Send message to specific client."""
        websocket = self.active_connections.get(client_id)
        if websocket:
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)
                
    async def broadcast(self, message: str, channel: Optional[str] = None):
        """Broadcast message to all or subscribed clients."""
        disconnected_clients = []
        
        for client_id, websocket in self.active_connections.items():
            # Check if client is subscribed to channel
            if channel and channel not in self.subscriptions.get(client_id, set()):
                continue
                
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting to {client_id}: {e}")
                disconnected_clients.append(client_id)
                
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)
            
    def subscribe(self, client_id: str, channel: str):
        """Subscribe client to a channel."""
        if client_id in self.subscriptions:
            self.subscriptions[client_id].add(channel)
            logger.info(f"Client {client_id} subscribed to {channel}")
            
    def unsubscribe(self, client_id: str, channel: str):
        """Unsubscribe client from a channel."""
        if client_id in self.subscriptions:
            self.subscriptions[client_id].discard(channel)
            logger.info(f"Client {client_id} unsubscribed from {channel}")


class WebSocketManager:
    """
    WebSocket manager for real-time data streaming.
    
    Handles market data, order updates, and notifications.
    """
    
    def __init__(self, redis_client: Redis):
        """Initialize WebSocket manager."""
        self.redis = redis_client
        self.manager = ConnectionManager()
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
    async def start(self):
        """Start WebSocket manager background tasks."""
        if self._running:
            return
            
        self._running = True
        
        # Start Redis subscription handlers
        self._tasks = [
            asyncio.create_task(self._handle_market_data()),
            asyncio.create_task(self._handle_order_updates()),
            asyncio.create_task(self._handle_notifications()),
        ]
        
        logger.info("WebSocket manager started")
        
    async def stop(self):
        """Stop WebSocket manager."""
        self._running = False
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
            
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        
        logger.info("WebSocket manager stopped")
        
    async def handle_connection(self, websocket: WebSocket, client_id: str):
        """Handle WebSocket connection lifecycle."""
        await self.manager.connect(websocket, client_id)
        
        try:
            # Send welcome message
            await websocket.send_json({
                "type": "connection",
                "status": "connected",
                "timestamp": datetime.utcnow().isoformat(),
                "client_id": client_id,
            })
            
            # Handle incoming messages
            while True:
                data = await websocket.receive_text()
                await self._handle_client_message(client_id, data)
                
        except WebSocketDisconnect:
            self.manager.disconnect(client_id)
        except Exception as e:
            logger.error(f"WebSocket error for {client_id}: {e}")
            self.manager.disconnect(client_id)
            
    async def _handle_client_message(self, client_id: str, message: str):
        """Handle incoming client message."""
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            
            if msg_type == "subscribe":
                # Subscribe to channels
                channels = data.get("channels", [])
                for channel in channels:
                    self.manager.subscribe(client_id, channel)
                    
                # Send confirmation
                await self.manager.send_personal_message(
                    json.dumps({
                        "type": "subscribed",
                        "channels": channels,
                        "timestamp": datetime.utcnow().isoformat(),
                    }),
                    client_id
                )
                
            elif msg_type == "unsubscribe":
                # Unsubscribe from channels
                channels = data.get("channels", [])
                for channel in channels:
                    self.manager.unsubscribe(client_id, channel)
                    
                # Send confirmation
                await self.manager.send_personal_message(
                    json.dumps({
                        "type": "unsubscribed",
                        "channels": channels,
                        "timestamp": datetime.utcnow().isoformat(),
                    }),
                    client_id
                )
                
            elif msg_type == "ping":
                # Respond to ping
                await self.manager.send_personal_message(
                    json.dumps({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat(),
                    }),
                    client_id
                )
                
            else:
                # Unknown message type
                await self.manager.send_personal_message(
                    json.dumps({
                        "type": "error",
                        "message": f"Unknown message type: {msg_type}",
                        "timestamp": datetime.utcnow().isoformat(),
                    }),
                    client_id
                )
                
        except json.JSONDecodeError:
            await self.manager.send_personal_message(
                json.dumps({
                    "type": "error",
                    "message": "Invalid JSON",
                    "timestamp": datetime.utcnow().isoformat(),
                }),
                client_id
            )
        except Exception as e:
            logger.error(f"Error handling client message: {e}")
            
    async def _handle_market_data(self):
        """Subscribe to and broadcast market data updates."""
        pubsub = self.redis.pubsub()
        await pubsub.psubscribe("market_data:*")
        
        while self._running:
            try:
                message = await pubsub.get_message(ignore_subscribe_messages=True)
                if message and message["type"] == "pmessage":
                    # Extract channel name
                    channel = message["channel"].decode().replace("market_data:", "")
                    
                    # Broadcast to subscribed clients
                    data = json.loads(message["data"])
                    await self.manager.broadcast(
                        json.dumps({
                            "type": "market_data",
                            "channel": channel,
                            "data": data,
                            "timestamp": datetime.utcnow().isoformat(),
                        }),
                        channel=f"market:{channel}"
                    )
                    
                await asyncio.sleep(0.01)  # Small delay to prevent CPU spinning
                
            except Exception as e:
                logger.error(f"Error in market data handler: {e}")
                await asyncio.sleep(1)
                
    async def _handle_order_updates(self):
        """Subscribe to and broadcast order updates."""
        pubsub = self.redis.pubsub()
        await pubsub.subscribe("order_updates")
        
        while self._running:
            try:
                message = await pubsub.get_message(ignore_subscribe_messages=True)
                if message and message["type"] == "message":
                    data = json.loads(message["data"])
                    
                    # Broadcast to user
                    user_id = data.get("user_id")
                    if user_id:
                        await self.manager.send_personal_message(
                            json.dumps({
                                "type": "order_update",
                                "data": data,
                                "timestamp": datetime.utcnow().isoformat(),
                            }),
                            user_id
                        )
                        
                await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in order update handler: {e}")
                await asyncio.sleep(1)
                
    async def _handle_notifications(self):
        """Subscribe to and broadcast notifications."""
        pubsub = self.redis.pubsub()
        await pubsub.subscribe("notifications")
        
        while self._running:
            try:
                message = await pubsub.get_message(ignore_subscribe_messages=True)
                if message and message["type"] == "message":
                    data = json.loads(message["data"])
                    
                    # Broadcast notification
                    notification_type = data.get("type", "general")
                    await self.manager.broadcast(
                        json.dumps({
                            "type": "notification",
                            "notification_type": notification_type,
                            "data": data,
                            "timestamp": datetime.utcnow().isoformat(),
                        }),
                        channel=f"notifications:{notification_type}"
                    )
                    
                await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in notification handler: {e}")
                await asyncio.sleep(1)