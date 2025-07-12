"""
Analytics Dashboard Module

Provides real-time and historical performance visualization with
interactive charts, metrics, and customizable views.
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict

from ..core.interfaces import Injectable
from ..core.decorators import injectable
from ..core.logger import get_logger
from ..core.event_bus import EventBus, Event
from ..core.websocket import WebSocketManager
from .performance_analyzer import PerformanceAnalyzer, AnalysisTimeframe

logger = get_logger(__name__)


class DashboardView(Enum):
    """Available dashboard views"""
    OVERVIEW = "overview"
    PERFORMANCE = "performance"
    POSITIONS = "positions"
    RISK = "risk"
    STRATEGIES = "strategies"
    MARKET_DATA = "market_data"
    SYSTEM_HEALTH = "system_health"


class ChartType(Enum):
    """Available chart types"""
    LINE = "line"
    AREA = "area"
    CANDLESTICK = "candlestick"
    BAR = "bar"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    GAUGE = "gauge"
    PIE = "pie"
    TREEMAP = "treemap"


class UpdateFrequency(Enum):
    """Dashboard update frequencies"""
    REAL_TIME = "real_time"
    SECOND = "second"
    MINUTE = "minute"
    FIVE_MINUTES = "five_minutes"
    HOURLY = "hourly"
    DAILY = "daily"


@dataclass
class DashboardConfig:
    """Dashboard configuration"""
    default_view: DashboardView = DashboardView.OVERVIEW
    update_frequency: UpdateFrequency = UpdateFrequency.SECOND
    enable_websocket: bool = True
    websocket_port: int = 8765
    chart_history_days: int = 30
    max_data_points: int = 1000
    theme: str = "dark"
    layout_config: Dict[str, Any] = field(default_factory=dict)
    enabled_widgets: List[str] = field(default_factory=lambda: [
        "equity_curve", "pnl_chart", "positions_table", "metrics_panel",
        "risk_gauges", "strategy_performance", "market_heatmap", "system_status"
    ])


@dataclass
class Widget:
    """Dashboard widget definition"""
    widget_id: str
    title: str
    type: ChartType
    data: Dict[str, Any]
    config: Dict[str, Any]
    position: Dict[str, int]  # x, y, width, height
    update_frequency: UpdateFrequency
    last_update: datetime


@dataclass
class DashboardState:
    """Current dashboard state"""
    active_view: DashboardView
    widgets: Dict[str, Widget]
    filters: Dict[str, Any]
    time_range: Tuple[Optional[datetime], Optional[datetime]]
    selected_strategies: List[str]
    selected_symbols: List[str]
    refresh_rate: UpdateFrequency


@injectable
class AnalyticsDashboard(Injectable):
    """Real-time analytics dashboard"""
    
    def __init__(
        self,
        config: DashboardConfig = None,
        performance_analyzer: PerformanceAnalyzer = None,
        event_bus: EventBus = None,
        websocket_manager: WebSocketManager = None
    ):
        self.config = config or DashboardConfig()
        self.performance_analyzer = performance_analyzer
        self.event_bus = event_bus
        self.websocket_manager = websocket_manager
        
        # Dashboard state
        self.state = DashboardState(
            active_view=self.config.default_view,
            widgets={},
            filters={},
            time_range=(None, None),
            selected_strategies=[],
            selected_symbols=[],
            refresh_rate=self.config.update_frequency
        )
        
        # Widget registry
        self._widget_builders: Dict[str, Callable] = {}
        self._register_default_widgets()
        
        # Update tasks
        self._update_tasks: Dict[str, asyncio.Task] = {}
        
        # Client connections
        self._clients: List[str] = []
        
    async def start(self):
        """Start dashboard service"""
        # Initialize widgets
        await self._initialize_widgets()
        
        # Start WebSocket server if enabled
        if self.config.enable_websocket and self.websocket_manager:
            await self.websocket_manager.start(self.config.websocket_port)
            logger.info(f"Dashboard WebSocket server started on port {self.config.websocket_port}")
        
        # Subscribe to events
        if self.event_bus:
            await self._subscribe_to_events()
        
        # Start update tasks
        await self._start_update_tasks()
        
        logger.info("Analytics dashboard started")
    
    async def stop(self):
        """Stop dashboard service"""
        # Cancel update tasks
        for task in self._update_tasks.values():
            task.cancel()
        
        # Stop WebSocket server
        if self.websocket_manager:
            await self.websocket_manager.stop()
        
        logger.info("Analytics dashboard stopped")
    
    async def get_view_data(self, view: DashboardView) -> Dict[str, Any]:
        """Get data for specific dashboard view"""
        # Switch to requested view
        self.state.active_view = view
        
        # Get widgets for this view
        view_widgets = self._get_widgets_for_view(view)
        
        # Update widget data
        widget_data = {}
        for widget_id, widget in view_widgets.items():
            widget_data[widget_id] = {
                "title": widget.title,
                "type": widget.type.value,
                "data": widget.data,
                "config": widget.config,
                "position": widget.position,
                "last_update": widget.last_update.isoformat()
            }
        
        return {
            "view": view.value,
            "widgets": widget_data,
            "filters": self.state.filters,
            "time_range": {
                "start": self.state.time_range[0].isoformat() if self.state.time_range[0] else None,
                "end": self.state.time_range[1].isoformat() if self.state.time_range[1] else None
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def update_widget(self, widget_id: str, data: Dict[str, Any] = None):
        """Update specific widget data"""
        if widget_id not in self.state.widgets:
            logger.warning(f"Widget {widget_id} not found")
            return
        
        widget = self.state.widgets[widget_id]
        
        # Update data if provided, otherwise refresh
        if data:
            widget.data = data
        else:
            # Use widget builder to refresh data
            if widget_id in self._widget_builders:
                widget.data = await self._widget_builders[widget_id]()
        
        widget.last_update = datetime.utcnow()
        
        # Broadcast update to clients
        await self._broadcast_widget_update(widget_id, widget)
    
    async def set_filters(self, filters: Dict[str, Any]):
        """Update dashboard filters"""
        self.state.filters.update(filters)
        
        # Update time range
        if "start_date" in filters:
            self.state.time_range = (filters["start_date"], self.state.time_range[1])
        if "end_date" in filters:
            self.state.time_range = (self.state.time_range[0], filters["end_date"])
        
        # Update selections
        if "strategies" in filters:
            self.state.selected_strategies = filters["strategies"]
        if "symbols" in filters:
            self.state.selected_symbols = filters["symbols"]
        
        # Refresh all widgets with new filters
        await self._refresh_all_widgets()
    
    async def export_dashboard_data(self, format: str = "json") -> str:
        """Export dashboard data"""
        data = {
            "timestamp": datetime.utcnow().isoformat(),
            "view": self.state.active_view.value,
            "widgets": {}
        }
        
        # Collect all widget data
        for widget_id, widget in self.state.widgets.items():
            data["widgets"][widget_id] = {
                "title": widget.title,
                "type": widget.type.value,
                "data": widget.data,
                "last_update": widget.last_update.isoformat()
            }
        
        if format == "json":
            return json.dumps(data, indent=2)
        else:
            # TODO: Support other formats (CSV, PDF, etc.)
            return json.dumps(data)
    
    def _register_default_widgets(self):
        """Register default widget builders"""
        self._widget_builders.update({
            "equity_curve": self._build_equity_curve_widget,
            "pnl_chart": self._build_pnl_chart_widget,
            "positions_table": self._build_positions_table_widget,
            "metrics_panel": self._build_metrics_panel_widget,
            "risk_gauges": self._build_risk_gauges_widget,
            "strategy_performance": self._build_strategy_performance_widget,
            "market_heatmap": self._build_market_heatmap_widget,
            "system_status": self._build_system_status_widget
        })
    
    async def _initialize_widgets(self):
        """Initialize dashboard widgets"""
        # Default layout positions
        default_positions = {
            "equity_curve": {"x": 0, "y": 0, "width": 8, "height": 4},
            "pnl_chart": {"x": 8, "y": 0, "width": 4, "height": 4},
            "positions_table": {"x": 0, "y": 4, "width": 6, "height": 4},
            "metrics_panel": {"x": 6, "y": 4, "width": 6, "height": 4},
            "risk_gauges": {"x": 0, "y": 8, "width": 4, "height": 3},
            "strategy_performance": {"x": 4, "y": 8, "width": 4, "height": 3},
            "market_heatmap": {"x": 8, "y": 8, "width": 4, "height": 3},
            "system_status": {"x": 0, "y": 11, "width": 12, "height": 2}
        }
        
        # Create widgets
        for widget_id in self.config.enabled_widgets:
            if widget_id in self._widget_builders:
                # Get widget configuration
                widget_config = self._get_widget_config(widget_id)
                
                # Build initial data
                data = await self._widget_builders[widget_id]()
                
                # Create widget
                widget = Widget(
                    widget_id=widget_id,
                    title=widget_config["title"],
                    type=widget_config["type"],
                    data=data,
                    config=widget_config.get("config", {}),
                    position=default_positions.get(widget_id, {"x": 0, "y": 0, "width": 4, "height": 4}),
                    update_frequency=widget_config.get("update_frequency", self.config.update_frequency),
                    last_update=datetime.utcnow()
                )
                
                self.state.widgets[widget_id] = widget
    
    def _get_widget_config(self, widget_id: str) -> Dict[str, Any]:
        """Get widget configuration"""
        widget_configs = {
            "equity_curve": {
                "title": "Equity Curve",
                "type": ChartType.AREA,
                "update_frequency": UpdateFrequency.MINUTE
            },
            "pnl_chart": {
                "title": "P&L Analysis",
                "type": ChartType.BAR,
                "update_frequency": UpdateFrequency.MINUTE
            },
            "positions_table": {
                "title": "Open Positions",
                "type": ChartType.BAR,  # Table rendered as bar chart
                "update_frequency": UpdateFrequency.SECOND
            },
            "metrics_panel": {
                "title": "Performance Metrics",
                "type": ChartType.GAUGE,
                "update_frequency": UpdateFrequency.MINUTE
            },
            "risk_gauges": {
                "title": "Risk Indicators",
                "type": ChartType.GAUGE,
                "update_frequency": UpdateFrequency.SECOND
            },
            "strategy_performance": {
                "title": "Strategy Performance",
                "type": ChartType.BAR,
                "update_frequency": UpdateFrequency.FIVE_MINUTES
            },
            "market_heatmap": {
                "title": "Market Heatmap",
                "type": ChartType.HEATMAP,
                "update_frequency": UpdateFrequency.MINUTE
            },
            "system_status": {
                "title": "System Status",
                "type": ChartType.GAUGE,
                "update_frequency": UpdateFrequency.SECOND
            }
        }
        
        return widget_configs.get(widget_id, {
            "title": widget_id,
            "type": ChartType.LINE,
            "update_frequency": UpdateFrequency.MINUTE
        })
    
    async def _build_equity_curve_widget(self) -> Dict[str, Any]:
        """Build equity curve widget data"""
        if not self.performance_analyzer:
            return {}
        
        # Get performance snapshots
        snapshots = self.performance_analyzer._snapshots
        
        # Apply time filter
        filtered_snapshots = self._filter_by_time_range(snapshots)
        
        # Prepare chart data
        data = {
            "labels": [s.timestamp.isoformat() for s in filtered_snapshots],
            "datasets": [{
                "label": "Equity",
                "data": [s.equity for s in filtered_snapshots],
                "fill": True
            }]
        }
        
        # Add benchmark if available
        if self.performance_analyzer.config.calculate_benchmarks:
            # TODO: Add benchmark data
            pass
        
        return data
    
    async def _build_pnl_chart_widget(self) -> Dict[str, Any]:
        """Build P&L chart widget data"""
        if not self.performance_analyzer:
            return {}
        
        # Get performance snapshots
        snapshots = self.performance_analyzer._snapshots
        filtered_snapshots = self._filter_by_time_range(snapshots)
        
        # Calculate daily P&L
        daily_pnl = []
        for i in range(1, len(filtered_snapshots)):
            pnl = filtered_snapshots[i].equity - filtered_snapshots[i-1].equity
            daily_pnl.append(pnl)
        
        return {
            "labels": [s.timestamp.strftime("%Y-%m-%d") for s in filtered_snapshots[1:]],
            "datasets": [{
                "label": "Daily P&L",
                "data": daily_pnl,
                "backgroundColor": ["green" if p > 0 else "red" for p in daily_pnl]
            }]
        }
    
    async def _build_positions_table_widget(self) -> Dict[str, Any]:
        """Build positions table widget data"""
        # Get current positions from performance analyzer
        positions = list(self.performance_analyzer._positions.values())
        
        # Filter by selected symbols
        if self.state.selected_symbols:
            positions = [p for p in positions if p.symbol in self.state.selected_symbols]
        
        # Format for table display
        return {
            "columns": ["Symbol", "Side", "Quantity", "Entry Price", "Current Price", "P&L", "P&L %"],
            "rows": [
                [
                    p.symbol,
                    p.side,
                    p.quantity,
                    p.entry_price,
                    p.current_price,
                    p.unrealized_pnl,
                    f"{(p.unrealized_pnl / (p.quantity * p.entry_price)) * 100:.2f}%"
                ]
                for p in positions
            ]
        }
    
    async def _build_metrics_panel_widget(self) -> Dict[str, Any]:
        """Build metrics panel widget data"""
        if not self.performance_analyzer:
            return {}
        
        # Get real-time metrics
        metrics = await self.performance_analyzer.get_real_time_metrics()
        
        return {
            "metrics": [
                {"name": "Total Equity", "value": f"${metrics.get('equity', 0):,.2f}"},
                {"name": "Daily Return", "value": f"{metrics.get('daily_return', 0) * 100:.2f}%"},
                {"name": "Cumulative Return", "value": f"{metrics.get('cumulative_return', 0) * 100:.2f}%"},
                {"name": "Sharpe Ratio", "value": f"{metrics.get('sharpe_ratio', 0):.2f}"},
                {"name": "Max Drawdown", "value": f"{metrics.get('drawdown', 0) * 100:.2f}%"},
                {"name": "Win Rate", "value": f"{metrics.get('win_rate', 0) * 100:.1f}%"},
                {"name": "Trades Today", "value": metrics.get('trades_today', 0)},
                {"name": "Open Positions", "value": metrics.get('open_positions', 0)}
            ]
        }
    
    async def _build_risk_gauges_widget(self) -> Dict[str, Any]:
        """Build risk gauges widget data"""
        # TODO: Get risk metrics from risk manager
        return {
            "gauges": [
                {
                    "name": "Portfolio Risk",
                    "value": 35,
                    "max": 100,
                    "zones": [
                        {"from": 0, "to": 30, "color": "green"},
                        {"from": 30, "to": 70, "color": "yellow"},
                        {"from": 70, "to": 100, "color": "red"}
                    ]
                },
                {
                    "name": "Leverage",
                    "value": 1.5,
                    "max": 5,
                    "zones": [
                        {"from": 0, "to": 2, "color": "green"},
                        {"from": 2, "to": 3, "color": "yellow"},
                        {"from": 3, "to": 5, "color": "red"}
                    ]
                },
                {
                    "name": "Exposure",
                    "value": 60,
                    "max": 100,
                    "zones": [
                        {"from": 0, "to": 50, "color": "green"},
                        {"from": 50, "to": 80, "color": "yellow"},
                        {"from": 80, "to": 100, "color": "red"}
                    ]
                }
            ]
        }
    
    async def _build_strategy_performance_widget(self) -> Dict[str, Any]:
        """Build strategy performance widget data"""
        if not self.performance_analyzer:
            return {}
        
        # Get strategy analysis
        analysis = await self.performance_analyzer.analyze(
            timeframe=AnalysisTimeframe.DAILY,
            analysis_types=[AnalysisType.STRATEGY]
        )
        
        strategies = analysis.get("strategies", {})
        
        return {
            "labels": list(strategies.keys()),
            "datasets": [
                {
                    "label": "Total Return",
                    "data": [s.metrics.total_return * 100 for s in strategies.values()]
                },
                {
                    "label": "Sharpe Ratio",
                    "data": [s.metrics.sharpe_ratio for s in strategies.values()]
                }
            ]
        }
    
    async def _build_market_heatmap_widget(self) -> Dict[str, Any]:
        """Build market heatmap widget data"""
        # TODO: Get market data
        return {
            "symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"],
            "data": [
                [2.5, -1.2, 0.8, 3.1],
                [-0.5, 1.8, -2.1, 0.3],
                [1.2, 0.5, -0.8, 2.2],
                [0.8, -1.5, 1.9, -0.2]
            ],
            "colorScale": {
                "min": -5,
                "max": 5,
                "negative": "red",
                "positive": "green"
            }
        }
    
    async def _build_system_status_widget(self) -> Dict[str, Any]:
        """Build system status widget data"""
        # TODO: Get system health metrics
        return {
            "components": [
                {"name": "Trading Engine", "status": "healthy", "uptime": "99.9%"},
                {"name": "Market Data", "status": "healthy", "latency": "12ms"},
                {"name": "Risk Manager", "status": "healthy", "checks": "152/152"},
                {"name": "Database", "status": "warning", "usage": "85%"},
                {"name": "Message Queue", "status": "healthy", "queue": "234"}
            ]
        }
    
    def _get_widgets_for_view(self, view: DashboardView) -> Dict[str, Widget]:
        """Get widgets for specific view"""
        view_widgets = {
            DashboardView.OVERVIEW: [
                "equity_curve", "pnl_chart", "positions_table", "metrics_panel"
            ],
            DashboardView.PERFORMANCE: [
                "equity_curve", "pnl_chart", "metrics_panel", "strategy_performance"
            ],
            DashboardView.POSITIONS: [
                "positions_table", "pnl_chart", "risk_gauges"
            ],
            DashboardView.RISK: [
                "risk_gauges", "positions_table", "market_heatmap"
            ],
            DashboardView.STRATEGIES: [
                "strategy_performance", "equity_curve", "metrics_panel"
            ],
            DashboardView.MARKET_DATA: [
                "market_heatmap", "positions_table"
            ],
            DashboardView.SYSTEM_HEALTH: [
                "system_status", "metrics_panel"
            ]
        }
        
        widget_ids = view_widgets.get(view, [])
        return {wid: self.state.widgets[wid] for wid in widget_ids if wid in self.state.widgets}
    
    def _filter_by_time_range(self, items: List[Any]) -> List[Any]:
        """Filter items by time range"""
        if not items:
            return items
        
        filtered = items
        
        if self.state.time_range[0]:
            filtered = [i for i in filtered if hasattr(i, 'timestamp') and i.timestamp >= self.state.time_range[0]]
        
        if self.state.time_range[1]:
            filtered = [i for i in filtered if hasattr(i, 'timestamp') and i.timestamp <= self.state.time_range[1]]
        
        return filtered
    
    async def _subscribe_to_events(self):
        """Subscribe to relevant events"""
        event_handlers = {
            "trade.executed": self._handle_trade_event,
            "position.updated": self._handle_position_event,
            "performance.snapshot": self._handle_performance_event,
            "risk.alert": self._handle_risk_event,
            "system.health": self._handle_system_event
        }
        
        for event_type, handler in event_handlers.items():
            await self.event_bus.subscribe(event_type, handler)
    
    async def _handle_trade_event(self, event: Event):
        """Handle trade execution event"""
        # Update relevant widgets
        await self.update_widget("pnl_chart")
        await self.update_widget("metrics_panel")
        await self.update_widget("strategy_performance")
    
    async def _handle_position_event(self, event: Event):
        """Handle position update event"""
        await self.update_widget("positions_table")
        await self.update_widget("risk_gauges")
    
    async def _handle_performance_event(self, event: Event):
        """Handle performance snapshot event"""
        await self.update_widget("equity_curve")
        await self.update_widget("metrics_panel")
    
    async def _handle_risk_event(self, event: Event):
        """Handle risk alert event"""
        await self.update_widget("risk_gauges")
        
        # Broadcast alert to clients
        if self.websocket_manager:
            await self.websocket_manager.broadcast({
                "type": "risk_alert",
                "data": event.data
            })
    
    async def _handle_system_event(self, event: Event):
        """Handle system health event"""
        await self.update_widget("system_status")
    
    async def _start_update_tasks(self):
        """Start periodic widget update tasks"""
        for widget_id, widget in self.state.widgets.items():
            if widget.update_frequency != UpdateFrequency.REAL_TIME:
                interval = self._get_update_interval(widget.update_frequency)
                task = asyncio.create_task(
                    self._periodic_widget_update(widget_id, interval)
                )
                self._update_tasks[widget_id] = task
    
    async def _periodic_widget_update(self, widget_id: str, interval: float):
        """Periodically update widget"""
        while True:
            try:
                await asyncio.sleep(interval)
                await self.update_widget(widget_id)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error updating widget {widget_id}: {e}")
    
    def _get_update_interval(self, frequency: UpdateFrequency) -> float:
        """Get update interval in seconds"""
        intervals = {
            UpdateFrequency.SECOND: 1,
            UpdateFrequency.MINUTE: 60,
            UpdateFrequency.FIVE_MINUTES: 300,
            UpdateFrequency.HOURLY: 3600,
            UpdateFrequency.DAILY: 86400
        }
        return intervals.get(frequency, 60)
    
    async def _refresh_all_widgets(self):
        """Refresh all widgets with current filters"""
        for widget_id in self.state.widgets:
            await self.update_widget(widget_id)
    
    async def _broadcast_widget_update(self, widget_id: str, widget: Widget):
        """Broadcast widget update to all clients"""
        if self.websocket_manager:
            await self.websocket_manager.broadcast({
                "type": "widget_update",
                "widget_id": widget_id,
                "data": {
                    "title": widget.title,
                    "type": widget.type.value,
                    "data": widget.data,
                    "last_update": widget.last_update.isoformat()
                }
            })
    
    async def connect_client(self, client_id: str):
        """Connect new dashboard client"""
        self._clients.append(client_id)
        
        # Send initial dashboard state
        initial_data = await self.get_view_data(self.state.active_view)
        
        if self.websocket_manager:
            await self.websocket_manager.send_to_client(client_id, {
                "type": "initial_state",
                "data": initial_data
            })
    
    async def disconnect_client(self, client_id: str):
        """Disconnect dashboard client"""
        if client_id in self._clients:
            self._clients.remove(client_id)