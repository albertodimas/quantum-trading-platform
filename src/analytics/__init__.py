"""
Performance Analytics Module

Provides comprehensive performance analysis, metrics calculation,
and dashboard functionality for the quantum trading platform.
"""

from .performance_analyzer import PerformanceAnalyzer, AnalysisConfig
from .metrics_calculator import MetricsCalculator, PerformanceMetrics
from .dashboard import AnalyticsDashboard, DashboardConfig
from .report_generator import ReportGenerator, ReportFormat
from .portfolio_analytics import PortfolioAnalytics
from .trade_analytics import TradeAnalytics
from .risk_analytics import RiskAnalytics

__all__ = [
    'PerformanceAnalyzer',
    'AnalysisConfig',
    'MetricsCalculator',
    'PerformanceMetrics',
    'AnalyticsDashboard',
    'DashboardConfig',
    'ReportGenerator',
    'ReportFormat',
    'PortfolioAnalytics',
    'TradeAnalytics',
    'RiskAnalytics'
]