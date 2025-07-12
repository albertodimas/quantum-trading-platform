"""
Market Data Aggregator for Quantum Trading Platform

This module provides real-time market data aggregation from multiple exchanges,
normalization, and distribution to consuming components.
"""

from .aggregator import MarketDataAggregator, AggregatorConfig
from .normalizer import DataNormalizer
from .collector import MarketDataCollector, CollectorConfig
from .distributor import MarketDataDistributor, DistributorConfig
from .storage import MarketDataStorage, StorageConfig, StorageBackend
from .quality_checker import MarketDataQualityChecker, QualityConfig, QualityCheck, QualityIssue

__all__ = [
    'MarketDataAggregator',
    'AggregatorConfig',
    'DataNormalizer',
    'MarketDataCollector',
    'CollectorConfig',
    'MarketDataDistributor',
    'DistributorConfig',
    'MarketDataStorage',
    'StorageConfig',
    'StorageBackend',
    'MarketDataQualityChecker',
    'QualityConfig',
    'QualityCheck',
    'QualityIssue',
]