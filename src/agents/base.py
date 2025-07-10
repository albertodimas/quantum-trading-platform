"""
Base agent class for all AI agents.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime
import asyncio

from src.core.logging import get_logger

logger = get_logger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all trading agents.
    
    Each agent specializes in a specific aspect of trading analysis
    and decision making.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize base agent."""
        self.name = name
        self.config = config or {}
        self._running = False
        self._last_analysis = None
        self._confidence_threshold = self.config.get("confidence_threshold", 0.7)
        
    @abstractmethod
    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform analysis on provided data.
        
        Args:
            data: Input data for analysis
            
        Returns:
            Analysis results with confidence scores
        """
        pass
    
    @abstractmethod
    async def get_signals(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate trading signals based on analysis.
        
        Args:
            analysis: Analysis results
            
        Returns:
            List of trading signals
        """
        pass
    
    async def start(self):
        """Start the agent."""
        self._running = True
        logger.info(f"Agent {self.name} started")
        
    async def stop(self):
        """Stop the agent."""
        self._running = False
        logger.info(f"Agent {self.name} stopped")
        
    def is_running(self) -> bool:
        """Check if agent is running."""
        return self._running
    
    def get_last_analysis(self) -> Optional[Dict[str, Any]]:
        """Get the last analysis results."""
        return self._last_analysis
    
    def set_confidence_threshold(self, threshold: float):
        """Set minimum confidence threshold for signals."""
        self._confidence_threshold = max(0.0, min(1.0, threshold))
        
    async def validate_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate input data before analysis.
        
        Args:
            data: Input data to validate
            
        Returns:
            True if data is valid
        """
        if not data:
            logger.warning(f"Agent {self.name}: Empty data provided")
            return False
            
        required_fields = self._get_required_fields()
        for field in required_fields:
            if field not in data:
                logger.warning(f"Agent {self.name}: Missing required field '{field}'")
                return False
                
        return True
    
    @abstractmethod
    def _get_required_fields(self) -> List[str]:
        """Get list of required fields for this agent."""
        pass
    
    def _calculate_signal_strength(self, indicators: Dict[str, float]) -> float:
        """
        Calculate overall signal strength from multiple indicators.
        
        Args:
            indicators: Dictionary of indicator scores (0-1)
            
        Returns:
            Combined signal strength (0-1)
        """
        if not indicators:
            return 0.0
            
        # Weighted average of indicators
        weights = self.config.get("indicator_weights", {})
        total_weight = 0.0
        weighted_sum = 0.0
        
        for indicator, score in indicators.items():
            weight = weights.get(indicator, 1.0)
            weighted_sum += score * weight
            total_weight += weight
            
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    async def _store_analysis(self, analysis: Dict[str, Any]):
        """Store analysis results."""
        self._last_analysis = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent": self.name,
            **analysis
        }
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics."""
        return {
            "name": self.name,
            "running": self._running,
            "confidence_threshold": self._confidence_threshold,
            "last_analysis_time": self._last_analysis["timestamp"] if self._last_analysis else None,
        }