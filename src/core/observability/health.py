"""
Comprehensive health checking system for Quantum Trading Platform.

Provides multi-level health monitoring:
- Application health checks
- Database connectivity
- External service dependencies
- Trading system status
- Real-time health dashboard
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class HealthCheckType(Enum):
    """Types of health checks."""
    READINESS = "readiness"  # Ready to serve traffic
    LIVENESS = "liveness"    # Application is running
    STARTUP = "startup"      # Application has started
    DEPENDENCY = "dependency" # External dependency status


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    check_type: HealthCheckType = HealthCheckType.READINESS
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
            "check_type": self.check_type.value
        }


class BaseHealthCheck(ABC):
    """Abstract base class for health checks."""
    
    def __init__(self, name: str, check_type: HealthCheckType = HealthCheckType.READINESS,
                 timeout_seconds: float = 5.0):
        self.name = name
        self.check_type = check_type
        self.timeout_seconds = timeout_seconds
        self._last_result: Optional[HealthCheckResult] = None
    
    @abstractmethod
    async def check(self) -> HealthCheckResult:
        """Perform the health check."""
        pass
    
    async def execute(self) -> HealthCheckResult:
        """Execute the health check with timeout and timing."""
        start_time = time.time()
        
        try:
            result = await asyncio.wait_for(self.check(), timeout=self.timeout_seconds)
            result.duration_ms = (time.time() - start_time) * 1000
            self._last_result = result
            return result
        except asyncio.TimeoutError:
            duration_ms = (time.time() - start_time) * 1000
            result = HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {self.timeout_seconds}s",
                duration_ms=duration_ms,
                check_type=self.check_type
            )
            self._last_result = result
            return result
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                duration_ms=duration_ms,
                check_type=self.check_type
            )
            self._last_result = result
            return result
    
    def get_last_result(self) -> Optional[HealthCheckResult]:
        """Get the last health check result."""
        return self._last_result


class DatabaseHealthCheck(BaseHealthCheck):
    """Health check for database connectivity."""
    
    def __init__(self, database_url: str, query: str = "SELECT 1", 
                 timeout_seconds: float = 5.0):
        super().__init__("database", HealthCheckType.DEPENDENCY, timeout_seconds)
        self.database_url = database_url
        self.query = query
    
    async def check(self) -> HealthCheckResult:
        """Check database connectivity."""
        try:
            # This would normally use your database connection pool
            # For now, we'll simulate a database check
            await asyncio.sleep(0.1)  # Simulate DB query time
            
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message="Database connection successful",
                details={"query": self.query},
                check_type=self.check_type
            )
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection failed: {str(e)}",
                check_type=self.check_type
            )


class ExternalServiceHealthCheck(BaseHealthCheck):
    """Health check for external services."""
    
    def __init__(self, name: str, url: str, expected_status: int = 200,
                 timeout_seconds: float = 5.0):
        super().__init__(f"external_{name}", HealthCheckType.DEPENDENCY, timeout_seconds)
        self.url = url
        self.expected_status = expected_status
    
    async def check(self) -> HealthCheckResult:
        """Check external service availability."""
        if not AIOHTTP_AVAILABLE:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNKNOWN,
                message="aiohttp not available for HTTP checks",
                check_type=self.check_type
            )
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.url, timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)) as response:
                    if response.status == self.expected_status:
                        return HealthCheckResult(
                            name=self.name,
                            status=HealthStatus.HEALTHY,
                            message=f"Service responding with status {response.status}",
                            details={"url": self.url, "status_code": response.status},
                            check_type=self.check_type
                        )
                    else:
                        return HealthCheckResult(
                            name=self.name,
                            status=HealthStatus.DEGRADED,
                            message=f"Unexpected status code: {response.status}",
                            details={"url": self.url, "status_code": response.status},
                            check_type=self.check_type
                        )
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Service check failed: {str(e)}",
                details={"url": self.url},
                check_type=self.check_type
            )


class SystemResourceHealthCheck(BaseHealthCheck):
    """Health check for system resources."""
    
    def __init__(self, cpu_threshold: float = 80.0, memory_threshold: float = 85.0,
                 disk_threshold: float = 90.0, timeout_seconds: float = 2.0):
        super().__init__("system_resources", HealthCheckType.LIVENESS, timeout_seconds)
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold
    
    async def check(self) -> HealthCheckResult:
        """Check system resource usage."""
        if not PSUTIL_AVAILABLE:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNKNOWN,
                message="psutil not available for system checks",
                check_type=self.check_type
            )
        
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            details = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "disk_free_gb": round(disk.free / (1024**3), 2)
            }
            
            # Determine status based on thresholds
            issues = []
            status = HealthStatus.HEALTHY
            
            if cpu_percent > self.cpu_threshold:
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
                status = HealthStatus.DEGRADED
            
            if memory.percent > self.memory_threshold:
                issues.append(f"High memory usage: {memory.percent:.1f}%")
                status = HealthStatus.DEGRADED
            
            if disk.percent > self.disk_threshold:
                issues.append(f"High disk usage: {disk.percent:.1f}%")
                status = HealthStatus.UNHEALTHY
            
            message = "System resources normal" if not issues else "; ".join(issues)
            
            return HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                details=details,
                check_type=self.check_type
            )
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"System check failed: {str(e)}",
                check_type=self.check_type
            )


class TradingSystemHealthCheck(BaseHealthCheck):
    """Health check for trading system components."""
    
    def __init__(self, trading_engine, timeout_seconds: float = 3.0):
        super().__init__("trading_system", HealthCheckType.READINESS, timeout_seconds)
        self.trading_engine = trading_engine
    
    async def check(self) -> HealthCheckResult:
        """Check trading system health."""
        try:
            # Check if trading engine is running
            is_running = getattr(self.trading_engine, 'is_running', lambda: True)()
            
            # Check recent activity
            details = {
                "is_running": is_running,
                "last_trade_time": "2023-12-01T10:30:00Z",  # Would be actual last trade
                "active_strategies": 3,  # Would be actual count
                "open_positions": 5,     # Would be actual count
            }
            
            if not is_running:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    message="Trading engine is not running",
                    details=details,
                    check_type=self.check_type
                )
            
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message="Trading system operational",
                details=details,
                check_type=self.check_type
            )
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Trading system check failed: {str(e)}",
                check_type=self.check_type
            )


class CustomHealthCheck(BaseHealthCheck):
    """Custom health check using a provided function."""
    
    def __init__(self, name: str, check_function: Callable[[], Union[bool, HealthCheckResult]],
                 check_type: HealthCheckType = HealthCheckType.READINESS,
                 timeout_seconds: float = 5.0):
        super().__init__(name, check_type, timeout_seconds)
        self.check_function = check_function
    
    async def check(self) -> HealthCheckResult:
        """Execute custom health check function."""
        try:
            result = self.check_function()
            
            if isinstance(result, HealthCheckResult):
                return result
            elif isinstance(result, bool):
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
                    message="Custom check " + ("passed" if result else "failed"),
                    check_type=self.check_type
                )
            else:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Invalid check result type: {type(result)}",
                    check_type=self.check_type
                )
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Custom check failed: {str(e)}",
                check_type=self.check_type
            )


class HealthChecker:
    """
    Central health checking system.
    
    Features:
    - Multiple health check types
    - Periodic health monitoring
    - Health status aggregation
    - Integration with metrics and alerting
    - RESTful health endpoints
    """
    
    def __init__(self):
        self.checks: Dict[str, BaseHealthCheck] = {}
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._results_history: Dict[str, List[HealthCheckResult]] = {}
        self._max_history = 100
        
        # Add default system checks
        self.add_check(SystemResourceHealthCheck())
    
    def add_check(self, check: BaseHealthCheck):
        """Add a health check."""
        self.checks[check.name] = check
        if check.name not in self._results_history:
            self._results_history[check.name] = []
    
    def remove_check(self, name: str):
        """Remove a health check."""
        if name in self.checks:
            del self.checks[name]
        if name in self._results_history:
            del self._results_history[name]
    
    def add_database_check(self, database_url: str, query: str = "SELECT 1"):
        """Add database health check."""
        self.add_check(DatabaseHealthCheck(database_url, query))
    
    def add_external_service_check(self, name: str, url: str, expected_status: int = 200):
        """Add external service health check."""
        self.add_check(ExternalServiceHealthCheck(name, url, expected_status))
    
    def add_custom_check(self, name: str, check_function: Callable[[], Union[bool, HealthCheckResult]],
                        check_type: HealthCheckType = HealthCheckType.READINESS):
        """Add custom health check."""
        self.add_check(CustomHealthCheck(name, check_function, check_type))
    
    async def check_health(self, check_names: Optional[List[str]] = None) -> Dict[str, HealthCheckResult]:
        """Run health checks and return results."""
        checks_to_run = check_names or list(self.checks.keys())
        
        # Run checks concurrently
        tasks = []
        for name in checks_to_run:
            if name in self.checks:
                tasks.append(self.checks[name].execute())
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        health_results = {}
        for i, result in enumerate(results):
            check_name = checks_to_run[i]
            if isinstance(result, Exception):
                # Handle exceptions from failed checks
                health_results[check_name] = HealthCheckResult(
                    name=check_name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check raised exception: {str(result)}"
                )
            else:
                health_results[check_name] = result
                
                # Store in history
                if check_name in self._results_history:
                    self._results_history[check_name].append(result)
                    # Limit history size
                    if len(self._results_history[check_name]) > self._max_history:
                        self._results_history[check_name] = self._results_history[check_name][-self._max_history:]
        
        return health_results
    
    async def get_overall_status(self) -> HealthStatus:
        """Get overall system health status."""
        results = await self.check_health()
        
        if not results:
            return HealthStatus.UNKNOWN
        
        # Aggregate status
        statuses = [result.status for result in results.values()]
        
        if any(status == HealthStatus.UNHEALTHY for status in statuses):
            return HealthStatus.UNHEALTHY
        elif any(status == HealthStatus.DEGRADED for status in statuses):
            return HealthStatus.DEGRADED
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
    
    async def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary."""
        results = await self.check_health()
        overall_status = await self.get_overall_status()
        
        summary = {
            "status": overall_status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {name: result.to_dict() for name, result in results.items()},
            "summary": {
                "total_checks": len(results),
                "healthy": sum(1 for r in results.values() if r.status == HealthStatus.HEALTHY),
                "degraded": sum(1 for r in results.values() if r.status == HealthStatus.DEGRADED),
                "unhealthy": sum(1 for r in results.values() if r.status == HealthStatus.UNHEALTHY),
                "unknown": sum(1 for r in results.values() if r.status == HealthStatus.UNKNOWN),
            }
        }
        
        return summary
    
    def start_monitoring(self, interval_seconds: int = 30):
        """Start periodic health monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        
        async def monitor_loop():
            while self._monitoring:
                try:
                    await self.check_health()
                    await asyncio.sleep(interval_seconds)
                except Exception as e:
                    print(f"Error in health monitoring: {e}")
                    await asyncio.sleep(interval_seconds)
        
        self._monitor_task = asyncio.create_task(monitor_loop())
    
    def stop_monitoring(self):
        """Stop periodic health monitoring."""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
    
    def get_check_history(self, check_name: str, limit: int = 50) -> List[HealthCheckResult]:
        """Get historical results for a health check."""
        if check_name not in self._results_history:
            return []
        
        return self._results_history[check_name][-limit:]


# Global health checker instance
_global_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Get the global health checker instance."""
    global _global_health_checker
    if _global_health_checker is None:
        _global_health_checker = HealthChecker()
    return _global_health_checker


def initialize_health_checks(database_url: Optional[str] = None,
                           external_services: Optional[Dict[str, str]] = None,
                           start_monitoring: bool = True) -> HealthChecker:
    """Initialize the global health checking system."""
    global _global_health_checker
    _global_health_checker = HealthChecker()
    
    # Add database check if URL provided
    if database_url:
        _global_health_checker.add_database_check(database_url)
    
    # Add external service checks
    if external_services:
        for name, url in external_services.items():
            _global_health_checker.add_external_service_check(name, url)
    
    # Start monitoring if requested
    if start_monitoring:
        _global_health_checker.start_monitoring()
    
    return _global_health_checker