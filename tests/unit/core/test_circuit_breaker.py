"""
Unit tests for Circuit Breaker Pattern
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Optional

from src.core.architecture import (
    CircuitBreaker,
    CircuitBreakerState,
    CircuitBreakerError,
    CircuitBreakerConfig
)


# Test functions
async def successful_async_function():
    """Async function that always succeeds"""
    await asyncio.sleep(0.01)
    return "success"


async def failing_async_function():
    """Async function that always fails"""
    await asyncio.sleep(0.01)
    raise Exception("Operation failed")


async def flaky_async_function(success_rate: float = 0.5):
    """Async function that fails based on success rate"""
    import random
    await asyncio.sleep(0.01)
    if random.random() < success_rate:
        return "success"
    raise Exception("Random failure")


def successful_sync_function():
    """Sync function that always succeeds"""
    time.sleep(0.01)
    return "success"


def failing_sync_function():
    """Sync function that always fails"""
    time.sleep(0.01)
    raise Exception("Operation failed")


class TestService:
    """Test service with methods to wrap"""
    
    def __init__(self):
        self.call_count = 0
        self.should_fail = False
        self.fail_count = 0
        self.success_count = 0
    
    async def async_operation(self, value: int) -> int:
        """Async operation that can be configured to fail"""
        self.call_count += 1
        await asyncio.sleep(0.01)
        
        if self.should_fail:
            self.fail_count += 1
            raise Exception(f"Service error on call {self.call_count}")
        
        self.success_count += 1
        return value * 2
    
    def sync_operation(self, value: int) -> int:
        """Sync operation that can be configured to fail"""
        self.call_count += 1
        time.sleep(0.01)
        
        if self.should_fail:
            self.fail_count += 1
            raise Exception(f"Service error on call {self.call_count}")
        
        self.success_count += 1
        return value * 2


# Fixtures
@pytest.fixture
def circuit_breaker_config():
    """Create circuit breaker configuration"""
    return CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=1.0,
        expected_exception=Exception,
        success_threshold=2,
        half_open_max_calls=3
    )


@pytest.fixture
def circuit_breaker(circuit_breaker_config):
    """Create circuit breaker instance"""
    return CircuitBreaker(
        name="test_breaker",
        config=circuit_breaker_config
    )


@pytest.fixture
def test_service():
    """Create test service instance"""
    return TestService()


@pytest.mark.unit
class TestCircuitBreakerBasics:
    """Test basic circuit breaker functionality"""
    
    def test_initial_state(self, circuit_breaker):
        """Test initial circuit breaker state"""
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.failure_count == 0
        assert circuit_breaker.success_count == 0
        assert circuit_breaker.last_failure_time is None
    
    @pytest.mark.asyncio
    async def test_successful_calls(self, circuit_breaker):
        """Test successful calls keep circuit closed"""
        for i in range(5):
            result = await circuit_breaker.call(successful_async_function)
            assert result == "success"
        
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.failure_count == 0
        assert circuit_breaker.success_count == 5
    
    @pytest.mark.asyncio
    async def test_failure_threshold(self, circuit_breaker):
        """Test circuit opens after failure threshold"""
        # Make calls that fail
        for i in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_async_function)
        
        # Circuit should be open
        assert circuit_breaker.state == CircuitBreakerState.OPEN
        assert circuit_breaker.failure_count == 3
        
        # Further calls should fail immediately
        with pytest.raises(CircuitBreakerError) as exc_info:
            await circuit_breaker.call(successful_async_function)
        
        assert "Circuit breaker is OPEN" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_half_open_state(self, circuit_breaker):
        """Test transition to half-open state"""
        # Open the circuit
        for i in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_async_function)
        
        assert circuit_breaker.state == CircuitBreakerState.OPEN
        
        # Wait for recovery timeout
        await asyncio.sleep(1.1)
        
        # Next call should transition to half-open
        result = await circuit_breaker.call(successful_async_function)
        assert result == "success"
        assert circuit_breaker.state == CircuitBreakerState.HALF_OPEN
    
    @pytest.mark.asyncio
    async def test_recovery_to_closed(self, circuit_breaker):
        """Test successful recovery from half-open to closed"""
        # Open the circuit
        for i in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_async_function)
        
        # Wait and transition to half-open
        await asyncio.sleep(1.1)
        
        # Make successful calls to close circuit
        for i in range(2):  # success_threshold = 2
            result = await circuit_breaker.call(successful_async_function)
            assert result == "success"
        
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.failure_count == 0
        assert circuit_breaker.success_count == 2
    
    @pytest.mark.asyncio
    async def test_half_open_failure(self, circuit_breaker):
        """Test failure in half-open state returns to open"""
        # Open the circuit
        for i in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_async_function)
        
        # Wait and transition to half-open
        await asyncio.sleep(1.1)
        
        # Fail in half-open state
        with pytest.raises(Exception):
            await circuit_breaker.call(failing_async_function)
        
        # Should return to open state
        assert circuit_breaker.state == CircuitBreakerState.OPEN
        
        # Next call should fail immediately
        with pytest.raises(CircuitBreakerError):
            await circuit_breaker.call(successful_async_function)
    
    def test_sync_function_calls(self, circuit_breaker):
        """Test circuit breaker with sync functions"""
        # Successful calls
        for i in range(3):
            result = circuit_breaker.call_sync(successful_sync_function)
            assert result == "success"
        
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        
        # Failing calls
        for i in range(3):
            with pytest.raises(Exception):
                circuit_breaker.call_sync(failing_sync_function)
        
        assert circuit_breaker.state == CircuitBreakerState.OPEN


@pytest.mark.unit
class TestCircuitBreakerWithService:
    """Test circuit breaker with service methods"""
    
    @pytest.mark.asyncio
    async def test_wrapping_service_method(self, circuit_breaker, test_service):
        """Test wrapping service methods with circuit breaker"""
        # Create wrapped method
        async def protected_operation(value: int) -> int:
            return await circuit_breaker.call(
                test_service.async_operation,
                value
            )
        
        # Successful calls
        result = await protected_operation(5)
        assert result == 10
        assert test_service.call_count == 1
        
        # Make service fail
        test_service.should_fail = True
        
        # Failures open circuit
        for i in range(3):
            with pytest.raises(Exception):
                await protected_operation(5)
        
        assert circuit_breaker.state == CircuitBreakerState.OPEN
        assert test_service.call_count == 4
        
        # Circuit open - no more calls to service
        with pytest.raises(CircuitBreakerError):
            await protected_operation(5)
        
        assert test_service.call_count == 4  # No additional calls
    
    @pytest.mark.asyncio
    async def test_decorator_pattern(self, circuit_breaker_config):
        """Test using circuit breaker as decorator"""
        cb = CircuitBreaker("decorator_test", circuit_breaker_config)
        
        @cb.protect
        async def protected_function(x: int) -> int:
            if x < 0:
                raise ValueError("Negative value")
            return x * 2
        
        # Successful calls
        assert await protected_function(5) == 10
        assert await protected_function(10) == 20
        
        # Failing calls
        for i in range(3):
            with pytest.raises(ValueError):
                await protected_function(-1)
        
        # Circuit should be open
        with pytest.raises(CircuitBreakerError):
            await protected_function(5)
    
    def test_sync_decorator_pattern(self, circuit_breaker_config):
        """Test sync decorator pattern"""
        cb = CircuitBreaker("sync_decorator_test", circuit_breaker_config)
        
        @cb.protect_sync
        def protected_function(x: int) -> int:
            if x < 0:
                raise ValueError("Negative value")
            return x * 2
        
        # Successful calls
        assert protected_function(5) == 10
        
        # Failing calls open circuit
        for i in range(3):
            with pytest.raises(ValueError):
                protected_function(-1)
        
        # Circuit should be open
        with pytest.raises(CircuitBreakerError):
            protected_function(5)


@pytest.mark.unit
class TestCircuitBreakerMetrics:
    """Test circuit breaker metrics and monitoring"""
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self, circuit_breaker):
        """Test metrics are collected correctly"""
        # Make some successful calls
        for i in range(3):
            await circuit_breaker.call(successful_async_function)
        
        # Make some failing calls
        for i in range(2):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_async_function)
        
        metrics = circuit_breaker.get_metrics()
        
        assert metrics["total_calls"] == 5
        assert metrics["successful_calls"] == 3
        assert metrics["failed_calls"] == 2
        assert metrics["success_rate"] == 0.6
        assert metrics["current_state"] == CircuitBreakerState.CLOSED.value
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self, circuit_breaker):
        """Test performance metrics collection"""
        # Make calls with timing
        start_time = time.time()
        
        for i in range(5):
            await circuit_breaker.call(successful_async_function)
        
        metrics = circuit_breaker.get_metrics()
        
        assert "average_response_time" in metrics
        assert metrics["average_response_time"] > 0
        assert "last_call_time" in metrics
    
    @pytest.mark.asyncio
    async def test_state_transitions_tracking(self, circuit_breaker):
        """Test state transition tracking"""
        transitions = []
        
        # Add state change listener
        def on_state_change(old_state: CircuitBreakerState, new_state: CircuitBreakerState):
            transitions.append((old_state, new_state))
        
        circuit_breaker.on_state_change = on_state_change
        
        # Trigger state changes
        for i in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_async_function)
        
        # Should have transitioned from CLOSED to OPEN
        assert len(transitions) == 1
        assert transitions[0] == (CircuitBreakerState.CLOSED, CircuitBreakerState.OPEN)
        
        # Wait and make successful call
        await asyncio.sleep(1.1)
        await circuit_breaker.call(successful_async_function)
        
        # Should have transitioned to HALF_OPEN
        assert len(transitions) == 2
        assert transitions[1] == (CircuitBreakerState.OPEN, CircuitBreakerState.HALF_OPEN)


@pytest.mark.unit
class TestCircuitBreakerConfiguration:
    """Test circuit breaker configuration options"""
    
    @pytest.mark.asyncio
    async def test_custom_exceptions(self):
        """Test handling specific exceptions only"""
        class CustomError(Exception):
            pass
        
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.5,
            expected_exception=CustomError
        )
        cb = CircuitBreaker("custom_exception_test", config)
        
        # Regular exceptions don't count as failures
        for i in range(3):
            with pytest.raises(ValueError):
                await cb.call(lambda: (_ for _ in ()).throw(ValueError("Not counted")))
        
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0
        
        # Custom exceptions count as failures
        for i in range(2):
            with pytest.raises(CustomError):
                await cb.call(lambda: (_ for _ in ()).throw(CustomError("Counted")))
        
        assert cb.state == CircuitBreakerState.OPEN
    
    @pytest.mark.asyncio
    async def test_half_open_max_calls(self):
        """Test limiting calls in half-open state"""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.5,
            half_open_max_calls=2
        )
        cb = CircuitBreaker("half_open_limit_test", config)
        
        # Open the circuit
        for i in range(2):
            with pytest.raises(Exception):
                await cb.call(failing_async_function)
        
        # Wait for recovery
        await asyncio.sleep(0.6)
        
        # Make max allowed calls in half-open
        await cb.call(successful_async_function)
        await cb.call(successful_async_function)
        
        # Further calls should be rejected until state changes
        with pytest.raises(CircuitBreakerError) as exc_info:
            await cb.call(successful_async_function)
        
        assert "Half-open circuit has reached max calls" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test handling of slow operations"""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.5,
            timeout=0.1  # 100ms timeout
        )
        cb = CircuitBreaker("timeout_test", config)
        
        async def slow_operation():
            await asyncio.sleep(0.2)  # Exceeds timeout
            return "success"
        
        # Timeouts count as failures
        for i in range(2):
            with pytest.raises(asyncio.TimeoutError):
                await cb.call(slow_operation)
        
        assert cb.state == CircuitBreakerState.OPEN


@pytest.mark.unit
class TestCircuitBreakerAdvanced:
    """Test advanced circuit breaker scenarios"""
    
    @pytest.mark.asyncio
    async def test_concurrent_calls(self, circuit_breaker):
        """Test circuit breaker under concurrent load"""
        async def make_call(i: int):
            try:
                if i % 4 == 0:
                    return await circuit_breaker.call(failing_async_function)
                else:
                    return await circuit_breaker.call(successful_async_function)
            except Exception:
                return None
        
        # Make concurrent calls
        tasks = [make_call(i) for i in range(20)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Circuit should handle concurrent access correctly
        metrics = circuit_breaker.get_metrics()
        assert metrics["total_calls"] == 20
    
    @pytest.mark.asyncio
    async def test_fallback_mechanism(self, circuit_breaker):
        """Test fallback functionality"""
        async def fallback_function():
            return "fallback_value"
        
        async def protected_call():
            try:
                return await circuit_breaker.call(failing_async_function)
            except (Exception, CircuitBreakerError):
                return await fallback_function()
        
        # Open the circuit
        for i in range(3):
            result = await protected_call()
            assert result == "fallback_value"
        
        # Circuit is open, fallback still works
        result = await protected_call()
        assert result == "fallback_value"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_reset(self, circuit_breaker):
        """Test manual circuit breaker reset"""
        # Open the circuit
        for i in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_async_function)
        
        assert circuit_breaker.state == CircuitBreakerState.OPEN
        
        # Manual reset
        circuit_breaker.reset()
        
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.failure_count == 0
        assert circuit_breaker.success_count == 0
        
        # Can make calls again
        result = await circuit_breaker.call(successful_async_function)
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_gradual_recovery(self):
        """Test gradual recovery with increasing success threshold"""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=0.5,
            success_threshold=5,  # Require more successes to fully close
            half_open_max_calls=10
        )
        cb = CircuitBreaker("gradual_recovery_test", config)
        
        # Open the circuit
        for i in range(3):
            with pytest.raises(Exception):
                await cb.call(failing_async_function)
        
        # Wait for recovery
        await asyncio.sleep(0.6)
        
        # Need 5 successful calls to close
        for i in range(4):
            await cb.call(successful_async_function)
            assert cb.state == CircuitBreakerState.HALF_OPEN
        
        # Fifth call should close the circuit
        await cb.call(successful_async_function)
        assert cb.state == CircuitBreakerState.CLOSED
    
    @pytest.mark.asyncio
    async def test_error_rate_threshold(self):
        """Test error rate based circuit breaking"""
        # Would need to implement error rate tracking
        # This is a placeholder for future enhancement
        pass


@pytest.mark.unit
class TestCircuitBreakerIntegration:
    """Test circuit breaker integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_multiple_circuit_breakers(self):
        """Test multiple circuit breakers for different services"""
        cb1 = CircuitBreaker("service1", CircuitBreakerConfig(failure_threshold=2))
        cb2 = CircuitBreaker("service2", CircuitBreakerConfig(failure_threshold=3))
        
        # Open first circuit breaker
        for i in range(2):
            with pytest.raises(Exception):
                await cb1.call(failing_async_function)
        
        assert cb1.state == CircuitBreakerState.OPEN
        assert cb2.state == CircuitBreakerState.CLOSED
        
        # Second circuit breaker still works
        result = await cb2.call(successful_async_function)
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_cascading_failures(self):
        """Test handling cascading failures"""
        # Service A depends on Service B
        cb_service_a = CircuitBreaker("serviceA", CircuitBreakerConfig(failure_threshold=2))
        cb_service_b = CircuitBreaker("serviceB", CircuitBreakerConfig(failure_threshold=2))
        
        async def service_b_operation():
            raise Exception("Service B failed")
        
        async def service_a_operation():
            try:
                return await cb_service_b.call(service_b_operation)
            except (Exception, CircuitBreakerError):
                raise Exception("Service A failed due to Service B")
        
        # Both circuits should eventually open
        for i in range(4):
            with pytest.raises(Exception):
                await cb_service_a.call(service_a_operation)
        
        assert cb_service_a.state == CircuitBreakerState.OPEN
        assert cb_service_b.state == CircuitBreakerState.OPEN