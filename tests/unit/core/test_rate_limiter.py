"""
Unit tests for Rate Limiter Pattern
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import threading
from concurrent.futures import ThreadPoolExecutor

from src.core.architecture import (
    RateLimiter,
    TokenBucketLimiter,
    SlidingWindowLimiter,
    FixedWindowLimiter,
    LeakyBucketLimiter,
    RateLimiterStrategy,
    RateLimiterRegistry,
    RateLimitExceeded,
    rate_limit
)


# Test functions
async def test_async_function():
    """Simple async test function"""
    await asyncio.sleep(0.01)
    return "success"


def test_sync_function():
    """Simple sync test function"""
    time.sleep(0.01)
    return "success"


class TestService:
    """Test service with rate limited methods"""
    
    def __init__(self):
        self.call_count = 0
        self.last_call_time = None
    
    async def async_operation(self) -> str:
        """Async operation to be rate limited"""
        self.call_count += 1
        self.last_call_time = datetime.now()
        await asyncio.sleep(0.01)
        return f"call_{self.call_count}"
    
    def sync_operation(self) -> str:
        """Sync operation to be rate limited"""
        self.call_count += 1
        self.last_call_time = datetime.now()
        time.sleep(0.01)
        return f"call_{self.call_count}"


# Fixtures
@pytest.fixture
def token_bucket_limiter():
    """Create token bucket rate limiter"""
    return TokenBucketLimiter(
        capacity=10,
        refill_rate=5,  # 5 tokens per second
        name="test_token_bucket"
    )


@pytest.fixture
def sliding_window_limiter():
    """Create sliding window rate limiter"""
    return SlidingWindowLimiter(
        max_requests=10,
        window_size=1.0,  # 1 second window
        name="test_sliding_window"
    )


@pytest.fixture
def fixed_window_limiter():
    """Create fixed window rate limiter"""
    return FixedWindowLimiter(
        max_requests=10,
        window_size=1.0,  # 1 second window
        name="test_fixed_window"
    )


@pytest.fixture
def leaky_bucket_limiter():
    """Create leaky bucket rate limiter"""
    return LeakyBucketLimiter(
        capacity=10,
        leak_rate=5,  # 5 requests per second
        name="test_leaky_bucket"
    )


@pytest.fixture
def rate_limiter_registry():
    """Create rate limiter registry"""
    return RateLimiterRegistry()


@pytest.fixture
def test_service():
    """Create test service instance"""
    return TestService()


@pytest.mark.unit
class TestTokenBucketLimiter:
    """Test token bucket rate limiter"""
    
    @pytest.mark.asyncio
    async def test_basic_limiting(self, token_bucket_limiter):
        """Test basic rate limiting functionality"""
        # Should allow initial burst up to capacity
        for i in range(10):
            allowed = await token_bucket_limiter.allow()
            assert allowed is True
        
        # 11th request should be denied
        allowed = await token_bucket_limiter.allow()
        assert allowed is False
    
    @pytest.mark.asyncio
    async def test_token_refill(self, token_bucket_limiter):
        """Test token refill mechanism"""
        # Consume all tokens
        for i in range(10):
            await token_bucket_limiter.allow()
        
        # Should be denied
        assert await token_bucket_limiter.allow() is False
        
        # Wait for refill (5 tokens per second)
        await asyncio.sleep(0.4)  # Should refill ~2 tokens
        
        # Should allow 2 requests
        assert await token_bucket_limiter.allow() is True
        assert await token_bucket_limiter.allow() is True
        assert await token_bucket_limiter.allow() is False
    
    @pytest.mark.asyncio
    async def test_acquire_blocking(self, token_bucket_limiter):
        """Test blocking acquire"""
        # Consume all tokens
        for i in range(10):
            await token_bucket_limiter.allow()
        
        # Acquire with timeout should wait for token
        start_time = time.time()
        acquired = await token_bucket_limiter.acquire(timeout=0.3)
        elapsed = time.time() - start_time
        
        assert acquired is True
        assert elapsed >= 0.2  # Should wait for token refill
    
    @pytest.mark.asyncio
    async def test_acquire_timeout(self, token_bucket_limiter):
        """Test acquire timeout"""
        # Consume all tokens
        for i in range(10):
            await token_bucket_limiter.allow()
        
        # Acquire with short timeout should fail
        acquired = await token_bucket_limiter.acquire(timeout=0.1)
        assert acquired is False
    
    @pytest.mark.asyncio
    async def test_burst_capacity(self):
        """Test burst capacity configuration"""
        limiter = TokenBucketLimiter(
            capacity=20,
            refill_rate=1,
            name="burst_test"
        )
        
        # Should allow burst up to capacity
        success_count = 0
        for i in range(25):
            if await limiter.allow():
                success_count += 1
        
        assert success_count == 20
    
    @pytest.mark.asyncio
    async def test_consume_multiple_tokens(self):
        """Test consuming multiple tokens at once"""
        limiter = TokenBucketLimiter(
            capacity=10,
            refill_rate=5,
            name="multi_token_test"
        )
        
        # Consume 5 tokens
        allowed = await limiter.allow(tokens=5)
        assert allowed is True
        
        # Should have 5 tokens left
        for i in range(5):
            assert await limiter.allow() is True
        
        # 6th request should fail
        assert await limiter.allow() is False


@pytest.mark.unit
class TestSlidingWindowLimiter:
    """Test sliding window rate limiter"""
    
    @pytest.mark.asyncio
    async def test_basic_limiting(self, sliding_window_limiter):
        """Test basic rate limiting"""
        # Should allow up to max_requests
        for i in range(10):
            allowed = await sliding_window_limiter.allow()
            assert allowed is True
        
        # 11th request should be denied
        allowed = await sliding_window_limiter.allow()
        assert allowed is False
    
    @pytest.mark.asyncio
    async def test_sliding_window(self, sliding_window_limiter):
        """Test sliding window behavior"""
        # Make 5 requests
        for i in range(5):
            await sliding_window_limiter.allow()
        
        # Wait 0.5 seconds
        await asyncio.sleep(0.5)
        
        # Make 5 more requests
        for i in range(5):
            assert await sliding_window_limiter.allow() is True
        
        # 11th request should fail (still within window)
        assert await sliding_window_limiter.allow() is False
        
        # Wait for first requests to expire
        await asyncio.sleep(0.6)
        
        # Should allow more requests
        assert await sliding_window_limiter.allow() is True
    
    @pytest.mark.asyncio
    async def test_request_expiration(self, sliding_window_limiter):
        """Test request expiration"""
        # Make requests at different times
        await sliding_window_limiter.allow()
        await asyncio.sleep(0.3)
        await sliding_window_limiter.allow()
        await asyncio.sleep(0.3)
        await sliding_window_limiter.allow()
        
        # Check request count over time
        await asyncio.sleep(0.5)  # First request should expire
        
        # Should be able to make more requests
        for i in range(8):
            assert await sliding_window_limiter.allow() is True
    
    @pytest.mark.asyncio
    async def test_precise_timing(self):
        """Test precise timing of sliding window"""
        limiter = SlidingWindowLimiter(
            max_requests=5,
            window_size=1.0,
            name="precise_timing_test"
        )
        
        # Make requests with timestamps
        timestamps = []
        for i in range(5):
            if await limiter.allow():
                timestamps.append(datetime.now())
            await asyncio.sleep(0.1)
        
        # Should not allow more
        assert await limiter.allow() is False
        
        # Wait for exactly 1 second from first request
        await asyncio.sleep(0.6)
        
        # Should allow one more
        assert await limiter.allow() is True


@pytest.mark.unit
class TestFixedWindowLimiter:
    """Test fixed window rate limiter"""
    
    @pytest.mark.asyncio
    async def test_basic_limiting(self, fixed_window_limiter):
        """Test basic rate limiting"""
        # Should allow up to max_requests
        for i in range(10):
            allowed = await fixed_window_limiter.allow()
            assert allowed is True
        
        # 11th request should be denied
        allowed = await fixed_window_limiter.allow()
        assert allowed is False
    
    @pytest.mark.asyncio
    async def test_window_reset(self, fixed_window_limiter):
        """Test window reset behavior"""
        # Fill the window
        for i in range(10):
            await fixed_window_limiter.allow()
        
        # Should be blocked
        assert await fixed_window_limiter.allow() is False
        
        # Wait for window to reset
        await asyncio.sleep(1.1)
        
        # Should allow requests again
        for i in range(10):
            assert await fixed_window_limiter.allow() is True
    
    @pytest.mark.asyncio
    async def test_partial_window(self, fixed_window_limiter):
        """Test behavior at window boundaries"""
        # Make 5 requests
        for i in range(5):
            await fixed_window_limiter.allow()
        
        # Wait for partial window
        await asyncio.sleep(0.5)
        
        # Should still allow 5 more in same window
        for i in range(5):
            assert await fixed_window_limiter.allow() is True
        
        # But not more
        assert await fixed_window_limiter.allow() is False
    
    @pytest.mark.asyncio
    async def test_burst_at_window_start(self):
        """Test burst behavior at window start"""
        limiter = FixedWindowLimiter(
            max_requests=100,
            window_size=1.0,
            name="burst_test"
        )
        
        # Should allow full burst at window start
        success_count = 0
        start_time = time.time()
        
        for i in range(100):
            if await limiter.allow():
                success_count += 1
        
        elapsed = time.time() - start_time
        
        assert success_count == 100
        assert elapsed < 0.1  # Should be very fast


@pytest.mark.unit
class TestLeakyBucketLimiter:
    """Test leaky bucket rate limiter"""
    
    @pytest.mark.asyncio
    async def test_basic_limiting(self, leaky_bucket_limiter):
        """Test basic rate limiting"""
        # Should queue up to capacity
        for i in range(10):
            allowed = await leaky_bucket_limiter.allow()
            assert allowed is True
        
        # 11th request should be denied (bucket full)
        allowed = await leaky_bucket_limiter.allow()
        assert allowed is False
    
    @pytest.mark.asyncio
    async def test_leak_rate(self, leaky_bucket_limiter):
        """Test leak rate behavior"""
        # Fill the bucket
        for i in range(10):
            await leaky_bucket_limiter.allow()
        
        # Bucket is full
        assert await leaky_bucket_limiter.allow() is False
        
        # Wait for some requests to leak (5 per second)
        await asyncio.sleep(0.4)  # Should leak ~2 requests
        
        # Should allow 2 more requests
        assert await leaky_bucket_limiter.allow() is True
        assert await leaky_bucket_limiter.allow() is True
        assert await leaky_bucket_limiter.allow() is False
    
    @pytest.mark.asyncio
    async def test_smooth_rate(self):
        """Test smooth rate limiting"""
        limiter = LeakyBucketLimiter(
            capacity=10,
            leak_rate=10,  # 10 requests per second
            name="smooth_rate_test"
        )
        
        # Should process requests at smooth rate
        request_times = []
        
        for i in range(5):
            if await limiter.allow():
                request_times.append(time.time())
            await asyncio.sleep(0.08)  # Slightly faster than leak rate
        
        # Check spacing between requests
        if len(request_times) > 1:
            intervals = [request_times[i+1] - request_times[i] 
                        for i in range(len(request_times)-1)]
            avg_interval = sum(intervals) / len(intervals)
            
            # Should be close to 0.1 seconds (10 req/sec)
            assert 0.05 < avg_interval < 0.15


@pytest.mark.unit
class TestRateLimiterDecorator:
    """Test rate limiter decorator functionality"""
    
    @pytest.mark.asyncio
    async def test_async_decorator(self):
        """Test rate limiting async functions with decorator"""
        limiter = TokenBucketLimiter(capacity=3, refill_rate=1)
        
        @rate_limit(limiter)
        async def limited_function(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 2
        
        # Should allow first 3 calls
        assert await limited_function(1) == 2
        assert await limited_function(2) == 4
        assert await limited_function(3) == 6
        
        # 4th call should raise exception
        with pytest.raises(RateLimitExceeded):
            await limited_function(4)
    
    def test_sync_decorator(self):
        """Test rate limiting sync functions with decorator"""
        limiter = TokenBucketLimiter(capacity=2, refill_rate=1)
        
        @rate_limit(limiter, is_async=False)
        def limited_function(x: int) -> int:
            time.sleep(0.01)
            return x * 2
        
        # Should allow first 2 calls
        assert limited_function(1) == 2
        assert limited_function(2) == 4
        
        # 3rd call should raise exception
        with pytest.raises(RateLimitExceeded):
            limited_function(3)
    
    @pytest.mark.asyncio
    async def test_decorator_on_method(self, test_service):
        """Test decorator on class methods"""
        limiter = TokenBucketLimiter(capacity=2, refill_rate=1)
        
        # Decorate method
        test_service.async_operation = rate_limit(limiter)(test_service.async_operation)
        
        # Should work for first 2 calls
        result1 = await test_service.async_operation()
        result2 = await test_service.async_operation()
        
        assert result1 == "call_1"
        assert result2 == "call_2"
        
        # 3rd call should fail
        with pytest.raises(RateLimitExceeded):
            await test_service.async_operation()


@pytest.mark.unit
class TestRateLimiterRegistry:
    """Test rate limiter registry"""
    
    def test_register_and_get(self, rate_limiter_registry):
        """Test registering and retrieving limiters"""
        limiter1 = TokenBucketLimiter(capacity=10, refill_rate=5)
        limiter2 = SlidingWindowLimiter(max_requests=20, window_size=1.0)
        
        rate_limiter_registry.register("api_limiter", limiter1)
        rate_limiter_registry.register("db_limiter", limiter2)
        
        assert rate_limiter_registry.get("api_limiter") == limiter1
        assert rate_limiter_registry.get("db_limiter") == limiter2
        assert rate_limiter_registry.get("unknown") is None
    
    def test_create_limiter(self, rate_limiter_registry):
        """Test creating limiters through registry"""
        limiter = rate_limiter_registry.create_limiter(
            name="test_limiter",
            strategy=RateLimiterStrategy.TOKEN_BUCKET,
            capacity=50,
            refill_rate=10
        )
        
        assert isinstance(limiter, TokenBucketLimiter)
        assert rate_limiter_registry.get("test_limiter") == limiter
    
    def test_get_or_create(self, rate_limiter_registry):
        """Test get_or_create functionality"""
        # First call creates
        limiter1 = rate_limiter_registry.get_or_create(
            name="shared_limiter",
            strategy=RateLimiterStrategy.FIXED_WINDOW,
            max_requests=100,
            window_size=60
        )
        
        # Second call returns existing
        limiter2 = rate_limiter_registry.get_or_create(
            name="shared_limiter",
            strategy=RateLimiterStrategy.FIXED_WINDOW,
            max_requests=200,  # Different config ignored
            window_size=120
        )
        
        assert limiter1 is limiter2
    
    @pytest.mark.asyncio
    async def test_apply_limits(self, rate_limiter_registry):
        """Test applying multiple limiters"""
        # Register multiple limiters
        rate_limiter_registry.create_limiter(
            "limiter1",
            RateLimiterStrategy.TOKEN_BUCKET,
            capacity=5,
            refill_rate=5
        )
        rate_limiter_registry.create_limiter(
            "limiter2",
            RateLimiterStrategy.TOKEN_BUCKET,
            capacity=3,
            refill_rate=3
        )
        
        # Apply both limiters
        for i in range(3):
            allowed = await rate_limiter_registry.apply_limits(
                ["limiter1", "limiter2"]
            )
            assert allowed is True
        
        # 4th request should fail on limiter2
        allowed = await rate_limiter_registry.apply_limits(
            ["limiter1", "limiter2"]
        )
        assert allowed is False


@pytest.mark.unit
class TestConcurrency:
    """Test rate limiter concurrency handling"""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, token_bucket_limiter):
        """Test handling concurrent requests"""
        async def make_request():
            return await token_bucket_limiter.allow()
        
        # Make 20 concurrent requests (capacity is 10)
        tasks = [make_request() for _ in range(20)]
        results = await asyncio.gather(*tasks)
        
        # Exactly 10 should succeed
        success_count = sum(1 for r in results if r)
        assert success_count == 10
    
    def test_thread_safety(self):
        """Test thread safety of rate limiters"""
        limiter = TokenBucketLimiter(capacity=100, refill_rate=10)
        success_count = 0
        lock = threading.Lock()
        
        def make_requests():
            nonlocal success_count
            for _ in range(20):
                # Use sync allow for thread safety test
                if limiter.allow_sync():
                    with lock:
                        success_count += 1
                time.sleep(0.001)
        
        # Run from multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_requests)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should not exceed capacity
        assert success_count <= 100
    
    @pytest.mark.asyncio
    async def test_acquire_queue(self, token_bucket_limiter):
        """Test multiple acquires queuing properly"""
        # Exhaust tokens
        for _ in range(10):
            await token_bucket_limiter.allow()
        
        # Multiple acquires should queue
        acquire_tasks = [
            token_bucket_limiter.acquire(timeout=2.0)
            for _ in range(5)
        ]
        
        # Start refilling
        start_time = time.time()
        results = await asyncio.gather(*acquire_tasks)
        elapsed = time.time() - start_time
        
        # All should eventually succeed (5 tokens at 5/sec = 1 second)
        assert all(results)
        assert 0.8 < elapsed < 1.5


@pytest.mark.unit
class TestMetricsAndMonitoring:
    """Test rate limiter metrics and monitoring"""
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self, token_bucket_limiter):
        """Test metrics are collected correctly"""
        # Make some requests
        for i in range(15):
            await token_bucket_limiter.allow()
        
        metrics = token_bucket_limiter.get_metrics()
        
        assert metrics["total_requests"] == 15
        assert metrics["allowed_requests"] == 10
        assert metrics["rejected_requests"] == 5
        assert metrics["rejection_rate"] == 5/15
    
    @pytest.mark.asyncio
    async def test_reset_metrics(self, sliding_window_limiter):
        """Test resetting metrics"""
        # Generate some metrics
        for i in range(5):
            await sliding_window_limiter.allow()
        
        # Reset
        sliding_window_limiter.reset_metrics()
        metrics = sliding_window_limiter.get_metrics()
        
        assert metrics["total_requests"] == 0
        assert metrics["allowed_requests"] == 0
        assert metrics["rejected_requests"] == 0


@pytest.mark.unit
class TestAdaptiveRateLimiting:
    """Test adaptive rate limiting features"""
    
    @pytest.mark.asyncio
    async def test_dynamic_rate_adjustment(self):
        """Test dynamically adjusting rate limits"""
        limiter = TokenBucketLimiter(capacity=10, refill_rate=5)
        
        # Use initial capacity
        for i in range(10):
            assert await limiter.allow() is True
        assert await limiter.allow() is False
        
        # Adjust capacity
        limiter.capacity = 20
        limiter._tokens = 20  # Reset tokens
        
        # Should allow 20 now
        success_count = 0
        for i in range(25):
            if await limiter.allow():
                success_count += 1
        
        assert success_count == 20
    
    @pytest.mark.asyncio
    async def test_backpressure_handling(self):
        """Test handling backpressure scenarios"""
        limiter = LeakyBucketLimiter(
            capacity=5,
            leak_rate=2,  # Slow leak rate
        )
        
        # Fill bucket quickly
        fills = 0
        for i in range(10):
            if await limiter.allow():
                fills += 1
        
        assert fills == 5  # Bucket capacity
        
        # Monitor queue depth
        metrics = limiter.get_metrics()
        assert metrics["queue_depth"] == 5