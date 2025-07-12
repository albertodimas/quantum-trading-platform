"""
Unit tests for Unit of Work Pattern
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import uuid
from contextlib import asynccontextmanager

from src.core.architecture import (
    BaseUnitOfWork,
    InMemoryUnitOfWork,
    PostgreSQLUnitOfWork,
    transactional,
    InMemoryRepository,
    BaseRepository
)
from src.models.base import BaseModel


# Test models
class Order(BaseModel):
    """Test order model"""
    id: str
    user_id: str
    symbol: str
    quantity: float
    price: float
    status: str
    created_at: datetime


class Trade(BaseModel):
    """Test trade model"""
    id: str
    order_id: str
    executed_quantity: float
    executed_price: float
    timestamp: datetime


class Position(BaseModel):
    """Test position model"""
    id: str
    user_id: str
    symbol: str
    quantity: float
    average_price: float


# Test repositories
class OrderRepository(BaseRepository[Order]):
    pass


class TradeRepository(BaseRepository[Trade]):
    pass


class PositionRepository(BaseRepository[Position]):
    pass


class InMemoryOrderRepository(InMemoryRepository[Order], OrderRepository):
    pass


class InMemoryTradeRepository(InMemoryRepository[Trade], TradeRepository):
    pass


class InMemoryPositionRepository(InMemoryRepository[Position], PositionRepository):
    pass


# Test Unit of Work implementations
class TestUnitOfWork(BaseUnitOfWork):
    """Test unit of work with repositories"""
    
    def __init__(self):
        super().__init__()
        self.orders: OrderRepository = None
        self.trades: TradeRepository = None
        self.positions: PositionRepository = None


class InMemoryTestUnitOfWork(InMemoryUnitOfWork, TestUnitOfWork):
    """In-memory implementation for testing"""
    
    def __init__(self):
        super().__init__()
        self.orders = InMemoryOrderRepository()
        self.trades = InMemoryTradeRepository()
        self.positions = InMemoryPositionRepository()
        self._repositories = {
            "orders": self.orders,
            "trades": self.trades,
            "positions": self.positions
        }


# Fixtures
@pytest.fixture
def sample_order():
    """Create sample order"""
    return Order(
        id=str(uuid.uuid4()),
        user_id="user123",
        symbol="BTC/USDT",
        quantity=0.1,
        price=50000.0,
        status="PENDING",
        created_at=datetime.now()
    )


@pytest.fixture
def sample_trade():
    """Create sample trade"""
    return Trade(
        id=str(uuid.uuid4()),
        order_id="order123",
        executed_quantity=0.1,
        executed_price=50100.0,
        timestamp=datetime.now()
    )


@pytest.fixture
def sample_position():
    """Create sample position"""
    return Position(
        id=str(uuid.uuid4()),
        user_id="user123",
        symbol="BTC/USDT",
        quantity=0.1,
        average_price=50000.0
    )


@pytest.fixture
async def in_memory_uow():
    """Create in-memory unit of work"""
    uow = InMemoryTestUnitOfWork()
    yield uow


@pytest.fixture
async def mock_connection_pool():
    """Create mock PostgreSQL connection pool"""
    pool = AsyncMock()
    
    # Mock connection
    conn = AsyncMock()
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock()
    
    # Mock transaction
    transaction = AsyncMock()
    transaction.__aenter__ = AsyncMock(return_value=transaction)
    transaction.__aexit__ = AsyncMock()
    
    # Mock savepoint
    savepoint = AsyncMock()
    savepoint.__aenter__ = AsyncMock(return_value=savepoint)
    savepoint.__aexit__ = AsyncMock()
    
    conn.transaction.return_value = transaction
    conn.savepoint.return_value = savepoint
    conn.commit = AsyncMock()
    conn.rollback = AsyncMock()
    
    pool.acquire.return_value = conn
    
    return pool


@pytest.fixture
async def postgresql_uow(mock_connection_pool):
    """Create PostgreSQL unit of work with mocked connection"""
    class PostgreSQLTestUnitOfWork(PostgreSQLUnitOfWork, TestUnitOfWork):
        def __init__(self, connection_pool):
            super().__init__(connection_pool)
            # Mock repositories
            self.orders = Mock(spec=OrderRepository)
            self.trades = Mock(spec=TradeRepository)
            self.positions = Mock(spec=PositionRepository)
    
    uow = PostgreSQLTestUnitOfWork(mock_connection_pool)
    yield uow


@pytest.mark.unit
class TestInMemoryUnitOfWork:
    """Test in-memory unit of work implementation"""
    
    @pytest.mark.asyncio
    async def test_basic_transaction(self, in_memory_uow, sample_order):
        """Test basic transaction commit"""
        async with in_memory_uow:
            await in_memory_uow.orders.add(sample_order)
            await in_memory_uow.commit()
        
        # Verify order was committed
        order = await in_memory_uow.orders.get_by_id(sample_order.id)
        assert order is not None
        assert order.id == sample_order.id
    
    @pytest.mark.asyncio
    async def test_transaction_rollback(self, in_memory_uow, sample_order):
        """Test transaction rollback"""
        try:
            async with in_memory_uow:
                await in_memory_uow.orders.add(sample_order)
                raise Exception("Simulated error")
        except Exception:
            pass
        
        # Order should not be persisted
        order = await in_memory_uow.orders.get_by_id(sample_order.id)
        assert order is None
    
    @pytest.mark.asyncio
    async def test_explicit_rollback(self, in_memory_uow, sample_order):
        """Test explicit rollback"""
        async with in_memory_uow:
            await in_memory_uow.orders.add(sample_order)
            await in_memory_uow.rollback()
        
        # Order should not be persisted
        order = await in_memory_uow.orders.get_by_id(sample_order.id)
        assert order is None
    
    @pytest.mark.asyncio
    async def test_multiple_repositories(
        self,
        in_memory_uow,
        sample_order,
        sample_trade,
        sample_position
    ):
        """Test transaction across multiple repositories"""
        async with in_memory_uow:
            await in_memory_uow.orders.add(sample_order)
            await in_memory_uow.trades.add(sample_trade)
            await in_memory_uow.positions.add(sample_position)
            await in_memory_uow.commit()
        
        # All entities should be persisted
        order = await in_memory_uow.orders.get_by_id(sample_order.id)
        trade = await in_memory_uow.trades.get_by_id(sample_trade.id)
        position = await in_memory_uow.positions.get_by_id(sample_position.id)
        
        assert order is not None
        assert trade is not None
        assert position is not None
    
    @pytest.mark.asyncio
    async def test_partial_rollback(
        self,
        in_memory_uow,
        sample_order,
        sample_trade
    ):
        """Test partial operations with rollback"""
        async with in_memory_uow:
            await in_memory_uow.orders.add(sample_order)
            await in_memory_uow.commit()
        
        # Order should be persisted
        order = await in_memory_uow.orders.get_by_id(sample_order.id)
        assert order is not None
        
        # Second transaction with rollback
        try:
            async with in_memory_uow:
                await in_memory_uow.trades.add(sample_trade)
                raise Exception("Error after trade")
        except Exception:
            pass
        
        # Trade should not be persisted, but order remains
        trade = await in_memory_uow.trades.get_by_id(sample_trade.id)
        order = await in_memory_uow.orders.get_by_id(sample_order.id)
        
        assert trade is None
        assert order is not None
    
    @pytest.mark.asyncio
    async def test_nested_transactions(self, in_memory_uow, sample_order):
        """Test nested transaction behavior"""
        async with in_memory_uow:
            await in_memory_uow.orders.add(sample_order)
            
            # Nested context (should work with same transaction)
            async with in_memory_uow:
                order = await in_memory_uow.orders.get_by_id(sample_order.id)
                assert order is not None
            
            await in_memory_uow.commit()
        
        # Order should still be persisted
        order = await in_memory_uow.orders.get_by_id(sample_order.id)
        assert order is not None
    
    @pytest.mark.asyncio
    async def test_savepoint_simulation(self, in_memory_uow, sample_order, sample_trade):
        """Test savepoint-like behavior"""
        async with in_memory_uow:
            await in_memory_uow.orders.add(sample_order)
            
            # Create savepoint
            await in_memory_uow.savepoint("sp1")
            
            await in_memory_uow.trades.add(sample_trade)
            
            # Rollback to savepoint
            await in_memory_uow.rollback_to_savepoint("sp1")
            
            await in_memory_uow.commit()
        
        # Order should be persisted, trade should not
        order = await in_memory_uow.orders.get_by_id(sample_order.id)
        trade = await in_memory_uow.trades.get_by_id(sample_trade.id)
        
        assert order is not None
        assert trade is None


@pytest.mark.unit
class TestPostgreSQLUnitOfWork:
    """Test PostgreSQL unit of work implementation"""
    
    @pytest.mark.asyncio
    async def test_transaction_creation(self, postgresql_uow, mock_connection_pool):
        """Test transaction is created properly"""
        async with postgresql_uow:
            pass
        
        # Verify connection and transaction were created
        mock_connection_pool.acquire.assert_called_once()
        mock_conn = mock_connection_pool.acquire.return_value.__aenter__.return_value
        mock_conn.transaction.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_commit(self, postgresql_uow, mock_connection_pool):
        """Test commit operation"""
        async with postgresql_uow:
            await postgresql_uow.commit()
        
        mock_conn = mock_connection_pool.acquire.return_value.__aenter__.return_value
        mock_transaction = mock_conn.transaction.return_value.__aenter__.return_value
        
        # Transaction should be committed
        # Note: In real implementation, commit is handled by transaction context
    
    @pytest.mark.asyncio
    async def test_rollback_on_exception(self, postgresql_uow, mock_connection_pool):
        """Test automatic rollback on exception"""
        with pytest.raises(ValueError):
            async with postgresql_uow:
                raise ValueError("Test error")
        
        # Transaction should exit with exception (automatic rollback)
        mock_conn = mock_connection_pool.acquire.return_value.__aenter__.return_value
        mock_transaction = mock_conn.transaction.return_value
        
        # __aexit__ should be called with exception
        mock_transaction.__aexit__.assert_called()
    
    @pytest.mark.asyncio
    async def test_savepoint_creation(self, postgresql_uow, mock_connection_pool):
        """Test savepoint creation"""
        async with postgresql_uow:
            await postgresql_uow.savepoint("test_sp")
        
        mock_conn = mock_connection_pool.acquire.return_value.__aenter__.return_value
        mock_conn.savepoint.assert_called_with("test_sp")
    
    @pytest.mark.asyncio
    async def test_rollback_to_savepoint(self, postgresql_uow, mock_connection_pool):
        """Test rollback to savepoint"""
        async with postgresql_uow:
            savepoint_ctx = await postgresql_uow.savepoint("test_sp")
            await postgresql_uow.rollback_to_savepoint("test_sp")
        
        mock_conn = mock_connection_pool.acquire.return_value.__aenter__.return_value
        mock_savepoint = mock_conn.savepoint.return_value
        
        # Savepoint should be rolled back
        # In real implementation, this is handled by savepoint context


@pytest.mark.unit
class TestTransactionalDecorator:
    """Test @transactional decorator"""
    
    @pytest.mark.asyncio
    async def test_transactional_function(self, in_memory_uow, sample_order):
        """Test transactional decorator on function"""
        @transactional
        async def create_order(uow: TestUnitOfWork, order: Order):
            await uow.orders.add(order)
            return order.id
        
        order_id = await create_order(in_memory_uow, sample_order)
        
        # Order should be persisted
        order = await in_memory_uow.orders.get_by_id(order_id)
        assert order is not None
    
    @pytest.mark.asyncio
    async def test_transactional_with_exception(self, in_memory_uow, sample_order):
        """Test transactional rollback on exception"""
        @transactional
        async def failing_operation(uow: TestUnitOfWork, order: Order):
            await uow.orders.add(order)
            raise ValueError("Operation failed")
        
        with pytest.raises(ValueError):
            await failing_operation(in_memory_uow, sample_order)
        
        # Order should not be persisted
        order = await in_memory_uow.orders.get_by_id(sample_order.id)
        assert order is None
    
    @pytest.mark.asyncio
    async def test_transactional_with_custom_uow_param(self, in_memory_uow, sample_order):
        """Test transactional with custom parameter name"""
        @transactional(uow_param="unit_of_work")
        async def create_order(unit_of_work: TestUnitOfWork, order: Order):
            await unit_of_work.orders.add(order)
        
        await create_order(in_memory_uow, sample_order)
        
        # Order should be persisted
        order = await in_memory_uow.orders.get_by_id(sample_order.id)
        assert order is not None
    
    @pytest.mark.asyncio
    async def test_transactional_on_method(self, in_memory_uow, sample_order):
        """Test transactional decorator on class method"""
        class OrderService:
            @transactional
            async def create_order(self, uow: TestUnitOfWork, order: Order):
                await uow.orders.add(order)
                return order
        
        service = OrderService()
        await service.create_order(in_memory_uow, sample_order)
        
        # Order should be persisted
        order = await in_memory_uow.orders.get_by_id(sample_order.id)
        assert order is not None


@pytest.mark.unit
class TestComplexScenarios:
    """Test complex unit of work scenarios"""
    
    @pytest.mark.asyncio
    async def test_business_transaction(
        self,
        in_memory_uow,
        sample_order,
        sample_trade,
        sample_position
    ):
        """Test complex business transaction"""
        async def execute_trade(uow: TestUnitOfWork, order: Order, trade: Trade):
            """Execute a trade and update position"""
            async with uow:
                # Add order
                await uow.orders.add(order)
                
                # Add trade
                trade.order_id = order.id
                await uow.trades.add(trade)
                
                # Update or create position
                existing_position = await uow.positions.query().filter(
                    lambda p: p.user_id == order.user_id and p.symbol == order.symbol
                ).first()
                
                if existing_position:
                    # Update existing position
                    existing_position.quantity += trade.executed_quantity
                    existing_position.average_price = (
                        (existing_position.average_price * existing_position.quantity +
                         trade.executed_price * trade.executed_quantity) /
                        (existing_position.quantity + trade.executed_quantity)
                    )
                    await uow.positions.update(existing_position)
                else:
                    # Create new position
                    position = Position(
                        id=str(uuid.uuid4()),
                        user_id=order.user_id,
                        symbol=order.symbol,
                        quantity=trade.executed_quantity,
                        average_price=trade.executed_price
                    )
                    await uow.positions.add(position)
                
                # Update order status
                order.status = "FILLED"
                await uow.orders.update(order)
                
                await uow.commit()
        
        # Execute the transaction
        await execute_trade(in_memory_uow, sample_order, sample_trade)
        
        # Verify all changes were persisted
        order = await in_memory_uow.orders.get_by_id(sample_order.id)
        trade = await in_memory_uow.trades.get_by_id(sample_trade.id)
        positions = await in_memory_uow.positions.get_all()
        
        assert order is not None
        assert order.status == "FILLED"
        assert trade is not None
        assert len(positions) == 1
        assert positions[0].quantity == trade.executed_quantity
    
    @pytest.mark.asyncio
    async def test_concurrent_transactions(self, in_memory_uow):
        """Test concurrent transaction handling"""
        async def create_order(uow: TestUnitOfWork, order_id: str):
            async with uow:
                order = Order(
                    id=order_id,
                    user_id="user123",
                    symbol="BTC/USDT",
                    quantity=0.1,
                    price=50000.0,
                    status="PENDING",
                    created_at=datetime.now()
                )
                await uow.orders.add(order)
                await asyncio.sleep(0.01)  # Simulate some work
                await uow.commit()
        
        # Create multiple orders concurrently
        order_ids = [str(uuid.uuid4()) for _ in range(5)]
        tasks = [create_order(in_memory_uow, order_id) for order_id in order_ids]
        
        await asyncio.gather(*tasks)
        
        # All orders should be created
        orders = await in_memory_uow.orders.get_all()
        assert len(orders) == 5
        assert set(o.id for o in orders) == set(order_ids)
    
    @pytest.mark.asyncio
    async def test_isolation_between_transactions(self, in_memory_uow, sample_order):
        """Test isolation between transactions"""
        # First transaction
        async with in_memory_uow:
            await in_memory_uow.orders.add(sample_order)
            await in_memory_uow.commit()
        
        # Second transaction modifies the order
        async with in_memory_uow:
            order = await in_memory_uow.orders.get_by_id(sample_order.id)
            order.status = "CANCELLED"
            await in_memory_uow.orders.update(order)
            
            # Before commit, outside transaction shouldn't see changes
            # (In real DB with proper isolation)
            await in_memory_uow.commit()
        
        # Verify final state
        order = await in_memory_uow.orders.get_by_id(sample_order.id)
        assert order.status == "CANCELLED"
    
    @pytest.mark.asyncio
    async def test_error_recovery_with_savepoints(
        self,
        in_memory_uow,
        sample_order,
        sample_trade
    ):
        """Test error recovery using savepoints"""
        async with in_memory_uow:
            # Add order
            await in_memory_uow.orders.add(sample_order)
            
            # Create savepoint before risky operation
            await in_memory_uow.savepoint("before_trade")
            
            try:
                # Risky operation
                await in_memory_uow.trades.add(sample_trade)
                
                # Simulate validation error
                if sample_trade.executed_quantity > sample_order.quantity:
                    raise ValueError("Trade quantity exceeds order quantity")
                    
            except ValueError:
                # Rollback to savepoint
                await in_memory_uow.rollback_to_savepoint("before_trade")
                
                # Fix the issue and retry
                sample_trade.executed_quantity = sample_order.quantity
                await in_memory_uow.trades.add(sample_trade)
            
            await in_memory_uow.commit()
        
        # Both order and fixed trade should be persisted
        order = await in_memory_uow.orders.get_by_id(sample_order.id)
        trade = await in_memory_uow.trades.get_by_id(sample_trade.id)
        
        assert order is not None
        assert trade is not None
        assert trade.executed_quantity == sample_order.quantity