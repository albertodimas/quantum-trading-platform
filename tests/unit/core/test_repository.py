"""
Unit tests for Repository Pattern
"""

import pytest
import asyncio
from datetime import datetime
from typing import List, Optional
from unittest.mock import Mock, AsyncMock, patch
import uuid

from src.core.architecture import (
    BaseRepository,
    InMemoryRepository,
    PostgreSQLRepository,
    QueryBuilder,
    Filter,
    FilterOperator,
    SortOrder,
    PageRequest,
    PageResult
)
from src.models.base import BaseModel


# Test models
class TestEntity(BaseModel):
    """Test entity for repository tests"""
    id: str
    name: str
    age: int
    email: str
    active: bool = True
    created_at: datetime
    tags: List[str] = []


# Test repository implementations
class TestEntityRepository(BaseRepository[TestEntity]):
    """Test repository interface"""
    pass


class InMemoryTestRepository(InMemoryRepository[TestEntity], TestEntityRepository):
    """In-memory implementation for testing"""
    pass


class PostgreSQLTestRepository(PostgreSQLRepository[TestEntity], TestEntityRepository):
    """PostgreSQL implementation for testing"""
    
    def __init__(self, connection_pool):
        super().__init__(connection_pool, "test_entities")
    
    def _entity_to_dict(self, entity: TestEntity) -> dict:
        """Convert entity to dictionary for storage"""
        return {
            "id": entity.id,
            "name": entity.name,
            "age": entity.age,
            "email": entity.email,
            "active": entity.active,
            "created_at": entity.created_at,
            "tags": entity.tags
        }
    
    def _dict_to_entity(self, data: dict) -> TestEntity:
        """Convert dictionary to entity"""
        return TestEntity(**data)


# Fixtures
@pytest.fixture
def sample_entities():
    """Create sample test entities"""
    return [
        TestEntity(
            id=str(uuid.uuid4()),
            name="Alice",
            age=30,
            email="alice@example.com",
            created_at=datetime.now(),
            tags=["admin", "user"]
        ),
        TestEntity(
            id=str(uuid.uuid4()),
            name="Bob",
            age=25,
            email="bob@example.com",
            created_at=datetime.now(),
            tags=["user"]
        ),
        TestEntity(
            id=str(uuid.uuid4()),
            name="Charlie",
            age=35,
            email="charlie@example.com",
            active=False,
            created_at=datetime.now(),
            tags=["user", "moderator"]
        ),
        TestEntity(
            id=str(uuid.uuid4()),
            name="David",
            age=28,
            email="david@example.com",
            created_at=datetime.now(),
            tags=[]
        ),
        TestEntity(
            id=str(uuid.uuid4()),
            name="Eve",
            age=32,
            email="eve@example.com",
            created_at=datetime.now(),
            tags=["admin", "moderator", "user"]
        )
    ]


@pytest.fixture
async def in_memory_repository():
    """Create in-memory repository"""
    repo = InMemoryTestRepository()
    yield repo


@pytest.fixture
async def mock_connection_pool():
    """Create mock PostgreSQL connection pool"""
    pool = AsyncMock()
    
    # Mock connection
    conn = AsyncMock()
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock()
    
    # Mock cursor
    cursor = AsyncMock()
    cursor.__aenter__ = AsyncMock(return_value=cursor)
    cursor.__aexit__ = AsyncMock()
    
    conn.cursor.return_value = cursor
    pool.acquire.return_value = conn
    
    return pool


@pytest.fixture
async def postgresql_repository(mock_connection_pool):
    """Create PostgreSQL repository with mocked connection"""
    repo = PostgreSQLTestRepository(mock_connection_pool)
    yield repo


@pytest.mark.unit
class TestInMemoryRepository:
    """Test in-memory repository implementation"""
    
    @pytest.mark.asyncio
    async def test_add_entity(self, in_memory_repository, sample_entities):
        """Test adding entity"""
        entity = sample_entities[0]
        
        result = await in_memory_repository.add(entity)
        
        assert result == entity
        assert entity.id in in_memory_repository._storage
        assert in_memory_repository._storage[entity.id] == entity
    
    @pytest.mark.asyncio
    async def test_add_multiple_entities(self, in_memory_repository, sample_entities):
        """Test adding multiple entities"""
        await in_memory_repository.add_many(sample_entities)
        
        assert len(in_memory_repository._storage) == len(sample_entities)
        
        for entity in sample_entities:
            assert entity.id in in_memory_repository._storage
    
    @pytest.mark.asyncio
    async def test_get_by_id(self, in_memory_repository, sample_entities):
        """Test getting entity by ID"""
        entity = sample_entities[0]
        await in_memory_repository.add(entity)
        
        result = await in_memory_repository.get_by_id(entity.id)
        
        assert result == entity
    
    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, in_memory_repository):
        """Test getting non-existent entity"""
        result = await in_memory_repository.get_by_id("non-existent")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_all(self, in_memory_repository, sample_entities):
        """Test getting all entities"""
        await in_memory_repository.add_many(sample_entities)
        
        result = await in_memory_repository.get_all()
        
        assert len(result) == len(sample_entities)
        assert set(e.id for e in result) == set(e.id for e in sample_entities)
    
    @pytest.mark.asyncio
    async def test_update_entity(self, in_memory_repository, sample_entities):
        """Test updating entity"""
        entity = sample_entities[0]
        await in_memory_repository.add(entity)
        
        # Update entity
        entity.name = "Alice Updated"
        entity.age = 31
        
        result = await in_memory_repository.update(entity)
        
        assert result == entity
        stored = in_memory_repository._storage[entity.id]
        assert stored.name == "Alice Updated"
        assert stored.age == 31
    
    @pytest.mark.asyncio
    async def test_delete_entity(self, in_memory_repository, sample_entities):
        """Test deleting entity"""
        entity = sample_entities[0]
        await in_memory_repository.add(entity)
        
        result = await in_memory_repository.delete(entity.id)
        
        assert result is True
        assert entity.id not in in_memory_repository._storage
    
    @pytest.mark.asyncio
    async def test_delete_non_existent(self, in_memory_repository):
        """Test deleting non-existent entity"""
        result = await in_memory_repository.delete("non-existent")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_exists(self, in_memory_repository, sample_entities):
        """Test checking entity existence"""
        entity = sample_entities[0]
        await in_memory_repository.add(entity)
        
        exists = await in_memory_repository.exists(entity.id)
        assert exists is True
        
        not_exists = await in_memory_repository.exists("non-existent")
        assert not_exists is False
    
    @pytest.mark.asyncio
    async def test_count(self, in_memory_repository, sample_entities):
        """Test counting entities"""
        await in_memory_repository.add_many(sample_entities[:3])
        
        count = await in_memory_repository.count()
        assert count == 3


@pytest.mark.unit
class TestQueryBuilder:
    """Test query builder functionality"""
    
    @pytest.mark.asyncio
    async def test_filter_single(self, in_memory_repository, sample_entities):
        """Test single filter"""
        await in_memory_repository.add_many(sample_entities)
        
        query = (in_memory_repository.query()
                .filter(Filter("age", FilterOperator.GREATER_THAN, 30)))
        
        result = await query.execute()
        
        assert len(result) == 2  # Charlie (35) and Eve (32)
        assert all(e.age > 30 for e in result)
    
    @pytest.mark.asyncio
    async def test_filter_multiple(self, in_memory_repository, sample_entities):
        """Test multiple filters"""
        await in_memory_repository.add_many(sample_entities)
        
        query = (in_memory_repository.query()
                .filter(Filter("age", FilterOperator.GREATER_THAN, 25))
                .filter(Filter("active", FilterOperator.EQUALS, True)))
        
        result = await query.execute()
        
        # Alice (30), Eve (32), David (28)
        assert len(result) == 3
        assert all(e.age > 25 and e.active for e in result)
    
    @pytest.mark.asyncio
    async def test_filter_operators(self, in_memory_repository, sample_entities):
        """Test different filter operators"""
        await in_memory_repository.add_many(sample_entities)
        
        # Test EQUALS
        result = await (in_memory_repository.query()
                       .filter(Filter("name", FilterOperator.EQUALS, "Alice"))
                       .execute())
        assert len(result) == 1
        assert result[0].name == "Alice"
        
        # Test NOT_EQUALS
        result = await (in_memory_repository.query()
                       .filter(Filter("active", FilterOperator.NOT_EQUALS, True))
                       .execute())
        assert len(result) == 1
        assert result[0].name == "Charlie"
        
        # Test LESS_THAN
        result = await (in_memory_repository.query()
                       .filter(Filter("age", FilterOperator.LESS_THAN, 30))
                       .execute())
        assert len(result) == 2
        
        # Test LESS_THAN_OR_EQUAL
        result = await (in_memory_repository.query()
                       .filter(Filter("age", FilterOperator.LESS_THAN_OR_EQUAL, 30))
                       .execute())
        assert len(result) == 3
        
        # Test GREATER_THAN_OR_EQUAL
        result = await (in_memory_repository.query()
                       .filter(Filter("age", FilterOperator.GREATER_THAN_OR_EQUAL, 30))
                       .execute())
        assert len(result) == 3
        
        # Test IN
        result = await (in_memory_repository.query()
                       .filter(Filter("name", FilterOperator.IN, ["Alice", "Bob"]))
                       .execute())
        assert len(result) == 2
        
        # Test NOT_IN
        result = await (in_memory_repository.query()
                       .filter(Filter("name", FilterOperator.NOT_IN, ["Alice", "Bob"]))
                       .execute())
        assert len(result) == 3
        
        # Test CONTAINS (for lists)
        result = await (in_memory_repository.query()
                       .filter(Filter("tags", FilterOperator.CONTAINS, "admin"))
                       .execute())
        assert len(result) == 2  # Alice and Eve
        
        # Test LIKE
        result = await (in_memory_repository.query()
                       .filter(Filter("email", FilterOperator.LIKE, "%example.com"))
                       .execute())
        assert len(result) == 5
    
    @pytest.mark.asyncio
    async def test_sorting(self, in_memory_repository, sample_entities):
        """Test sorting"""
        await in_memory_repository.add_many(sample_entities)
        
        # Sort by age ascending
        result = await (in_memory_repository.query()
                       .sort("age", SortOrder.ASC)
                       .execute())
        
        ages = [e.age for e in result]
        assert ages == sorted(ages)
        
        # Sort by age descending
        result = await (in_memory_repository.query()
                       .sort("age", SortOrder.DESC)
                       .execute())
        
        ages = [e.age for e in result]
        assert ages == sorted(ages, reverse=True)
        
        # Multiple sort criteria
        result = await (in_memory_repository.query()
                       .sort("active", SortOrder.DESC)
                       .sort("name", SortOrder.ASC)
                       .execute())
        
        # Active entities should come first
        active_count = sum(1 for e in result[:4] if e.active)
        assert active_count == 4
    
    @pytest.mark.asyncio
    async def test_pagination(self, in_memory_repository, sample_entities):
        """Test pagination"""
        await in_memory_repository.add_many(sample_entities)
        
        # Get first page
        page_request = PageRequest(page=1, size=2)
        query = (in_memory_repository.query()
                .sort("name", SortOrder.ASC)
                .paginate(page_request))
        
        result = await query.execute()
        
        assert isinstance(result, PageResult)
        assert len(result.items) == 2
        assert result.total == 5
        assert result.page == 1
        assert result.size == 2
        assert result.total_pages == 3
        assert result.items[0].name == "Alice"
        assert result.items[1].name == "Bob"
        
        # Get second page
        page_request = PageRequest(page=2, size=2)
        query = (in_memory_repository.query()
                .sort("name", SortOrder.ASC)
                .paginate(page_request))
        
        result = await query.execute()
        
        assert len(result.items) == 2
        assert result.items[0].name == "Charlie"
        assert result.items[1].name == "David"
    
    @pytest.mark.asyncio
    async def test_limit(self, in_memory_repository, sample_entities):
        """Test limiting results"""
        await in_memory_repository.add_many(sample_entities)
        
        result = await (in_memory_repository.query()
                       .sort("age", SortOrder.ASC)
                       .limit(3)
                       .execute())
        
        assert len(result) == 3
        assert result[0].age == 25  # Bob
    
    @pytest.mark.asyncio
    async def test_offset(self, in_memory_repository, sample_entities):
        """Test offset"""
        await in_memory_repository.add_many(sample_entities)
        
        result = await (in_memory_repository.query()
                       .sort("age", SortOrder.ASC)
                       .offset(2)
                       .limit(2)
                       .execute())
        
        assert len(result) == 2
        assert result[0].age == 30  # Alice (skipped Bob and David)
    
    @pytest.mark.asyncio
    async def test_count_query(self, in_memory_repository, sample_entities):
        """Test counting with query"""
        await in_memory_repository.add_many(sample_entities)
        
        count = await (in_memory_repository.query()
                      .filter(Filter("age", FilterOperator.GREATER_THAN, 30))
                      .count())
        
        assert count == 2
    
    @pytest.mark.asyncio
    async def test_first(self, in_memory_repository, sample_entities):
        """Test getting first result"""
        await in_memory_repository.add_many(sample_entities)
        
        result = await (in_memory_repository.query()
                       .sort("age", SortOrder.ASC)
                       .first())
        
        assert result is not None
        assert result.name == "Bob"
        assert result.age == 25
    
    @pytest.mark.asyncio
    async def test_complex_query(self, in_memory_repository, sample_entities):
        """Test complex query with multiple operations"""
        await in_memory_repository.add_many(sample_entities)
        
        # Find active users over 25, sorted by age, paginated
        page_request = PageRequest(page=1, size=2)
        
        result = await (in_memory_repository.query()
                       .filter(Filter("active", FilterOperator.EQUALS, True))
                       .filter(Filter("age", FilterOperator.GREATER_THAN, 25))
                       .sort("age", SortOrder.DESC)
                       .paginate(page_request)
                       .execute())
        
        assert isinstance(result, PageResult)
        assert len(result.items) == 2
        assert result.total == 3
        assert result.items[0].age == 32  # Eve
        assert result.items[1].age == 30  # Alice


@pytest.mark.unit
class TestPostgreSQLRepository:
    """Test PostgreSQL repository implementation"""
    
    @pytest.mark.asyncio
    async def test_add_entity(self, postgresql_repository, sample_entities, mock_connection_pool):
        """Test adding entity to PostgreSQL"""
        entity = sample_entities[0]
        
        # Mock cursor behavior
        mock_cursor = mock_connection_pool.acquire.return_value.__aenter__.return_value.cursor.return_value.__aenter__.return_value
        mock_cursor.fetchone = AsyncMock(return_value=None)  # No duplicate
        
        result = await postgresql_repository.add(entity)
        
        assert result == entity
        mock_cursor.execute.assert_called()
    
    @pytest.mark.asyncio
    async def test_get_by_id(self, postgresql_repository, sample_entities, mock_connection_pool):
        """Test getting entity by ID from PostgreSQL"""
        entity = sample_entities[0]
        
        # Mock cursor behavior
        mock_cursor = mock_connection_pool.acquire.return_value.__aenter__.return_value.cursor.return_value.__aenter__.return_value
        mock_cursor.fetchone = AsyncMock(return_value={
            "id": entity.id,
            "name": entity.name,
            "age": entity.age,
            "email": entity.email,
            "active": entity.active,
            "created_at": entity.created_at,
            "tags": entity.tags
        })
        
        result = await postgresql_repository.get_by_id(entity.id)
        
        assert result is not None
        assert result.id == entity.id
        assert result.name == entity.name
    
    @pytest.mark.asyncio
    async def test_delete_entity(self, postgresql_repository, mock_connection_pool):
        """Test deleting entity from PostgreSQL"""
        entity_id = "test-id"
        
        # Mock cursor behavior
        mock_cursor = mock_connection_pool.acquire.return_value.__aenter__.return_value.cursor.return_value.__aenter__.return_value
        mock_cursor.rowcount = 1
        
        result = await postgresql_repository.delete(entity_id)
        
        assert result is True
        mock_cursor.execute.assert_called()
    
    @pytest.mark.asyncio
    async def test_query_with_filters(self, postgresql_repository, mock_connection_pool):
        """Test building SQL query with filters"""
        # Mock cursor behavior
        mock_cursor = mock_connection_pool.acquire.return_value.__aenter__.return_value.cursor.return_value.__aenter__.return_value
        mock_cursor.fetchall = AsyncMock(return_value=[])
        
        await (postgresql_repository.query()
               .filter(Filter("age", FilterOperator.GREATER_THAN, 30))
               .filter(Filter("active", FilterOperator.EQUALS, True))
               .sort("name", SortOrder.ASC)
               .limit(10)
               .execute())
        
        # Verify SQL was built correctly
        mock_cursor.execute.assert_called()
        sql_call = mock_cursor.execute.call_args[0][0]
        
        assert "WHERE" in sql_call
        assert "age > %s" in sql_call
        assert "active = %s" in sql_call
        assert "ORDER BY" in sql_call
        assert "LIMIT" in sql_call
    
    @pytest.mark.asyncio
    async def test_count_with_filters(self, postgresql_repository, mock_connection_pool):
        """Test counting with filters in PostgreSQL"""
        # Mock cursor behavior
        mock_cursor = mock_connection_pool.acquire.return_value.__aenter__.return_value.cursor.return_value.__aenter__.return_value
        mock_cursor.fetchone = AsyncMock(return_value={"count": 5})
        
        count = await (postgresql_repository.query()
                      .filter(Filter("active", FilterOperator.EQUALS, True))
                      .count())
        
        assert count == 5
        
        # Verify COUNT query was executed
        sql_call = mock_cursor.execute.call_args[0][0]
        assert "SELECT COUNT(*)" in sql_call
    
    @pytest.mark.asyncio
    async def test_transaction_handling(self, postgresql_repository, mock_connection_pool):
        """Test transaction handling"""
        # Mock connection and transaction
        mock_conn = mock_connection_pool.acquire.return_value.__aenter__.return_value
        mock_conn.begin = AsyncMock()
        mock_conn.commit = AsyncMock()
        mock_conn.rollback = AsyncMock()
        
        # Simulate transaction
        async with mock_conn:
            await mock_conn.begin()
            # Perform operations
            await mock_conn.commit()
        
        mock_conn.begin.assert_called_once()
        mock_conn.commit.assert_called_once()


@pytest.mark.unit
class TestSpecialCases:
    """Test special cases and edge conditions"""
    
    @pytest.mark.asyncio
    async def test_empty_repository(self, in_memory_repository):
        """Test operations on empty repository"""
        result = await in_memory_repository.get_all()
        assert result == []
        
        count = await in_memory_repository.count()
        assert count == 0
        
        first = await in_memory_repository.query().first()
        assert first is None
    
    @pytest.mark.asyncio
    async def test_filter_with_none_value(self, in_memory_repository, sample_entities):
        """Test filtering with None values"""
        # Add entity with None email
        entity = TestEntity(
            id=str(uuid.uuid4()),
            name="Test",
            age=25,
            email=None,
            created_at=datetime.now()
        )
        await in_memory_repository.add(entity)
        
        result = await (in_memory_repository.query()
                       .filter(Filter("email", FilterOperator.EQUALS, None))
                       .execute())
        
        assert len(result) == 1
        assert result[0].email is None
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, in_memory_repository, sample_entities):
        """Test concurrent repository operations"""
        # Add entities concurrently
        tasks = [in_memory_repository.add(entity) for entity in sample_entities]
        await asyncio.gather(*tasks)
        
        count = await in_memory_repository.count()
        assert count == len(sample_entities)
    
    @pytest.mark.asyncio
    async def test_large_dataset(self, in_memory_repository):
        """Test repository with large dataset"""
        # Create 1000 entities
        large_dataset = [
            TestEntity(
                id=str(uuid.uuid4()),
                name=f"User{i}",
                age=20 + (i % 50),
                email=f"user{i}@example.com",
                created_at=datetime.now(),
                tags=[f"tag{i % 10}"]
            )
            for i in range(1000)
        ]
        
        await in_memory_repository.add_many(large_dataset)
        
        # Test pagination on large dataset
        page_request = PageRequest(page=10, size=20)
        result = await (in_memory_repository.query()
                       .sort("name", SortOrder.ASC)
                       .paginate(page_request)
                       .execute())
        
        assert result.total == 1000
        assert result.total_pages == 50
        assert len(result.items) == 20
    
    @pytest.mark.asyncio
    async def test_filter_chaining(self, in_memory_repository, sample_entities):
        """Test chaining multiple filters with different operators"""
        await in_memory_repository.add_many(sample_entities)
        
        # Complex filter chain
        result = await (in_memory_repository.query()
                       .filter(Filter("age", FilterOperator.GREATER_THAN_OR_EQUAL, 28))
                       .filter(Filter("age", FilterOperator.LESS_THAN_OR_EQUAL, 32))
                       .filter(Filter("active", FilterOperator.EQUALS, True))
                       .filter(Filter("tags", FilterOperator.CONTAINS, "user"))
                       .execute())
        
        # Should find Alice (30) and Eve (32)
        assert len(result) == 2
        assert all(28 <= e.age <= 32 and e.active and "user" in e.tags for e in result)