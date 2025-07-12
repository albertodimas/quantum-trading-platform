# Core Architecture Unit Tests

This directory contains comprehensive unit tests for all core architectural components of the Quantum Trading Platform.

## Test Coverage

### ✅ Completed Tests

#### 1. **test_dependency_injection.py**
- Tests the Dependency Injection Container functionality
- Covers singleton, transient, and scoped lifecycles
- Tests circular dependency detection
- Validates decorator patterns (@injectable, @inject)
- Tests concurrent resolution and thread safety

#### 2. **test_repository.py**
- Tests Repository Pattern implementation
- Covers both InMemory and PostgreSQL repositories
- Tests QueryBuilder with complex filtering and sorting
- Validates pagination and limiting functionality
- Tests transaction handling and concurrent operations

#### 3. **test_unit_of_work.py**
- Tests Unit of Work pattern for transaction management
- Covers commit and rollback scenarios
- Tests savepoint functionality
- Validates @transactional decorator
- Tests complex business transactions across multiple repositories

#### 4. **test_event_bus.py**
- Tests Event Bus system for decoupled communication
- Covers event publishing and subscription
- Tests wildcard patterns and event filtering
- Validates priority-based processing
- Tests error handling and dead letter queues

#### 5. **test_circuit_breaker.py**
- Tests Circuit Breaker pattern for fault tolerance
- Covers all states: CLOSED, OPEN, HALF_OPEN
- Tests failure threshold and recovery mechanisms
- Validates timeout handling
- Tests decorator patterns and concurrent access

#### 6. **test_rate_limiter.py**
- Tests all rate limiting strategies:
  - Token Bucket
  - Sliding Window
  - Fixed Window
  - Leaky Bucket
- Tests decorator functionality
- Validates concurrent request handling
- Tests adaptive rate limiting

## Running the Tests

### Run all core architecture tests:
```bash
pytest tests/unit/core -v
```

### Run specific test file:
```bash
pytest tests/unit/core/test_dependency_injection.py -v
```

### Run with coverage:
```bash
pytest tests/unit/core --cov=src/core/architecture --cov-report=html
```

### Run specific test class:
```bash
pytest tests/unit/core/test_event_bus.py::TestEventBus -v
```

### Run specific test method:
```bash
pytest tests/unit/core/test_circuit_breaker.py::TestCircuitBreakerBasics::test_initial_state -v
```

## Test Structure

Each test file follows a consistent structure:

1. **Imports and Setup**: Required dependencies and test utilities
2. **Test Models/Classes**: Mock objects and test-specific implementations
3. **Fixtures**: Reusable test data and configurations
4. **Test Classes**: Organized by functionality area
5. **Test Methods**: Individual test cases with descriptive names

## Key Testing Patterns

### 1. **Async Testing**
```python
@pytest.mark.asyncio
async def test_async_operation(self):
    result = await async_function()
    assert result == expected
```

### 2. **Mocking**
```python
@pytest.fixture
def mock_service():
    service = AsyncMock()
    service.method.return_value = "mocked"
    return service
```

### 3. **Parameterized Tests**
```python
@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6)
])
def test_multiplication(self, input, expected):
    assert input * 2 == expected
```

### 4. **Exception Testing**
```python
with pytest.raises(CustomException) as exc_info:
    raise_exception()
assert "expected message" in str(exc_info.value)
```

## Test Markers

- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.asyncio`: Async tests
- `@pytest.mark.slow`: Slow running tests
- `@pytest.mark.requires_redis`: Tests requiring Redis
- `@pytest.mark.requires_postgres`: Tests requiring PostgreSQL

## Coverage Goals

- **Target**: 80%+ code coverage
- **Current**: Check with `pytest --cov`
- **Focus Areas**: 
  - Core business logic
  - Error handling paths
  - Edge cases and boundaries
  - Concurrent operations

## Common Test Utilities

### Mock Event Bus
```python
@pytest.fixture
def mock_event_bus():
    bus = AsyncMock()
    bus.publish = AsyncMock()
    bus.subscribe = Mock()
    return bus
```

### Sample Data Generators
```python
def create_sample_order():
    return Order(
        id=str(uuid.uuid4()),
        symbol="BTC/USDT",
        quantity=0.1,
        price=50000.0
    )
```

## Best Practices

1. **Test Independence**: Each test should be independent and not rely on others
2. **Clear Naming**: Test names should describe what they test
3. **Arrange-Act-Assert**: Follow AAA pattern for test structure
4. **Mock External Dependencies**: Don't test external services
5. **Test Edge Cases**: Include boundary conditions and error scenarios
6. **Keep Tests Fast**: Mock slow operations like I/O
7. **Use Fixtures**: Reuse common test setup through fixtures

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure PYTHONPATH includes project root
2. **Async Warnings**: Use pytest-asyncio for async tests
3. **Fixture Scope**: Be aware of fixture lifecycle
4. **Mock Reset**: Reset mocks between tests if reused

### Debugging Tests

```bash
# Run with verbose output
pytest -vv tests/unit/core/test_name.py

# Run with print statements visible
pytest -s tests/unit/core/test_name.py

# Run with debugger
pytest --pdb tests/unit/core/test_name.py
```

## Next Steps

1. Continue with trading component unit tests
2. Add performance benchmarks to critical paths
3. Implement property-based testing for complex logic
4. Add mutation testing to verify test quality