# Weaviate Instrumentation Tests

This directory contains tests for the OpenTelemetry Weaviate instrumentation.

## Test Structure

- `conftest.py` - Pytest fixtures and configuration
- `test_instrumentation.py` - Basic instrumentation tests
- `test_weaviate_v3.py` - Weaviate v3 client specific tests
- `test_weaviate_v4.py` - Weaviate v4 client specific tests
- `test_utils.py` - Utility function tests

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run specific test file
```bash
pytest tests/test_instrumentation.py
```

### Run with coverage
```bash
pytest --cov=opentelemetry.instrumentation.weaviate tests/
```

### Run only unit tests (skip integration tests)
```bash
pytest tests/ -m "not integration"
```

## Test Requirements

Install test dependencies:
```bash
pip install pytest pytest-cov weaviate-client
```

## Integration Tests

Some tests are marked with `@pytest.mark.integration` and require a running Weaviate instance:

```bash
docker run -d -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:latest
```

Run integration tests:
```bash
pytest tests/ -m integration
```

## Test Coverage

The tests cover:
- ✅ Instrumentation initialization and cleanup
- ✅ Version detection (v3 vs v4)
- ✅ Span name mapping for all operations
- ✅ Utility functions (URL parsing, collection name extraction)
- ✅ Double instrumentation handling
- ✅ Module structure validation

## Adding New Tests

When adding new Weaviate operations to the instrumentation:
1. Add the operation mapping to `mapping.py`
2. Add corresponding test in `test_weaviate_v3.py` or `test_weaviate_v4.py`
3. Verify span names and attributes are correct
