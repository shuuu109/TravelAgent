# Testing

## Test Structure

Tests are located in the `tests/` directory and use Python's standard testing tools.

```
tests/
├── test_graph_integration.py   # Full pipeline integration tests
├── test_memory.py              # Memory system unit tests
└── ...
```

## Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run a specific test file
python -m pytest tests/test_memory.py

# Run with verbose output
python -m pytest tests/ -v
```

## Integration Test Tips

Since the system depends on external LLM and MCP services, integration tests may require:

1. **Valid API keys** in `config.py` for the Doubao LLM and Amap services
2. **Network access** to the LLM and MCP endpoints
3. **Embedding model** downloaded to `data/models/bge-small-zh-v1.5/`

For unit testing without external dependencies, mock the LLM and MCP calls:

```python
from unittest.mock import AsyncMock, patch

@patch("graph.nodes.intent_node.create_intent_node")
async def test_intent_parsing(mock_intent):
    mock_intent.return_value = AsyncMock(return_value={
        "intent_data": {...},
        "intent_schedule": [...],
    })
    # Test logic here
```

## Manual Verification

Use the CLI's built-in commands to verify system health:

```bash
python cli.py
> health    # Check LLM connectivity
> status    # Verify system state
```

Try a sample query to confirm end-to-end functionality:

```
> 我想从北京去杭州玩3天
```

Verify that:
- Intent is correctly recognized
- POIs are fetched and displayed
- Daily routes are generated
- The response is well-formatted Markdown
