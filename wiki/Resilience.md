# Resilience

Aligo includes built-in resilience patterns to handle LLM service instability, network failures, and rate limiting.

## Circuit Breaker

**Module**: `utils/circuit_breaker.py`

Prevents cascading failures by tracking consecutive LLM call failures and temporarily blocking requests when the service appears unhealthy.

### State Machine

```
        success (threshold met)
   ┌─────────────────────────┐
   │                         │
   v         failure          │
CLOSED ─────────────────> OPEN
   ^     (threshold reached)  │
   │                          │ timeout expired
   │    success               v
   └──────────── HALF_OPEN <──┘
                    │
                    │ failure
                    v
                   OPEN
```

| State | Behavior |
|-------|----------|
| **CLOSED** | Normal operation. Counts consecutive failures. Transitions to OPEN after `failure_threshold` consecutive failures. |
| **OPEN** | All calls blocked. `CircuitOpenError` raised immediately. Transitions to HALF_OPEN after `recovery_timeout_sec`. |
| **HALF_OPEN** | Probe mode — allows calls through. If `half_open_successes` consecutive successes occur, transitions to CLOSED. Any failure returns to OPEN. |

### Configuration

In `config.py` under `RESILIENCE_CONFIG`:

```python
RESILIENCE_CONFIG = {
    "circuit_failure_threshold": 5,         # Failures before opening
    "circuit_recovery_timeout_sec": 60.0,   # Seconds before half-open
    "circuit_half_open_successes": 2,        # Successes to close
}
```

### Usage in CLI

```python
# Before each query
try:
    self.circuit_breaker.raise_if_open()
except CircuitOpenError:
    print("Service temporarily unavailable")
    return

# After successful call
self.circuit_breaker.record_success()

# After failed call
self.circuit_breaker.record_failure()
```

---

## Retry with Exponential Backoff

**Module**: `utils/llm_resilience.py`

### `retry_with_backoff()`

Wraps async LLM calls with automatic retry logic:

- **Max retries**: 3 (configurable) — total attempts = 1 + max_retries
- **Backoff formula**: `min(base_delay * 2^attempt, max_delay)`
- **Jitter**: Random factor between 0.5x and 1.5x to prevent thundering herd
- **Retriable errors**: Timeouts, connection errors, HTTP 429/5xx

```python
result = await retry_with_backoff(
    coro_factory=lambda: llm.ainvoke(prompt),
    max_retries=3,
    base_delay_sec=1.0,
    max_delay_sec=30.0,
    jitter=True,
)
```

The `coro_factory` pattern (a callable returning a new coroutine each time) prevents the "coroutine already awaited" error that would occur if passing a coroutine directly.

### `is_retriable_error()`

Classifies exceptions as retriable or permanent:

| Retriable | Not Retriable |
|-----------|---------------|
| `asyncio.TimeoutError` | `ValueError` |
| `ConnectionError` | `KeyError` |
| HTTP 429 (rate limit) | HTTP 400 (bad request) |
| HTTP 500/502/503/504 | HTTP 401/403 (auth) |
| `OSError` | Other errors |

### Configuration

```python
RESILIENCE_CONFIG = {
    "max_retries": 3,
    "retry_base_delay_sec": 1.0,
    "retry_max_delay_sec": 30.0,
}
```

---

## Health Check

**Function**: `run_health_check()` in `utils/llm_resilience.py`

A minimal probe that sends a single-token request to the LLM service to verify connectivity:

```python
success, message = await run_health_check(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key="...",
    model_name="doubao-seed-1-6-lite-251015",
    timeout_sec=10.0,
)
```

Accessible from the CLI via the `health` command.

---

## Resilience Flow

When a user submits a query, the system applies resilience in this order:

1. **Circuit breaker check**: If OPEN, reject immediately with a user-friendly message
2. **LLM call**: Execute through the LangGraph pipeline
3. **On success**: `circuit_breaker.record_success()`
4. **On failure**: `circuit_breaker.record_failure()` — increments failure count
5. If the failure count reaches the threshold, the circuit opens for `recovery_timeout_sec`
