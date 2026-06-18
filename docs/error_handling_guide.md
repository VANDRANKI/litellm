# LiteLLM Error Handling Guide

This guide documents how LiteLLM handles exceptions from 100+ LLM providers
and translates them into consistent, OpenAI-compatible error types.

## Exception Hierarchy

```
Exception
└── openai.OpenAIError  (base for all LiteLLM exceptions)
    ├── openai.AuthenticationError   → 401
    ├── openai.PermissionDeniedError → 403
    ├── openai.NotFoundError         → 404
    ├── openai.UnprocessableEntityError → 422
    ├── openai.RateLimitError        → 429
    ├── openai.InternalServerError   → 500
    ├── openai.APIConnectionError    → network failure
    └── openai.APITimeoutError       → timeout
```

All provider-specific exceptions (Anthropic `OverloadedError`, OpenAI `RateLimitError`,
Bedrock `ThrottlingException`, etc.) are caught internally and re-raised as the
corresponding OpenAI-compatible type listed above.

## Catching Errors

```python
import litellm
from openai import RateLimitError, APIConnectionError, AuthenticationError

try:
    response = litellm.completion(
        model="anthropic/claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": "Hello"}],
    )
except RateLimitError as e:
    # Provider returned 429. e.status_code == 429.
    # e.response contains the raw HTTP response if available.
    print(f"Rate limited: {e.message}")
except APIConnectionError as e:
    # Network failure — safe to retry with backoff.
    print(f"Connection error: {e}")
except AuthenticationError as e:
    # Bad API key — do NOT retry.
    print(f"Auth failure: {e}")
```

## Retry Strategy

LiteLLM's `Router` handles retries automatically across deployments.
For single-model usage, use `litellm.completion` with `num_retries`:

```python
response = litellm.completion(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
    num_retries=3,          # retry up to 3 times on retryable errors
    timeout=30,             # per-request timeout in seconds
)
```

Errors that are **retried by default**:
- `RateLimitError` (429)
- `APIConnectionError` (network)
- `InternalServerError` (500, 502, 503)
- `APITimeoutError`

Errors that are **not retried**:
- `AuthenticationError` (401)
- `PermissionDeniedError` (403)
- `NotFoundError` (404) — usually a model name typo
- `UnprocessableEntityError` (422) — invalid request

## Router Fallbacks

For production workloads, use the Router with fallbacks:

```python
from litellm import Router

router = Router(
    model_list=[
        {"model_name": "gpt-4o", "litellm_params": {"model": "openai/gpt-4o"}},
        {"model_name": "gpt-4o", "litellm_params": {"model": "azure/gpt-4o", "api_base": "..."}},
    ],
    fallbacks=[{"gpt-4o": ["claude-3-5"]}],
    num_retries=2,
    retry_after=5,          # seconds to wait before retry
    allowed_fails=3,        # mark deployment unhealthy after 3 fails
    cooldown_time=60,       # seconds to cool down unhealthy deployment
)
```

## Provider-Specific Error Codes

| Provider       | Native Error              | LiteLLM Mapping        |
|----------------|--------------------------|------------------------|
| OpenAI         | `RateLimitError`         | `RateLimitError`       |
| Anthropic      | `OverloadedError`        | `InternalServerError`  |
| Anthropic      | `RateLimitError`         | `RateLimitError`       |
| Bedrock        | `ThrottlingException`    | `RateLimitError`       |
| Bedrock        | `ModelNotReadyException` | `InternalServerError`  |
| Azure          | `429 Too Many Requests`  | `RateLimitError`       |
| Vertex AI      | `ResourceExhausted`      | `RateLimitError`       |
| Groq           | `RateLimitError`         | `RateLimitError`       |

## Adding Error Handling for a New Provider

When implementing a new provider in `litellm/llms/<provider>/`, add an
exception mapping in the provider's `__init__.py`:

```python
from litellm.exceptions import RateLimitError, InternalServerError

def handle_provider_error(e: Exception, provider_name: str) -> None:
    """Map provider-specific errors to OpenAI-compatible LiteLLM exceptions.

    Args:
        e: The raw exception from the provider SDK.
        provider_name: Used to prefix the error message for debugging.

    Raises:
        RateLimitError: For quota/throttle errors.
        InternalServerError: For 5xx and overload errors.
        AuthenticationError: For invalid API key errors.
    """
    error_str = str(e).lower()
    if "rate limit" in error_str or "quota" in error_str or "throttl" in error_str:
        raise RateLimitError(
            message=f"{provider_name}: {e}",
            llm_provider=provider_name,
            model="unknown",
        ) from e
    if "overload" in error_str or "server error" in error_str:
        raise InternalServerError(
            message=f"{provider_name}: {e}",
            llm_provider=provider_name,
            model="unknown",
        ) from e
    raise e
```

## Testing Error Handling

Unit tests for error handling live in `tests/test_litellm/test_exceptions.py`.
To add tests for a new provider's error mapping:

```python
import pytest
from unittest.mock import patch, MagicMock
from openai import RateLimitError
import litellm

def test_provider_rate_limit_maps_correctly():
    with patch("litellm.main.my_provider_completion") as mock_call:
        mock_call.side_effect = MyProviderRateLimitError("quota exceeded")
        with pytest.raises(RateLimitError):
            litellm.completion(
                model="myprovider/my-model",
                messages=[{"role": "user", "content": "test"}],
            )
```
