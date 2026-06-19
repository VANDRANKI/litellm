# Error Handling Guide

This guide describes best practices for error handling in LiteLLM.

## Provider Errors

Map provider-specific exceptions to OpenAI-compatible error classes using `litellm.exceptions`.

### Retry Logic

Use exponential backoff with jitter for transient errors:
- Initial delay: 1s
- Max delay: 60s
- Max retries: 3 (configurable via `num_retries`)

### Logging

Always log the original exception before re-raising a mapped exception.
Use `litellm._logging.verbose_logger` for debug output.

## Caching Errors

Cache misses should never raise exceptions — fall through to the provider.
Cache write failures should be logged at WARNING level and not propagate.

## Router Fallback

When a model falls back to an alternate deployment, log:
- The original model alias
- The reason for fallback (timeout, rate limit, etc.)
- The selected fallback model

This helps operators diagnose routing issues in production.
