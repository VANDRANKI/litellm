# LiteLLM Caching Strategy Guide

This guide explains when and how to use LiteLLM's caching layer to reduce latency and
cost for repeated prompts.

## Why Cache?

LLM API calls are slow and expensive. Identical or near-identical prompts sent within a
short window are perfect candidates for caching — the model output is deterministic at
`temperature=0` and highly consistent at low temperatures.

## Supported Backends

| Backend | Best For | Persistence | Setup Complexity |
|---------|----------|-------------|------------------|
| `InMemoryCache` | Development, single-process | None | Zero |
| `RedisCache` | Production, multi-process | Optional TTL | Low |
| `RedisSemanticCache` | Fuzzy/semantic match | TTL | Medium |
| `S3Cache` | Long-lived archival | Permanent | Medium |
| `DiskCache` | Local dev with persistence | On-disk | Low |

## Quick Start

### In-Memory Cache

```python
import litellm
from litellm.caching import Cache

litellm.cache = Cache()  # defaults to in-memory

response = litellm.completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    caching=True,
)
# Second call with identical messages hits the cache — zero latency, zero cost.
response2 = litellm.completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    caching=True,
)
print(response2._hidden_params["cache_hit"])  # True
```

### Redis Cache

```python
from litellm.caching import Cache

litellm.cache = Cache(
    type="redis",
    host="localhost",
    port=6379,
    ttl=3600,  # 1 hour TTL
)
```

Set via environment variables instead:

```bash
export REDIS_HOST=localhost
export REDIS_PORT=6379
export LITELLM_CACHE_TTL=3600
```

### Semantic Cache (Redis)

Use when prompts vary slightly but should return the same answer:

```python
from litellm.caching import Cache

litellm.cache = Cache(
    type="redis-semantic",
    host="localhost",
    port=6379,
    similarity_threshold=0.90,  # 0.0–1.0; higher = stricter matching
    redis_semantic_cache_embedding_model="text-embedding-ada-002",
)
```

## Proxy Server Configuration

```yaml
# proxy_server_config.yaml
general_settings:
  cache: true
  cache_params:
    type: redis
    host: ${REDIS_HOST}
    port: 6379
    ttl: 86400
```

## Controlling Cache at Call Time

```python
# Force a fresh response, bypass cache read
response = litellm.completion(..., cache={"no-cache": True})

# Store the result but do not read from cache
response = litellm.completion(..., cache={"no-store": True})

# Override TTL for this specific call
response = litellm.completion(..., cache={"ttl": 60})  # 60-second TTL
```

## Cache Key Customisation

By default the cache key is a hash of `(model, messages, temperature, max_tokens)`.
Pass `cache_key_params` to add or exclude fields:

```python
response = litellm.completion(
    model="gpt-4o",
    messages=[...],
    cache={"key_params": ["model", "messages"]},  # ignore temperature in key
    caching=True,
)
```

## Async Usage

```python
import asyncio
import litellm
from litellm.caching import Cache

litellm.cache = Cache(type="redis", host="localhost", port=6379)

async def ask(prompt: str) -> str:
    response = await litellm.acompletion(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        caching=True,
    )
    return response.choices[0].message.content

asyncio.run(ask("Summarise quantum entanglement in one sentence."))
```

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `cache_hit` is always `False` | Cache not enabled globally | Set `litellm.cache` and pass `caching=True` |
| Redis connection errors | Wrong host/port/auth | Check `REDIS_HOST` / `REDIS_PORT` env vars |
| Stale responses returned | TTL too long | Reduce `ttl` or call with `no-cache: True` |
| Semantic cache returning wrong hits | Threshold too low | Increase `similarity_threshold` (try 0.95) |

## Best Practices

1. **Use `temperature=0`** for deterministic prompts — cache hit rates drop sharply with
   non-zero temperatures because responses diverge.
2. **Set a TTL** even for in-memory caches; use `ttl` parameter to avoid unbounded memory
   growth.
3. **Tag cache entries** with a `user_id` or tenant identifier by including it in the
   message system prompt so different tenants never share cached responses.
4. **Monitor cache hit rate** via the `_hidden_params["cache_hit"]` field or the
   Prometheus metric `litellm_cache_hit_total`.
