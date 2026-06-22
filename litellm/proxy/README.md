# litellm-proxy

A local, fast, and lightweight **OpenAI-compatible server** to call 100+ LLM APIs.

## usage 

```shell 
$ uv tool install litellm
```
```shell
$ litellm --model ollama/codellama 

#INFO: Ollama running on http://0.0.0.0:8000
```

## replace openai base
```python 
import openai # openai v1.0.0+
client = openai.OpenAI(api_key="anything",base_url="http://0.0.0.0:8000") # set proxy to base_url
# request sent to model set on litellm proxy, `litellm --model`
response = client.chat.completions.create(model="gpt-3.5-turbo", messages = [
    {
        "role": "user",
        "content": "this is a test request, write a short poem"
    }
])

print(response)
``` 

[**See how to call Huggingface,Bedrock,TogetherAI,Anthropic, etc.**](https://docs.litellm.ai/docs/simple_proxy)


---

### Folder Structure

**Routes**
- `proxy_server.py` - all openai-compatible routes - `/v1/chat/completion`, `/v1/embedding` + model info routes - `/v1/models`, `/v1/model/info`, `/v1/model_group_info` routes.
- `health_endpoints/` - `/health`, `/health/liveliness`, `/health/readiness`
- `management_endpoints/key_management_endpoints.py` - all `/key/*` routes
- `management_endpoints/team_endpoints.py` - all `/team/*` routes
- `management_endpoints/internal_user_endpoints.py` - all `/user/*` routes
- `management_endpoints/ui_sso.py` - all `/sso/*` routes

---

## Contributing

We welcome contributions! This section explains how to get up and running quickly.

### Development Setup

LiteLLM uses [`uv`](https://docs.astral.sh/uv/) for dependency management. Install it first:

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install the development dependencies:

```shell
# Core library development
make install-dev

# Proxy server development (includes all optional features)
make install-proxy-dev

# Full local test environment (also generates the Prisma client)
make install-test-deps
```

### Running Tests

```shell
# Run all tests
make test

# Run only unit tests (fast — 4 parallel workers, no network)
make test-unit

# Run integration tests (requires API keys)
make test-integration

# Run a single test file
uv run pytest tests/test_litellm/test_completion.py -v

# Run a single test function
uv run pytest tests/test_litellm/test_completion.py::test_completion_openai -v
```

### Code Quality

Black formatting is **enforced in CI**. Always format before pushing:

```shell
# Format all code (required before every commit)
uv run black .

# Run all linters (Ruff, MyPy, Black, circular-import check)
make lint

# Run individual linters
make lint-ruff    # Ruff linting only
make lint-mypy    # MyPy type checking only
```

### Code Style Rules

- **Type hints required** on all public API functions and methods.
- **No inline imports** — all imports go at the top of the file. The only exception is breaking circular imports.
- **Use dict spread** (`{**original, "key": new_value}`) instead of `dict(obj)` + mutation.
- **Guard at resolution time** — raise immediately when a resolved optional value is empty if that empty value is an error condition.
- **Pydantic v2** for data validation throughout.
- **Async/await** patterns are used throughout; sync and async variants should both be implemented for public APIs.

### Submitting a Pull Request

1. Fork the repo and create a feature branch from `main`.
2. Make your changes and add tests in `tests/test_litellm/` (unit) or `tests/llm_translation/` (integration).
3. Run `uv run black .` and `make lint` — both must pass.
4. Run `make test-unit` — all unit tests must pass.
5. Open a PR with a clear description of what changed and why.

For bug reports, feature requests, and questions, please use the [GitHub issue tracker](https://github.com/BerriAI/litellm/issues/new/choose).
