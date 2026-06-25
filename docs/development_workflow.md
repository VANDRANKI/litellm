# Development Workflow Guide

This guide covers the day-to-day development workflow for LiteLLM contributors.

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Docker (for running the proxy with a database)

## Setting Up Your Environment

```bash
# Clone the repository
git clone https://github.com/BerriAI/litellm.git
cd litellm

# Install core development dependencies
make install-dev

# Install proxy development dependencies (needed for proxy server work)
make install-proxy-dev

# Generate the Prisma client (needed for DB-related work)
make install-test-deps
```

## Running the Proxy Server Locally

```bash
# Start with a minimal config
uv run litellm --config dev_config.yaml

# Enable debug logging
LITELLM_LOG=DEBUG uv run litellm --config dev_config.yaml

# Start with Docker (includes PostgreSQL)
docker compose up
```

## Running Tests

```bash
# Run all unit tests (parallel, fast)
make test-unit

# Run a specific test file
uv run pytest tests/test_litellm/test_utils.py -v

# Run a single test function
uv run pytest tests/test_litellm/test_utils.py::test_function_name -v

# Run integration tests for a specific provider
uv run pytest tests/llm_translation/test_openai.py -v
```

## Code Quality

Always run these before committing:

```bash
# Format code (required by CI)
uv run black .

# Run linting suite
make lint

# Type checking only
make lint-mypy
```

## Common Development Scenarios

### Adding a New Provider

1. Create `litellm/llms/<provider_name>/` directory
2. Implement transformation functions (input → provider format, provider response → OpenAI format)
3. Register the provider in `litellm/utils.py`
4. Add tests in `tests/llm_translation/test_<provider>.py`
5. Add provider config to `litellm/main.py`

### Debugging Provider Errors

```python
import litellm
litellm.set_verbose = True  # Enables detailed request/response logging

response = litellm.completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}]
)
```

### Working on the Proxy

The proxy server is a FastAPI application. For hot-reload during development:

```bash
cd litellm/proxy
uv run uvicorn proxy_server:app --reload --port 4000
```

## Debugging Tips

- Use `LITELLM_LOG=DEBUG` to see full request/response payloads
- Set `litellm.drop_params = True` to ignore unsupported parameters during testing
- Use the setup wizard (`uv run python -m litellm.setup_wizard`) for quick provider validation
- Check `litellm/_logging.py` for logging configuration options

## Making a Pull Request

1. Create a feature branch: `git checkout -b feat/my-feature`
2. Make changes and add tests
3. Run `uv run black .` and `make lint`
4. Ensure `make test-unit` passes
5. Push and open a PR following the template in `.github/pull_request_template.md`
