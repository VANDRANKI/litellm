# Development Workflow Guide

This guide covers common development workflows for contributing to LiteLLM.

## Setting Up Your Environment

```bash
# Clone the repository
git clone https://github.com/BerriAI/litellm
cd litellm

# Install development dependencies
make install-dev

# Install proxy dependencies (for proxy server development)
make install-proxy-dev
```

## Running Tests

```bash
# Run all unit tests
make test-unit

# Run a specific test file
uv run pytest tests/test_litellm/test_completion.py -v

# Run a specific test function
uv run pytest tests/test_litellm/test_completion.py::test_completion_openai -v
```

## Code Quality Checks

Always run these before creating a PR:

```bash
# Format code
uv run black .

# Run all linters
make lint

# Run type checking only
make lint-mypy
```

## Adding a New LLM Provider

1. Create a directory under `litellm/llms/<provider_name>/`
2. Implement the transformation functions for request/response formatting
3. Add the provider to `litellm/main.py` routing logic
4. Add tests in `tests/llm_translation/test_<provider_name>.py`
5. Add documentation in `docs/providers/<provider_name>.md`

## Common Pitfalls

- **Never add inline imports** inside functions/methods unless strictly required for circular import avoidance
- **Use Pydantic v2 models** for all data validation — do not use raw dicts for API request/response shapes
- **Always handle streaming** in both sync and async paths when implementing a new provider
- **Test error mapping** — ensure provider-specific errors are mapped to OpenAI-compatible error codes

## Proxy Server Development

The proxy server is a FastAPI application under `litellm/proxy/`. To run it locally:

```bash
litellm --config dev_config.yaml --detailed_debug
```

Or with Docker:

```bash
docker compose up
```

## Database Schema Changes

When making Prisma schema changes:

1. Edit `schema.prisma`
2. Run `prisma migrate dev --name <description>` to generate a migration
3. Ensure the migration is also reflected in `litellm-proxy-extras/`
4. Test with both PostgreSQL and SQLite
