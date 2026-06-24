# Development Guide

This guide covers the essentials for contributing to LiteLLM.

## Prerequisites

- Python 3.9+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Docker (optional, for proxy integration tests)

## Setup

```bash
# Clone the repo
git clone https://github.com/BerriAI/litellm
cd litellm

# Install core dev dependencies
make install-dev

# Install proxy dev dependencies (needed for proxy tests)
make install-proxy-dev

# Generate the Prisma client and install full test environment
make install-test-deps
```

## Running Tests

```bash
# All unit tests with 4 parallel workers
make test-unit

# A single test file
uv run pytest tests/test_litellm/test_completion.py -v

# A single test function
uv run pytest tests/test_litellm/test_completion.py::test_completion_openai -v

# Integration tests (requires provider API keys)
make test-integration
```

## Code Style

LiteLLM uses **Black** for formatting, **Ruff** for linting, and **MyPy** for
type checking. CI enforces all three.

```bash
# Format code (required before every commit)
uv run black .

# Lint
make lint-ruff

# Type check
make lint-mypy

# Run all lint checks at once
make lint
```

### Key conventions

- All imports at module level — avoid inline imports inside functions unless
  strictly required to break a circular dependency.
- Use `{**original, "key": new_value}` dict spreads instead of mutating copies.
- Raise immediately when a resolved value is empty; don't pass empty strings
  downstream.
- No bare `except:` — always catch specific exceptions.

## Adding a New LLM Provider

1. Create `litellm/llms/<provider>/` with at minimum:
   - `__init__.py`
   - `chat/transformation.py` — request/response mapping
   - `chat/handler.py` — sync + async HTTP calls
2. Register the provider in `litellm/utils.py` (model cost map and provider list).
3. Add integration tests under `tests/llm_translation/test_<provider>.py`.
4. Update `litellm/main.py` to route calls to the new handler.

See `litellm/llms/openai/` as a reference implementation.

## Proxy Server Development

The proxy is a FastAPI application at `litellm/proxy/proxy_server.py`.

```bash
# Start the proxy locally
uv run python -m litellm.proxy.proxy_server --config path/to/config.yaml

# Run proxy unit tests
uv run pytest tests/proxy_unit_tests/ -v
```

### Database

The proxy uses Prisma ORM with PostgreSQL (production) or SQLite (dev).

```bash
# Apply schema migrations
prisma migrate deploy

# Generate client after schema changes
prisma generate
```

Always use `prisma_client.db.<model>` methods — avoid raw SQL.

## Pre-Commit Checklist

- [ ] `uv run black .` passes with no changes
- [ ] `make lint-ruff` reports no errors
- [ ] `make test-unit` passes
- [ ] New provider or feature includes at least one test in `tests/test_litellm/`
- [ ] PR description includes test steps and expected behavior
