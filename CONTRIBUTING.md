# Contributing to LiteLLM

Thank you for contributing! This guide covers the essential workflow for getting started.

## Quick Start

### 1. Fork & Clone

```bash
git clone https://github.com/<your-username>/litellm.git
cd litellm
```

### 2. Set Up Environment with `uv`

We use [`uv`](https://github.com/astral-sh/uv) for fast, reproducible Python environments.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install core dev dependencies
make install-dev

# Install proxy dev dependencies (full feature set)
make install-proxy-dev

# Install test dependencies and generate Prisma client
make install-test-deps
```

### 3. Make Your Changes

Create a feature branch:

```bash
git checkout -b feat/your-feature-name
```

### 4. Code Style Requirements

Before committing, ensure your code follows these rules:

**Formatting (required — enforced in CI):**
```bash
uv run black .
```

**Linting:**
```bash
make lint-ruff     # Ruff linter
make lint-mypy     # MyPy type checking
make lint          # All linters (Ruff + MyPy + Black + circular imports)
```

**Key style rules:**

- **No inline imports** — all imports belong at the top of the file (module level). Inline imports inside functions make dependencies hard to trace. The only exception is avoiding circular imports where truly necessary.
- **Use dict spread for immutable copies** — prefer `{**original, "key": new_value}` over `dict(obj)` followed by mutation. The spread produces the final dict in one step.
- **Guard at resolution time** — when resolving an optional value via a fallback chain (e.g. `a or b or ""`), raise immediately if the empty result is an error condition. Don't pass empty strings downstream.
- **Type hints required** — all public API functions must have complete type hints including return types.
- **FastAPI parameter declarations** — mark required query/form params with `= Query(...)` / `= Form(...)` explicitly.

### 5. Testing

```bash
# Run all tests
make test

# Run unit tests only (faster, 4 parallel workers)
make test-unit

# Run a specific test file
uv run pytest tests/path/to/test_file.py -v

# Run a specific test function
uv run pytest tests/path/to/test_file.py::test_function -v
```

**Testing guidelines:**
- Add tests in `tests/litellm/` for all new features or bug fixes
- Keep monkeypatch stubs in sync with real function signatures
- Test all branches of name→ID resolution (resolves+allowed, resolves+not-allowed, doesn't resolve)

### 6. Submit a Pull Request

- Reference any related issue: `Closes #123`
- Ensure `make test-unit` passes
- Ensure `make lint` passes
- The PR template in `.github/pull_request_template.md` provides the required structure

## Project Layout

```
litellm/
├── litellm/          # Core library
│   ├── main.py       # Core completion() entry point
│   ├── llms/         # Provider implementations
│   ├── router.py     # Load balancing & fallback logic
│   ├── types/        # Pydantic models & type hints
│   ├── integrations/ # Third-party observability, caching, logging
│   └── proxy/        # FastAPI proxy server
├── tests/
│   ├── test_litellm/ # Unit tests
│   └── llm_translation/ # Integration tests
└── scripts/          # Developer helper scripts
```

## Getting Help

- Open an issue for bugs or feature requests using the templates in `.github/ISSUE_TEMPLATE/`
- Join the [Discord](https://discord.gg/wuPM9dRgDw) for questions
- Check existing issues and PRs before opening a duplicate
