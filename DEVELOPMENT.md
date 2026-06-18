# LiteLLM Development Guide

A practical reference for contributors working on the LiteLLM codebase.

## Quick Start

```bash
# Clone and install dev dependencies
git clone https://github.com/VANDRANKI/litellm.git
cd litellm
make install-dev
```

## Python Requirements

- **Minimum**: Python 3.9
- **Recommended**: Python 3.11+ (matches CI)
- We use `uv` for all dependency management — do **not** use bare `pip install`.

```bash
# Run any script through uv to guarantee the correct env
uv run python myscript.py
uv run pytest tests/path/to/test.py -v
```

## Type Hints

All public functions require type hints. Use `Optional[X]` for nullable
parameters, never bare `X = None` without annotating the type.

```python
# Good
def call_model(
    model: str,
    messages: list[dict],
    timeout: Optional[float] = None,
) -> ModelResponse:
    ...

# Bad — missing return type, bare Optional implied by default
def call_model(model, messages, timeout=None):
    ...
```

Run MyPy before opening a PR:
```bash
make lint-mypy
```

## Testing

| Command | What it runs |
|---|---|
| `make test-unit` | Fast unit tests in `tests/test_litellm/` (no network) |
| `make test-integration` | Integration tests requiring API keys |
| `uv run pytest tests/path/to/test.py -v` | Single file |

Unit tests must not make real API calls. Use `unittest.mock.patch` or
`pytest-mock` fixtures for external calls.

## Code Style

- **Formatter**: Black (`make format` or `uv run black .`)
- **Linter**: Ruff (`make lint-ruff`)
- **Type checker**: MyPy (`make lint-mypy`)

Black is enforced in CI — always format before pushing.

## Imports

Place all imports at the **top** of the file. Inline imports inside
functions are only allowed when necessary to break a circular import —
they must carry a comment explaining why.

## Error Handling

Map provider-specific exceptions to the closest OpenAI-compatible error
class (e.g., `litellm.exceptions.AuthenticationError`). Never swallow
exceptions silently — at minimum log them with `litellm._logging.logger`.

## Pre-commit Checklist

- [ ] `uv run black .` passes
- [ ] `make lint-ruff` passes
- [ ] `make lint-mypy` passes
- [ ] New public functions have docstrings and type hints
- [ ] Tests added for new functionality
- [ ] `make test-unit` passes locally
