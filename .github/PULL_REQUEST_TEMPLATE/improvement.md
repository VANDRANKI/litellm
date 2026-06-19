## What changed

<!-- Describe the specific change and which files were modified -->

## Why

<!-- Explain the motivation. Link any relevant issue: Fixes #123 -->

## How to test

- [ ] Run `make test-unit` — all existing tests pass
- [ ] If adding a new provider feature, add a test in `tests/llm_translation/`
- [ ] For proxy changes, verify with `tests/proxy_unit_tests/`

## Checklist

- [ ] Added or updated docstrings for all public functions
- [ ] Added type hints for all new parameters
- [ ] Ran `uv run black .` to format code
- [ ] No inline imports introduced (all imports at module top)
