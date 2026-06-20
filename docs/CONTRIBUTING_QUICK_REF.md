# Quick Reference for LiteLLM Contributors

## Local setup in 60 seconds

```bash
git clone https://github.com/BerriAI/litellm.git
cd litellm
make install-dev
source .venv/bin/activate
make test-unit
```

## Before every commit

```bash
uv run black .          # format
make lint               # ruff + mypy
make test-unit          # unit tests only (fast)
```

## Adding a new provider

1. Create `litellm/llms/<provider>/` directory.
2. Add `completion.py` (sync) and `async_completion.py`.
3. Register in `litellm/utils.py` `get_llm_provider()`.
4. Add integration tests in `tests/llm_translation/test_<provider>.py`.
5. Update the provider table in `README.md`.

## Debugging provider calls

```python
import litellm
litellm.set_verbose = True  # log full request/response
response = litellm.completion(model="openai/gpt-4o", messages=[...])
```

## Environment variables for testing

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export LITELLM_LOG=DEBUG   # verbose logging
```
