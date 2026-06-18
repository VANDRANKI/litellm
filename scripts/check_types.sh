#!/usr/bin/env bash
# Run MyPy on the litellm package with the project's configured settings.
# Usage: ./scripts/check_types.sh [optional extra mypy args]
set -euo pipefail

echo "Running MyPy type check..."
uv run mypy litellm/ "$@"
echo "MyPy check complete."
