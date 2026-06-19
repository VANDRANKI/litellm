#!/usr/bin/env bash
# lint_check.sh — Run Ruff and MyPy checks and exit non-zero if either fails.
#
# Usage:
#   ./scripts/lint_check.sh
#
# Requirements:
#   uv must be installed (https://github.com/astral-sh/uv)
#   Run `make install-dev` first to set up the environment.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)"
cd "$ROOT"

RUFF_EXIT=0
MYPY_EXIT=0

echo "==> Running Ruff linter on litellm/..."
uv run ruff check litellm/ || RUFF_EXIT=$?

echo ""
echo "==> Running MyPy type checker on litellm/..."
uv run mypy litellm/ --ignore-missing-imports || MYPY_EXIT=$?

echo ""
if [ "$RUFF_EXIT" -ne 0 ] || [ "$MYPY_EXIT" -ne 0 ]; then
    echo "FAIL: One or more lint checks did not pass."
    echo "  Ruff exit code : $RUFF_EXIT"
    echo "  MyPy exit code : $MYPY_EXIT"
    exit 1
fi

echo "OK: All lint checks passed."
