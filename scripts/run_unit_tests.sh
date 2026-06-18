#!/usr/bin/env bash
# Convenience wrapper around the unit test suite.
# Usage: ./scripts/run_unit_tests.sh [optional pytest args]
#   e.g. ./scripts/run_unit_tests.sh -k test_completion -v
set -euo pipefail

echo "Running unit tests with 4 workers..."
uv run pytest tests/test_litellm/ -n 4 --tb=short "$@"
echo "Done."
