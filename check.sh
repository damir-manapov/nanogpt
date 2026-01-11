#!/usr/bin/env bash
set -euo pipefail

echo "=== Formatting with ruff ==="
uv run ruff format .

echo ""
echo "=== Linting with ruff ==="
uv run ruff check . --fix

echo ""
echo "=== Type checking would go here ==="
echo "Note: Consider adding mypy or pyright for type checking"

echo ""
echo "=== Running tests ==="
uv run pytest

echo ""
echo "All checks passed!"
