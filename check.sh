#!/usr/bin/env bash
set -euo pipefail

echo "=== Formatting with ruff ==="
ruff format .

echo ""
echo "=== Linting with ruff ==="
ruff check . --fix

echo ""
echo "=== Type checking would go here ==="
echo "Note: Consider adding mypy or pyright for type checking"

echo ""
echo "=== Running tests ==="
pytest

echo ""
echo "✅ All checks passed!"
