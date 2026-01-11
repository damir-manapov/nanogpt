#!/usr/bin/env bash
set -euo pipefail

echo "========================================"
echo "Running check.sh"
echo "========================================"
./check.sh

echo ""
echo "========================================"
echo "Running health.sh"
echo "========================================"
./health.sh

echo ""
echo "========================================"
echo "All checks completed successfully!"
echo "========================================"
