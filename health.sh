#!/usr/bin/env bash
set -euo pipefail

echo "=== Checking for secrets with gitleaks ==="
if ! command -v gitleaks &> /dev/null; then
  echo "Installing gitleaks..."
  TEMP_DIR=$(mktemp -d)
  cd "$TEMP_DIR"

  ARCH=$(uname -m)
  if [ "$ARCH" = "x86_64" ]; then
    GITLEAKS_ARCH="x64"
  elif [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    GITLEAKS_ARCH="arm64"
  else
    echo "Unsupported architecture: $ARCH"
    exit 1
  fi

  curl -sSfL "https://github.com/gitleaks/gitleaks/releases/latest/download/gitleaks_*_linux_${GITLEAKS_ARCH}.tar.gz" | tar -xz
  sudo mv gitleaks /usr/local/bin/
  cd - > /dev/null
  rm -rf "$TEMP_DIR"
  echo "Gitleaks installed successfully"
fi

gitleaks detect --source . -v

echo ""
echo "=== Checking for dependency vulnerabilities with pip-audit ==="
if ! command -v pip-audit &> /dev/null; then
  echo "Installing pip-audit..."
  uv pip install pip-audit
fi

uv run pip-audit

echo ""
echo "=== Checking for outdated dependencies ==="
./renovate-check.sh

echo ""
echo "All health checks passed!"
