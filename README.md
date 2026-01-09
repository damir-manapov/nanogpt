# nanogpt

A simple reimplementation of ChatGPT focusing on core transformer architecture and training principles.

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management.

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Development

### Running checks

```bash
# Format, lint, and test
./check.sh

# Check dependencies and security
./health.sh

# Run all checks
./all-checks.sh
```

### Testing

```bash
pytest
```

## Project Structure

```
src/          # Source code
tests/        # Tests
```
