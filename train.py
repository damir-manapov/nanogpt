"""Training script for nanoGPT.

Run with: uv run train.py
"""

from __future__ import annotations

import time
from pathlib import Path


def main() -> None:
    """Load training data and display statistics."""
    data_path = Path("data/composed-stories-all.txt")

    print(f"Loading data from {data_path}...")

    start_time = time.time()

    # Count lines
    with data_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    load_time = time.time() - start_time

    total_lines = len(lines)
    print(f"Total lines: {total_lines:,}")

    # Skip comment lines
    data_lines = [line for line in lines if not line.startswith("#")]

    # Calculate total characters
    total_chars = sum(len(line) for line in data_lines)
    print(f"Total characters: {total_chars:,}")
    print(f"Load time: {load_time:.2f}s")


if __name__ == "__main__":
    main()
