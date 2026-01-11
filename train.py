"""Training script for nanoGPT.

Run with: uv run train.py
"""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    """Load training data and display statistics."""
    data_path = Path("data/composed-stories-all.txt")

    print(f"Loading data from {data_path}...")

    # Count lines
    with data_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    total_lines = len(lines)
    print(f"Total lines: {total_lines:,}")

    # Skip comment lines
    data_lines = [line for line in lines if not line.startswith("#")]
    story_count = len(data_lines)

    print(f"Stories (excluding comments): {story_count:,}")

    # Calculate total characters
    total_chars = sum(len(line) for line in data_lines)
    print(f"Total characters: {total_chars:,}")

    # Show sample
    print("\nFirst story (first 200 chars):")
    if data_lines:
        print(data_lines[0][:200].strip() + "...")


if __name__ == "__main__":
    main()
