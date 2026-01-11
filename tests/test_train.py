"""Tests for the training module."""

from pathlib import Path


def test_data_file_exists() -> None:
    """Test that the training data file exists."""
    data_path = Path("data/composed-stories-all.txt")
    assert data_path.exists(), f"Data file not found: {data_path}"
