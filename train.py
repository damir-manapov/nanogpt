"""Training script for nanoGPT.

Run with: uv run train.py
"""

from __future__ import annotations

import time
from pathlib import Path

import tiktoken

MAX_LINE_PREVIEW = 100


def main() -> None:  # noqa: PLR0915
    """Load training data and display statistics."""
    start_time_total = time.time()

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

    # Calculate vocabulary
    text = "".join(data_lines)
    vocab = sorted(set(text))
    vocab_size = len(vocab)
    print(f"\nVocabulary size: {vocab_size}")
    print(f"Vocabulary: {''.join(vocab)}")

    # Create character mappings
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = dict(enumerate(vocab))

    # Encode and decode functions
    def encode(s: str) -> list[int]:
        """Convert string to list of integers."""
        return [stoi[c] for c in s]

    def decode(tokens: list[int]) -> str:
        """Convert list of integers to string."""
        return "".join([itos[i] for i in tokens])

    # Test encoding and decoding
    test_sentence = "😎 Hello, world! This is a test."

    # Character-level encoding
    print(f"\n{'=' * 60}")
    print("ENCODING COMPARISON")
    print(f"{'=' * 60}")
    print(f"\nTest sentence: {test_sentence}")

    # 1. Character-level
    print("\n--- Character-level encoding ---")
    start = time.time()
    encoded_char = encode(test_sentence)
    char_encode_time = time.time() - start
    print(f"Encoded: {encoded_char}")
    print(f"Length: {len(encoded_char)} tokens")
    print(f"Encoding time: {char_encode_time * 1000:.4f}ms")

    start = time.time()
    decoded_char = decode(encoded_char)
    char_decode_time = time.time() - start
    print(f"Decoded: {decoded_char}")
    print(f"Decoding time: {char_decode_time * 1000:.4f}ms")

    # 2. Tiktoken (GPT-2 encoding)
    print("\n--- Tiktoken (GPT-2) encoding ---")
    tiktoken_enc = tiktoken.get_encoding("gpt2")

    start = time.time()
    encoded_tiktoken = tiktoken_enc.encode(test_sentence)
    tiktoken_encode_time = time.time() - start
    print(f"Encoded: {encoded_tiktoken}")
    print(f"Length: {len(encoded_tiktoken)} tokens")
    print(f"Encoding time: {tiktoken_encode_time * 1000:.4f}ms")

    start = time.time()
    decoded_tiktoken = tiktoken_enc.decode(encoded_tiktoken)
    tiktoken_decode_time = time.time() - start
    print(f"Decoded: {decoded_tiktoken}")
    print(f"Decoding time: {tiktoken_decode_time * 1000:.4f}ms")

    # Performance comparison
    print(f"\n{'=' * 60}")
    print("PERFORMANCE SUMMARY")
    print(f"{'=' * 60}")
    print(
        f"Character-level: {len(encoded_char)} tokens, "
        f"encode: {char_encode_time * 1000:.4f}ms, decode: {char_decode_time * 1000:.4f}ms"
    )
    print(
        f"Tiktoken (GPT-2): {len(encoded_tiktoken)} tokens, "
        f"encode: {tiktoken_encode_time * 1000:.4f}ms, decode: {tiktoken_decode_time * 1000:.4f}ms"
    )
    print("\nCompression ratio (vs char-level):")
    print(f"  Tiktoken: {len(encoded_char) / len(encoded_tiktoken):.2f}x")

    # Show first 5 lines
    print(f"\n{'=' * 60}")
    print("FIRST 5 LINES")
    print(f"{'=' * 60}")
    for i, line in enumerate(data_lines[:5], 1):
        print(
            f"{i}. {line[:MAX_LINE_PREVIEW].strip()}{'...' if len(line) > MAX_LINE_PREVIEW else ''}"
        )

    total_time = time.time() - start_time_total
    print(f"\nTotal execution time: {total_time:.2f}s")


if __name__ == "__main__":
    main()
