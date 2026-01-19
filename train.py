"""Training script for nanoGPT.

Run with: uv run train.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import sentencepiece as spm
import tiktoken
import torch

MAX_LINE_PREVIEW = 100
MAX_LINES = 1_000_000  # Use only 1 million lines
TRAIN_SPLIT = 0.9  # 90% train, 10% validation


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
    print(f"Total lines in file: {total_lines:,}")

    # Skip comment lines
    data_lines = [line for line in lines if not line.startswith("#")]

    # Limit to MAX_LINES
    data_lines = data_lines[:MAX_LINES]
    print(f"Using {len(data_lines):,} lines (limited to {MAX_LINES:,})")

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
    test_sentence = "Hello, world! This is a test."

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

    # 3. SentencePiece (using a temporary model trained on sample data)
    print("\n--- SentencePiece encoding ---")
    print("Note: Training a small SentencePiece model on sample data...")

    # Create a temporary sample file for training
    sample_path = Path("data/.sentencepiece_sample.txt")
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    with sample_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(data_lines[:10000]))  # Use first 10k lines for training

    # Train a small sentencepiece model
    model_prefix = "data/.sentencepiece_model"
    spm.SentencePieceTrainer.train(
        f"--input={sample_path} --model_prefix={model_prefix} --vocab_size=1000 "
        "--model_type=unigram --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 "
        "--normalization_rule_name=identity"
    )

    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")

    start = time.time()
    encoded_sp = sp.encode(test_sentence)
    sp_encode_time = time.time() - start
    print(f"Encoded: {encoded_sp}")
    print(f"Length: {len(encoded_sp)} tokens")
    print(f"Encoding time: {sp_encode_time * 1000:.4f}ms")

    start = time.time()
    decoded_sp = sp.decode(encoded_sp)
    sp_decode_time = time.time() - start
    print(f"Decoded: {decoded_sp}")
    print(f"Decoding time: {sp_decode_time * 1000:.4f}ms")

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
    print(
        f"SentencePiece: {len(encoded_sp)} tokens, "
        f"encode: {sp_encode_time * 1000:.4f}ms, decode: {sp_decode_time * 1000:.4f}ms"
    )
    print("\nCompression ratio (vs char-level):")
    print(f"  Tiktoken: {len(encoded_char) / len(encoded_tiktoken):.2f}x")
    print(f"  SentencePiece: {len(encoded_char) / len(encoded_sp):.2f}x")

    # Tokenize entire dataset with character-level tokenizer
    print(f"\n{'=' * 60}")
    print("TOKENIZING ENTIRE DATASET (CHARACTER-LEVEL)")
    print(f"{'=' * 60}")
    print(f"Total characters to tokenize: {total_chars:,}")

    start = time.time()
    encoded_dataset = encode(text)
    tokenize_time = time.time() - start

    print(f"Total tokens: {len(encoded_dataset):,}")
    print(f"Tokenization time: {tokenize_time:.2f}s")
    print(f"Throughput: {total_chars / tokenize_time / 1_000_000:.2f}M chars/sec")

    # Memory estimation
    memory_mb = sys.getsizeof(encoded_dataset) / (1024 * 1024)
    print(f"Memory usage (token array): {memory_mb:.2f} MB")

    # Convert to torch tensor
    print(f"\n{'=' * 60}")
    print("LOADING TO TORCH AND SPLITTING")
    print(f"{'=' * 60}")

    start = time.time()
    data = torch.tensor(encoded_dataset, dtype=torch.long)
    torch_load_time = time.time() - start

    print(f"Tensor shape: {data.shape}")
    print(f"Tensor dtype: {data.dtype}")
    print(f"Tensor device: {data.device}")
    torch_memory_mb = data.element_size() * data.nelement() / (1024 * 1024)
    print(f"Tensor memory: {torch_memory_mb:.2f} MB")
    print(f"Torch conversion time: {torch_load_time:.2f}s")

    # Split into train and validation
    n = len(data)
    train_size = int(n * TRAIN_SPLIT)
    train_data = data[:train_size]
    val_data = data[train_size:]

    print(f"\nSplit: {TRAIN_SPLIT * 100:.0f}% train, {(1 - TRAIN_SPLIT) * 100:.0f}% validation")
    print(f"Train size: {len(train_data):,} tokens")
    print(f"Validation size: {len(val_data):,} tokens")

    train_memory_mb = train_data.element_size() * train_data.nelement() / (1024 * 1024)
    val_memory_mb = val_data.element_size() * val_data.nelement() / (1024 * 1024)
    print(f"Train memory: {train_memory_mb:.2f} MB")
    print(f"Validation memory: {val_memory_mb:.2f} MB")

    # Show first 100 tokens
    print(f"\nFirst 100 train tokens: {train_data[:100].tolist()}")
    print(f"Decoded first 100 train tokens: {decode(train_data[:100].tolist())}")

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
