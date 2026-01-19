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
import torch.nn as nn
from torch.nn import functional as F

MAX_LINE_PREVIEW = 100
MAX_LINES = 1_000_000  # Use only 1 million lines
TRAIN_SPLIT = 0.9  # 90% train, 10% validation
BLOCK_SIZE = 8  # Context length
BATCH_SIZE = 32  # Number of sequences per batch

# Seed for reproducibility
torch.manual_seed(1337)


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

    # --- STEP 1: Show first 5 lines (raw data preview) ---
    print(f"\n{'=' * 60}")
    print("STEP 1: FIRST 5 LINES (RAW DATA)")
    print(f"{'=' * 60}")
    for i, line in enumerate(data_lines[:5], 1):
        print(
            f"{i}. {line[:MAX_LINE_PREVIEW].strip()}{'...' if len(line) > MAX_LINE_PREVIEW else ''}"
        )

    # --- STEP 2: Show first 100 tokens (tokenized preview) ---
    print(f"\n{'=' * 60}")
    print("STEP 2: FIRST 100 TRAIN TOKENS")
    print(f"{'=' * 60}")
    print(f"Tokens: {train_data[:100].tolist()}")
    print(f"Decoded: {decode(train_data[:100].tolist())}")

    # --- STEP 3: Single sequence example (simple x -> y) ---
    print(f"\n{'=' * 60}")
    print(f"STEP 3: SINGLE SEQUENCE EXAMPLE (block_size={BLOCK_SIZE})")
    print(f"{'=' * 60}")

    # Get first chunk (BLOCK_SIZE + 1 tokens needed to create x and y)
    chunk = train_data[: BLOCK_SIZE + 1]
    x_single = chunk[:BLOCK_SIZE]  # inputs: BLOCK_SIZE tokens
    y_single = chunk[1 : BLOCK_SIZE + 1]  # targets: shifted by 1

    print(f"\nx (inputs):  {x_single.tolist()}")
    print(f"y (targets): {y_single.tolist()}")
    print(f"\nDecoded x: {decode(x_single.tolist())!r}")
    print(f"Decoded y: {decode(y_single.tolist())!r}")

    print("\nFor each position, predict next token based on context:")
    print("context -> target\n")

    for t in range(BLOCK_SIZE):
        context = x_single[: t + 1]
        target = y_single[t]
        context_str = decode(context.tolist())
        target_str = decode([target.item()])
        print(f"  {context.tolist()!s:40} -> {target.item()}")
        print(f"  {context_str!r:40} -> {target_str!r}")
        print()

    # --- STEP 4: Batched data (generalized approach) ---
    print(f"\n{'=' * 60}")
    print(f"STEP 4: BATCHED DATA (batch_size={BATCH_SIZE}, block_size={BLOCK_SIZE})")
    print(f"{'=' * 60}")

    # Function to get a batch of data
    def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a random batch of training or validation data.

        Args:
            split: Either 'train' or 'val'

        Returns:
            Tuple of (x, y) tensors, each of shape (batch_size, block_size)
        """
        data_split = train_data if split == "train" else val_data
        # Generate random starting indices for each sequence in the batch
        ix = torch.randint(len(data_split) - BLOCK_SIZE, (BATCH_SIZE,))
        # Stack the sequences into batches
        x = torch.stack([data_split[i : i + BLOCK_SIZE] for i in ix])
        y = torch.stack([data_split[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
        return x, y

    xb, yb = get_batch("train")
    print(f"\nBatch shapes: x={xb.shape}, y={yb.shape}")

    print("\n--- Train batch ---")
    for b in range(BATCH_SIZE):
        print(f"\nBatch {b}:")
        print(f"  x: {xb[b].tolist()}")
        print(f"  y: {yb[b].tolist()}")
        print(f"  Decoded x: {decode(xb[b].tolist())!r}")
        print(f"  Decoded y: {decode(yb[b].tolist())!r}")

    # --- STEP 5: Bigram Language Model ---
    print(f"\n{'=' * 60}")
    print("STEP 5: BIGRAM LANGUAGE MODEL")
    print(f"{'=' * 60}")

    class BigramLanguageModel(nn.Module):
        """Simple bigram language model.

        Each token directly looks up the logits for the next token
        from an embedding table. No context is used beyond the
        immediately preceding token.
        """

        def __init__(self, vocab_size: int) -> None:
            super().__init__()
            # Each token directly reads the logits for the next token
            self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

        def forward(
            self, idx: torch.Tensor, targets: torch.Tensor | None = None
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            """Forward pass.

            Args:
                idx: Input token indices, shape (B, T)
                targets: Target token indices, shape (B, T), optional

            Returns:
                logits: Shape (B, T, vocab_size)
                loss: Cross-entropy loss if targets provided, else None
            """
            # idx shape: (B, T), logits shape: (B, T, vocab_size)
            logits = self.token_embedding_table(idx)

            if targets is None:
                loss = None
            else:
                # Reshape for cross_entropy: (B*T, vocab_size) and (B*T,)
                b, t, c = logits.shape
                logits_flat = logits.view(b * t, c)
                targets_flat = targets.view(b * t)
                loss = F.cross_entropy(logits_flat, targets_flat)

            return logits, loss

        def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
            """Generate new tokens.

            Args:
                idx: Starting context, shape (B, T)
                max_new_tokens: Number of tokens to generate

            Returns:
                Extended sequence, shape (B, T + max_new_tokens)
            """
            for _ in range(max_new_tokens):
                # Get predictions
                logits, _ = self(idx)
                # Focus only on the last time step
                logits = logits[:, -1, :]  # (B, vocab_size)
                # Apply softmax to get probabilities
                probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
                # Sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
                # Append to the running sequence
                idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
            return idx

    # Create model
    model = BigramLanguageModel(vocab_size)
    print(f"Model created with vocab_size={vocab_size}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    logits, loss = model(xb, yb)
    print(f"\nForward pass:")
    print(f"  Input shape: {xb.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Expected loss (random): {-torch.log(torch.tensor(1.0 / vocab_size)).item():.4f}")

    # Generate some text (untrained model - will be random)
    print("\nGeneration (untrained model):")
    context = torch.zeros((1, 1), dtype=torch.long)  # Start with token 0
    generated = model.generate(context, max_new_tokens=100)
    generated_text = decode(generated[0].tolist())
    print(f"  {generated_text!r}")

    # --- STEP 6: Optimizer ---
    print(f"\n{'=' * 60}")
    print("STEP 6: OPTIMIZER")
    print(f"{'=' * 60}")

    # Create optimizer
    learning_rate = 1e-3
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print(f"Optimizer: AdamW")
    print(f"Learning rate: {learning_rate}")
    print(f"Parameters to optimize: {sum(p.numel() for p in model.parameters()):,}")

    # Training loop
    n_steps = 10000
    print(f"\n--- Training for {n_steps} steps ---")

    for step in range(n_steps):
        # Get batch
        xb, yb = get_batch("train")

        # Forward pass
        logits, loss = model(xb, yb)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Print progress
        if step % 1000 == 0 or step == n_steps - 1:
            print(f"Step {step:5d}: loss = {loss.item():.4f}")

    # Check final loss on a new batch
    xb, yb = get_batch("train")
    _, final_loss = model(xb, yb)
    print(f"\nFinal loss: {final_loss.item():.4f}")

    # Generate text from trained model
    print("\nGeneration (trained model):")
    context = torch.zeros((1, 1), dtype=torch.long)  # Start with token 0
    generated = model.generate(context, max_new_tokens=200)
    generated_text = decode(generated[0].tolist())
    print(f"  {generated_text!r}")

    total_time = time.time() - start_time_total
    print(f"\nTotal execution time: {total_time:.2f}s")


if __name__ == "__main__":
    main()
