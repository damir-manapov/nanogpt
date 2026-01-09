"""Tests for the model module."""

from __future__ import annotations

import pytest
import torch

from src.model import GPT, FeedForward, MultiHeadAttention, TransformerBlock


def test_multi_head_attention_shape() -> None:
    """Test that multi-head attention produces correct output shape."""
    d_model = 512
    num_heads = 8
    batch_size = 2
    seq_len = 10

    attn = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(batch_size, seq_len, d_model)

    output = attn(x, x, x)

    assert output.shape == (batch_size, seq_len, d_model)


def test_multi_head_attention_invalid_dimensions() -> None:
    """Test that invalid dimensions raise ValueError."""
    with pytest.raises(ValueError, match="d_model must be divisible by num_heads"):
        MultiHeadAttention(d_model=512, num_heads=7)


def test_feed_forward_shape() -> None:
    """Test that feed-forward network produces correct output shape."""
    d_model = 512
    d_ff = 2048
    batch_size = 2
    seq_len = 10

    ff = FeedForward(d_model, d_ff)
    x = torch.randn(batch_size, seq_len, d_model)

    output = ff(x)

    assert output.shape == (batch_size, seq_len, d_model)


def test_transformer_block_shape() -> None:
    """Test that transformer block produces correct output shape."""
    d_model = 512
    num_heads = 8
    d_ff = 2048
    batch_size = 2
    seq_len = 10

    block = TransformerBlock(d_model, num_heads, d_ff)
    x = torch.randn(batch_size, seq_len, d_model)

    output = block(x)

    assert output.shape == (batch_size, seq_len, d_model)


def test_gpt_forward_shape() -> None:
    """Test that GPT forward pass produces correct output shape."""
    vocab_size = 1000
    d_model = 256
    num_heads = 4
    num_layers = 2
    batch_size = 2
    seq_len = 10

    model = GPT(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
    )

    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = model(x)

    assert output.shape == (batch_size, seq_len, vocab_size)


def test_gpt_generate() -> None:
    """Test that GPT can generate tokens."""
    vocab_size = 1000
    d_model = 256
    num_heads = 4
    num_layers = 2
    batch_size = 1
    seq_len = 5
    max_new_tokens = 10

    model = GPT(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
    )
    model.eval()

    x = torch.randint(0, vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        output = model.generate(x, max_new_tokens=max_new_tokens)

    assert output.shape == (batch_size, seq_len + max_new_tokens)
    assert torch.all((output >= 0) & (output < vocab_size))


def test_gpt_generate_with_top_k() -> None:
    """Test that GPT can generate tokens with top-k sampling."""
    vocab_size = 1000
    d_model = 256
    num_heads = 4
    num_layers = 2
    batch_size = 1
    seq_len = 5
    max_new_tokens = 5
    top_k = 10

    model = GPT(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
    )
    model.eval()

    x = torch.randint(0, vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        output = model.generate(x, max_new_tokens=max_new_tokens, top_k=top_k)

    assert output.shape == (batch_size, seq_len + max_new_tokens)
