"""Transformer model implementation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from torch import Tensor


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        """Initialize multi-head attention.

        Args:
            d_model: Model dimensionality
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        if d_model % num_heads != 0:
            msg = "d_model must be divisible by num_heads"
            raise ValueError(msg)

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            query: Query tensor [batch, seq_len, d_model]
            key: Key tensor [batch, seq_len, d_model]
            value: Value tensor [batch, seq_len, d_model]
            mask: Optional attention mask

        Returns:
            Attended values [batch, seq_len, d_model]
        """
        batch_size = query.size(0)

        # Linear projections
        q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # Apply attention to values
        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.out(out)


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        """Initialize feed-forward network.

        Args:
            d_model: Model dimensionality
            d_ff: Hidden layer dimensionality
            dropout: Dropout probability
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Transformer decoder block."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        """Initialize transformer block.

        Args:
            d_model: Model dimensionality
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimensionality
            dropout: Dropout probability
        """
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional attention mask

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        # Self-attention with residual connection
        attn_out = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward with residual connection
        ff_out = self.ff(x)
        return self.norm2(x + self.dropout(ff_out))


class GPT(nn.Module):
    """Simple GPT model."""

    def __init__(  # noqa: PLR0913
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        """Initialize GPT model.

        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimensionality
            num_heads: Number of attention heads
            num_layers: Number of transformer blocks
            d_ff: Feed-forward hidden dimensionality
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Forward pass.

        Args:
            x: Input token indices [batch, seq_len]
            mask: Optional attention mask

        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        _batch_size, seq_len = x.size()

        # Embeddings
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)

        x = self.dropout(tok_emb + pos_emb)

        # Transformer blocks
        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        return self.fc_out(x)

    def generate(
        self,
        idx: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> Tensor:
        """Generate tokens autoregressively.

        Args:
            idx: Starting token indices [batch, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: If set, only sample from top k tokens

        Returns:
            Generated token indices [batch, seq_len + max_new_tokens]
        """
        for _ in range(max_new_tokens):
            # Get predictions
            logits = self(idx)
            logits = logits[:, -1, :] / temperature

            # Optionally crop to top k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Sample
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
