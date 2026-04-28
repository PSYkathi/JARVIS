import torch
import torch.nn as nn

# ============================================
# TRANSFORMER COMPONENTS FROM SCRATCH
# 1. Layer Normalization
# 2. Feed-Forward Network
# 3. Residual Connection
# ============================================

class LayerNorm(nn.Module):
    """
    Normalizes each token's values to have
    mean=0 and std=1, then scales and shifts.
    Keeps training stable.
    """
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps   = eps
        self.scale = nn.Parameter(torch.ones(d_model))
        self.shift = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std  = x.std(dim=-1, keepdim=True)
        normalized = (x - mean) / (std + self.eps)
        return self.scale * normalized + self.shift


class FeedForward(nn.Module):
    """
    Two linear layers with GELU activation.
    Applied to each token independently.
    Expands then contracts: d_model -> 4*d_model -> d_model
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.layer1  = nn.Linear(d_model, 4 * d_model)
        self.layer2  = nn.Linear(4 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def gelu(self, x):
        # GELU activation - smoother than ReLU
        return 0.5 * x * (1.0 + torch.tanh(
            (2.0 / torch.pi) ** 0.5 *
            (x + 0.044715 * x ** 3)
        ))

    def forward(self, x):
        x = self.gelu(self.layer1(x))
        x = self.dropout(x)
        x = self.layer2(x)
        return x


class TransformerBlock(nn.Module):
    """
    One complete Transformer block:
    - Multi-Head Attention
    - Add & Norm (residual connection)
    - Feed-Forward Network
    - Add & Norm (residual connection)

    Residual connection: output = LayerNorm(x + SubLayer(x))
    This lets gradients flow freely during training.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        from attention import MultiHeadAttention
        self.attention  = MultiHeadAttention(d_model, num_heads)
        self.ff         = FeedForward(d_model, dropout)
        self.norm1      = LayerNorm(d_model)
        self.norm2      = LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Step 1: Self-attention + residual connection
        attended, weights = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attended))

        # Step 2: Feed-forward + residual connection
        fed = self.ff(x)
        x = self.norm2(x + self.dropout(fed))

        return x, weights


# --- TEST ---
if __name__ == "__main__":
    print("=== Transformer Components from Scratch ===\n")

    d_model   = 64
    num_heads = 4
    batch     = 2
    seq_len   = 10

    # Test LayerNorm
    print("1. Layer Normalization:")
    ln = LayerNorm(d_model)
    x  = torch.randn(batch, seq_len, d_model) * 5 + 3
    print(f"   Input  mean: {x.mean().item():.4f}, std: {x.std().item():.4f}")
    out = ln(x)
    print(f"   Output mean: {out.mean().item():.4f}, std: {out.std().item():.4f}")
    print(f"   (Mean near 0, std near 1 - normalized!) ✅\n")

    # Test FeedForward
    print("2. Feed-Forward Network:")
    ff  = FeedForward(d_model)
    x2  = torch.randn(batch, seq_len, d_model)
    out2 = ff(x2)
    print(f"   Input shape:  {x2.shape}")
    print(f"   Output shape: {out2.shape}")
    print(f"   (Shape preserved - correct!) ✅\n")

    # Test full Transformer Block
    print("3. Full Transformer Block:")
    block = TransformerBlock(d_model=64, num_heads=4)
    x3    = torch.randn(batch, seq_len, d_model)
    out3, w = block(x3)
    print(f"   Input shape:   {x3.shape}")
    print(f"   Output shape:  {out3.shape}")
    print(f"   Weights shape: {w.shape}")

    # Count parameters
    total = sum(p.numel() for p in block.parameters())
    print(f"   Parameters:    {total:,}")
    print(f"   One block working! ✅\n")

    print("Stack N of these blocks = the full Transformer brain!")