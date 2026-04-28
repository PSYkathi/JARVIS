import torch
import math

# ============================================
# SELF-ATTENTION FROM SCRATCH
# The heart of the Transformer
# ============================================

def self_attention(Q, K, V, mask=None):
    """
    Q = Query  (what am I looking for?)
    K = Key    (what do I contain?)
    V = Value  (what do I actually give?)

    For every token:
    1. Compare its Query against all Keys -> attention scores
    2. Normalize scores with softmax -> attention weights
    3. Weighted sum of Values -> output
    """
    d_k = Q.shape[-1]  # dimension of keys

    # Step 1: Compute attention scores
    # Q @ K^T = how much each query matches each key
    scores = torch.matmul(Q, K.transpose(-2, -1))

    # Step 2: Scale - prevents scores from getting too large
    scores = scores / math.sqrt(d_k)

    # Step 3: Apply mask (prevent looking at future tokens)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Step 4: Softmax - convert scores to probabilities
    weights = torch.softmax(scores, dim=-1)

    # Step 5: Weighted sum of values
    output = torch.matmul(weights, V)

    return output, weights


class MultiHeadAttention(torch.nn.Module):
    """
    Run self-attention H times in parallel
    Each head learns different relationships
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model    = d_model
        self.num_heads  = num_heads
        self.d_k        = d_model // num_heads

        # Linear projections for Q, K, V and output
        self.W_q = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_k = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_v = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_o = torch.nn.Linear(d_model, d_model, bias=False)

    def split_heads(self, x, batch_size):
        # Split d_model into num_heads x d_k
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # (batch, heads, seq, d_k)

    def forward(self, x, mask=None):
        batch_size = x.shape[0]

        # Project input to Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Split into multiple heads
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # Apply attention on each head
        attended, weights = self_attention(Q, K, V, mask)

        # Merge heads back together
        attended = attended.transpose(1, 2)
        attended = attended.contiguous().view(batch_size, -1, self.d_model)

        # Final linear projection
        output = self.W_o(attended)

        return output, weights


# --- TEST ---
if __name__ == "__main__":
    print("=== Self-Attention from Scratch ===\n")

    # Simple test: 4 words, each represented by 8 numbers
    batch_size = 1
    seq_len    = 4
    d_model    = 8

    # Simulate word embeddings
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"Input shape: {x.shape}")
    print(f"(batch={batch_size}, sequence_length={seq_len}, d_model={d_model})\n")

    # Test single-head attention
    Q = K = V = x
    output, weights = self_attention(Q, K, V)
    print(f"Single-head attention output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"\nAttention weights (how much each word attends to others):")
    print(weights[0].detach().numpy().round(3))

    # Test multi-head attention
    print(f"\n--- Multi-Head Attention ---")
    mha = MultiHeadAttention(d_model=64, num_heads=4)
    x2 = torch.randn(2, 10, 64)
    output2, weights2 = mha(x2)
    print(f"Input shape:  {x2.shape}")
    print(f"Output shape: {output2.shape}")
    print(f"Weights shape: {weights2.shape}")
    print(f"\nMulti-Head Attention working!")
    print(f"4 heads each looking at relationships differently")