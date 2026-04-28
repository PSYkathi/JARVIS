import torch
import math
import matplotlib.pyplot as plt

# ============================================
# POSITIONAL ENCODING FROM SCRATCH
# Injects position information using
# sine and cosine waves at different frequencies
# ============================================

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_seq_len=512):
        super().__init__()

        # Create a matrix of shape (max_seq_len, d_model)
        pe = torch.zeros(max_seq_len, d_model)

        # Position indices: 0, 1, 2, ..., max_seq_len-1
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()

        # Frequency divisors
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            -(math.log(10000.0) / d_model)
        )

        # Even dimensions: sine wave
        pe[:, 0::2] = torch.sin(position * div_term)

        # Odd dimensions: cosine wave
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension
        pe = pe.unsqueeze(0)

        # Register as buffer (not a learned parameter)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to input embeddings
        seq_len = x.shape[1]
        return x + self.pe[:, :seq_len]


# --- TEST ---
if __name__ == "__main__":
    print("=== Positional Encoding from Scratch ===\n")

    d_model = 64
    max_len = 100

    pe = PositionalEncoding(d_model=d_model, max_seq_len=max_len)

    # Test with a sequence
    x = torch.zeros(1, 20, d_model)  # 20 tokens, all zeros
    output = pe(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\nFirst token position values (first 8 dims):")
    print(output[0, 0, :8].detach().numpy().round(4))
    print(f"\nSecond token position values (first 8 dims):")
    print(output[0, 1, :8].detach().numpy().round(4))
    print(f"\nTenth token position values (first 8 dims):")
    print(output[0, 9, :8].detach().numpy().round(4))
    print(f"\nEvery token has a UNIQUE position signature!")
    print(f"No two positions produce the same values.")

    # Verify positions are unique
    pos_vectors = output[0].detach()
    all_unique = True
    for i in range(pos_vectors.shape[0]):
        for j in range(i+1, pos_vectors.shape[0]):
            if torch.allclose(pos_vectors[i], pos_vectors[j], atol=1e-5):
                all_unique = False
    print(f"\nAll positions unique: {all_unique} ✅")

    # Save visualization
    pe_visual = PositionalEncoding(d_model=64, max_seq_len=50)
    pe_matrix = pe_visual.pe[0].detach().numpy()

    plt.figure(figsize=(12, 6))
    plt.imshow(pe_matrix, cmap='RdBu', aspect='auto')
    plt.colorbar()
    plt.title('Positional Encoding Matrix\nEach row = unique position signature')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Position in Sequence')
    plt.savefig('brain/positional_encoding.png')
    print(f"\nVisualization saved to brain/positional_encoding.png")
    print("Open it - you can see the sine/cosine wave patterns!")