import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from attention import MultiHeadAttention
from positional_encoding import PositionalEncoding
from components import TransformerBlock, LayerNorm

# ============================================
# JARVIS TRANSFORMER - COMPLETE BRAIN
# Built entirely from scratch
# ============================================

class JarvisTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads,
                 num_layers, max_seq_len, dropout=0.1):
        super().__init__()

        self.d_model     = d_model
        self.vocab_size  = vocab_size
        self.num_layers  = num_layers

        # 1. Token Embedding: integer → vector
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 2. Positional Encoding: inject position info
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        # 3. Stack of Transformer Blocks - the core intelligence
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # 4. Final Layer Norm
        self.final_norm = LayerNorm(d_model)

        # 5. Output projection: vector → vocabulary probabilities
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share embedding and output weights
        # This is a key trick from the original Transformer paper
        self.output_proj.weight = self.embedding.weight

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def make_causal_mask(self, seq_len):
        # Causal mask: each token can only see previous tokens
        # Prevents JARVIS from "cheating" by looking at future words
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask

    def forward(self, tokens):
        batch_size, seq_len = tokens.shape

        # 1. Token embeddings + positional encoding
        x = self.embedding(tokens) * (self.d_model ** 0.5)
        x = self.pos_encoding(x)

        # 2. Causal mask - can't look at future tokens
        mask = self.make_causal_mask(seq_len).to(tokens.device)

        # 3. Pass through all Transformer blocks
        attention_maps = []
        for block in self.blocks:
            x, weights = block(x, mask)
            attention_maps.append(weights)

        # 4. Final normalization
        x = self.final_norm(x)

        # 5. Project to vocabulary size
        logits = self.output_proj(x)

        return logits, attention_maps

    def generate(self, tokens, max_new_tokens=100, temperature=0.8):
        """Generate new tokens autoregressively"""
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop context if too long
                context = tokens[:, -256:]

                # Get predictions
                logits, _ = self(context)

                # Focus on last token's prediction
                logits = logits[:, -1, :] / temperature

                # Convert to probabilities
                probs = torch.softmax(logits, dim=-1)

                # Sample next token
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to sequence
                tokens = torch.cat([tokens, next_token], dim=1)

        return tokens


# --- TEST & REPORT ---
if __name__ == "__main__":
    print("=" * 50)
    print("   JARVIS TRANSFORMER BRAIN")
    print("   Built from scratch - Phase 3")
    print("=" * 50 + "\n")

    # Small config - fits in RAM easily
    config = {
        'vocab_size':  3000,
        'd_model':     128,
        'num_heads':   4,
        'num_layers':  4,
        'max_seq_len': 256,
        'dropout':     0.1
    }

    print("Model Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()

    model = JarvisTransformer(**config)

    # Count parameters
    total_params    = sum(p.numel() for p in model.parameters())
    trainable       = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable:,}")
    print(f"Model Size (approx):  {total_params * 4 / 1024 / 1024:.2f} MB\n")

    # Test forward pass
    print("Testing forward pass...")
    batch_tokens = torch.randint(0, 3000, (2, 32))
    logits, attention_maps = model(batch_tokens)

    print(f"Input tokens shape:  {batch_tokens.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Attention maps:      {len(attention_maps)} layers")
    print(f"Each map shape:      {attention_maps[0].shape}\n")

    # Test generation
    print("Testing text generation...")
    seed = torch.randint(0, 3000, (1, 5))
    generated = model.generate(seed, max_new_tokens=10)
    print(f"Input length:  {seed.shape[1]} tokens")
    print(f"Output length: {generated.shape[1]} tokens")
    print(f"Generated {generated.shape[1] - seed.shape[1]} new tokens ✅\n")

    print("=" * 50)
    print("JARVIS TRANSFORMER COMPLETE!")
    print("This is the same architecture as GPT.")
    print("Built by you. From scratch.")
    print("=" * 50)