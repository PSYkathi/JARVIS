import torch
import torch.nn as nn
import json
import os
import random

# ================================================
# JARVIS MINI LANGUAGE MODEL
# Predicts the next character given previous ones
# Uses everything we built in Phase 1 & 2
# ================================================

class CharLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        # Embedding: converts character index to vector
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Two hidden layers
        self.layer1 = nn.Linear(embed_size * 8, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        # Output: predict next character
        self.output = nn.Linear(hidden_size, vocab_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x shape: (batch, context_length)
        embedded = self.embedding(x)
        # Flatten context window
        embedded = embedded.view(embedded.shape[0], -1)
        h1 = self.relu(self.layer1(embedded))
        h1 = self.dropout(h1)
        h2 = self.relu(self.layer2(h1))
        logits = self.output(h2)
        return logits


def build_char_vocab(text):
    chars = sorted(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}
    return char_to_idx, idx_to_char


def get_batch(data, context_len, batch_size):
    indices = [random.randint(0, len(data) - context_len - 1)
               for _ in range(batch_size)]
    X = torch.stack([data[i:i+context_len] for i in indices])
    y = torch.stack([data[i+context_len] for i in indices])
    return X, y


def generate_text(model, char_to_idx, idx_to_char, start_text, length=200):
    model.eval()
    context_len = 8
    result = start_text

    # Pad or trim start
    context = start_text[-context_len:].ljust(context_len)[-context_len:]
    indices = [char_to_idx.get(c, 0) for c in context]

    with torch.no_grad():
        for _ in range(length):
            x = torch.tensor([indices[-context_len:]])
            logits = model(x)
            # Sample from probabilities
            probs = torch.softmax(logits / 0.8, dim=-1)
            next_idx = torch.multinomial(probs, 1).item()
            result += idx_to_char[next_idx]
            indices.append(next_idx)

    return result


if __name__ == "__main__":
    print("=== JARVIS Mini Language Model ===\n")

    # Load clean text
    text = ""
    for fname in os.listdir('data/clean'):
        if fname.endswith('.txt'):
            with open(f'data/clean/{fname}', 'r', encoding='utf-8') as f:
                text += f.read()[:50000]  # 50k chars per book

    print(f"Loaded {len(text):,} characters of training text")

    # Build character vocabulary
    char_to_idx, idx_to_char = build_char_vocab(text)
    vocab_size = len(char_to_idx)
    print(f"Character vocabulary size: {vocab_size}")

    # Encode full text
    data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)

    # Model settings
    context_len = 8
    embed_size  = 32
    hidden_size = 256
    batch_size  = 128
    epochs      = 500

    model = CharLanguageModel(vocab_size, embed_size, hidden_size)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}\n")

    # Loss and optimizer - from scratch understanding
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    loss_fn   = nn.CrossEntropyLoss()

    # Training loop
    print("Training...\n")
    losses = []
    for epoch in range(epochs):
        model.train()
        X, y = get_batch(data, context_len, batch_size)
        logits = model(X)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 100 == 0:
            avg_loss = sum(losses[-100:]) / 100
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

    # Generate text
    print("\n--- JARVIS Generates Text ---")
    print("Seed: 'It was a '")
    print("-" * 40)
    generated = generate_text(model, char_to_idx, idx_to_char,
                              start_text="It was a ", length=300)
    print(generated)
    print("-" * 40)
    print("\nPhase 2 Complete!")
    print("JARVIS just generated its first text from scratch.")