import torch
from torch.utils.data import Dataset, DataLoader
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.tokenizer import BPETokenizer

# ============================================
# JARVIS TRAINING DATASET
# Converts raw text into training batches
# ============================================

class JarvisDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens  = tokens
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):
        # Input:  tokens[idx : idx+seq_len]
        # Target: tokens[idx+1 : idx+seq_len+1]
        # Model learns: given these tokens, predict next one
        x = self.tokens[idx     : idx + self.seq_len]
        y = self.tokens[idx + 1 : idx + self.seq_len + 1]
        return x, y


def build_dataset(seq_len=128):
    print("=== Building Training Dataset ===\n")

    # Load tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load('data/tokenizer')

    # Load and encode all clean text
    all_tokens = []
    clean_path = 'data/clean'

    for fname in sorted(os.listdir(clean_path)):
        if fname.endswith('.txt'):
            with open(f'{clean_path}/{fname}', 'r', encoding='utf-8') as f:
                text = f.read()
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)
            print(f"  {fname}: {len(tokens):,} tokens")

    print(f"\nTotal tokens: {len(all_tokens):,}")

    # Convert to tensor
    data = torch.tensor(all_tokens, dtype=torch.long)

    # Split into train and validation
    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data   = data[split:]

    print(f"Train tokens: {len(train_data):,}")
    print(f"Val tokens:   {len(val_data):,}")

    # Create datasets
    train_dataset = JarvisDataset(train_data, seq_len)
    val_dataset   = JarvisDataset(val_data,   seq_len)

    print(f"\nTraining samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    print(f"\nTrain batches: {len(train_loader):,}")
    print(f"Val batches:   {len(val_loader):,}")

    # Show one sample
    print(f"\nSample batch:")
    x, y = next(iter(train_loader))
    print(f"  Input shape:  {x.shape}")
    print(f"  Target shape: {y.shape}")
    print(f"  Input tokens (first 10):  {x[0][:10].tolist()}")
    print(f"  Target tokens (first 10): {y[0][:10].tolist()}")
    print(f"  (Target is input shifted by 1 - correct!) ✅")

    return train_loader, val_loader, tokenizer


if __name__ == "__main__":
    train_loader, val_loader, tokenizer = build_dataset()
    print("\nDataset pipeline ready for training!")