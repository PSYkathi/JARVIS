import torch
import json
import os
import time

def load_tokenizer(path):
    with open(f'{path}/vocab.json', 'r') as f:
        vocab = json.load(f)
    with open(f'{path}/merges.json', 'r') as f:
        merges_raw = json.load(f)
    merges = {tuple(k.split(' ', 1)): v for k, v in merges_raw.items()}
    return vocab, merges

def encode_fast(text, vocab, merges):
    """Faster encode: process word by word with early exit"""
    unk = vocab.get('<unk>', 0)
    tokens = []
    words = text.split()
    total = len(words)

    for i, word in enumerate(words):
        if i % 50000 == 0:
            print(f"  Encoding: {i:,}/{total:,} words ({100*i//total}%)")

        word_chars = list(word) + ['</w>']

        # Apply merges
        changed = True
        while changed:
            changed = False
            new_chars = []
            j = 0
            while j < len(word_chars):
                if (j < len(word_chars) - 1 and
                    (word_chars[j], word_chars[j+1]) in merges):
                    new_chars.append(merges[(word_chars[j], word_chars[j+1])])
                    j += 2
                    changed = True
                else:
                    new_chars.append(word_chars[j])
                    j += 1
            word_chars = new_chars

        for ch in word_chars:
            tokens.append(vocab.get(ch, unk))

    return tokens

print("=== Pre-encoding Training Data ===\n")

vocab, merges = load_tokenizer('data/tokenizer_v2')
print(f"Tokenizer loaded. Merges: {len(merges):,}\n")

# Load text
all_text = ""
for fname in sorted(os.listdir('data/clean')):
    if fname.endswith('.txt'):
        with open(f'data/clean/{fname}', 'r', encoding='utf-8') as f:
            all_text += f.read() + "\n"

# Use 1M chars first (fast, enough to train well)
sample = all_text
print(f"Encoding {len(sample):,} characters...\n")

start = time.time()
tokens = encode_fast(sample, vocab, merges)
elapsed = time.time() - start

print(f"\nEncoding done in {elapsed:.1f}s")
print(f"Total tokens: {len(tokens):,}")

# Save to disk
data = torch.tensor(tokens, dtype=torch.long)
torch.save(data, 'data/train_tokens.pt')
print(f"Saved to data/train_tokens.pt")
print("Now run: python train_v2.py")