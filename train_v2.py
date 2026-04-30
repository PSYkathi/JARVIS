import torch
import torch.nn as nn
import json
import os
import math
import time

# ============================================
# JARVIS v0.2.0 — UPGRADED TRAINING
# 15.6M characters | 8,000 vocab | 21 books
# ============================================

# --- Load Tokenizer v2 ---
def load_tokenizer(path):
    with open(f'{path}/vocab.json', 'r') as f:
        vocab = json.load(f)
    with open(f'{path}/merges.json', 'r') as f:
        merges_raw = json.load(f)
    merges = {tuple(k.split(' ', 1)): v for k, v in merges_raw.items()}
    return vocab, merges

def encode(text, vocab, merges):
    tokens = []
    for word in text.strip().split():
        word_chars = list(word) + ['</w>']
        for pair, merged in merges.items():
            i = 0
            new_chars = []
            while i < len(word_chars):
                if (i < len(word_chars) - 1 and
                        (word_chars[i], word_chars[i+1]) == pair):
                    new_chars.append(merged)
                    i += 2
                else:
                    new_chars.append(word_chars[i])
                    i += 1
            word_chars = new_chars
        for ch in word_chars:
            tokens.append(vocab.get(ch, vocab.get('<unk>', 0)))
    return tokens

# --- Model Architecture (Upgraded) ---
class JARVISConfig:
    vocab_size    = 8000
    embed_dim     = 256    # upgraded from 128
    num_heads     = 8      # upgraded from 4
    num_layers    = 6      # upgraded from 4
    context_len   = 256    # upgraded from 128
    dropout       = 0.1
    ff_dim        = 1024   # upgraded from 512

class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim  = cfg.embed_dim // cfg.num_heads
        self.qkv       = nn.Linear(cfg.embed_dim, 3 * cfg.embed_dim)
        self.proj      = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.dropout   = nn.Dropout(cfg.dropout)
        self.register_buffer('mask',
            torch.tril(torch.ones(cfg.context_len, cfg.context_len))
            .unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.num_heads, self.head_dim)
                   .transpose(1, 2) for t in qkv]
        scale  = math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) / scale
        scores = scores.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        weights = self.dropout(torch.softmax(scores, dim=-1))
        out = (weights @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn  = MultiHeadAttention(cfg)
        self.ff    = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.ff_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ff_dim, cfg.embed_dim),
        )
        self.norm1 = nn.LayerNorm(cfg.embed_dim)
        self.norm2 = nn.LayerNorm(cfg.embed_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class JARVISModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed   = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_emb = nn.Embedding(cfg.context_len, cfg.embed_dim)
        self.blocks  = nn.Sequential(*[TransformerBlock(cfg)
                                       for _ in range(cfg.num_layers)])
        self.norm    = nn.LayerNorm(cfg.embed_dim)
        self.head    = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)
        self.drop    = nn.Dropout(cfg.dropout)
        self.context_len = cfg.context_len

    def forward(self, x, targets=None):
        B, T = x.shape
        pos  = torch.arange(T, device=x.device)
        out  = self.drop(self.embed(x) + self.pos_emb(pos))
        out  = self.blocks(out)
        out  = self.norm(out)
        logits = self.head(out)
        if targets is None:
            return logits, None
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def generate(self, idx, max_tokens=100, temperature=0.8):
        self.eval()
        with torch.no_grad():
            for _ in range(max_tokens):
                idx_cond = idx[:, -self.context_len:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature
                probs  = torch.softmax(logits, dim=-1)
                next_t = torch.multinomial(probs, 1)
                idx    = torch.cat([idx, next_t], dim=1)
        return idx

# --- Training Setup ---
def main():
    print("=" * 50)
    print("  JARVIS v0.2.0 — Training")
    print("=" * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load tokenizer
    print("\nLoading tokenizer v2...")
    vocab, merges = load_tokenizer('data/tokenizer_v2')
    print(f"Vocab size: {len(vocab)}")

    # Load all training data
    print("\nLoading training data...")
    all_text = ""
    for fname in sorted(os.listdir('data/clean')):
        if fname.endswith('.txt'):
            with open(f'data/clean/{fname}', 'r', encoding='utf-8') as f:
                all_text += f.read() + "\n"
    print(f"Total characters: {len(all_text):,}")

    # Encode (use first 3M chars for speed — upgrade later)
    print("\nLoading pre-encoded tokens...")
    data = torch.load('data/train_tokens.pt')
    print(f"Total tokens: {len(data):,}")

    # Train/val split
    split  = int(0.9 * len(data))
    train_data = data[:split]
    val_data   = data[split:]
    print(f"Train tokens: {len(train_data):,}")
    print(f"Val tokens:   {len(val_data):,}")

    # Model
    cfg   = JARVISConfig()
    model = JARVISModel(cfg).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {params:,}")

    checkpoint = torch.load('models/jarvis_v2_best.pt',map_location=device)
    model.load_state_dict(checkpoint['model'])
    print(f"Resumed from loss: {checkpoint['val_loss']:.4f}")

    # Training hyperparams
    BATCH_SIZE   = 64
    CONTEXT      = cfg.context_len
    LR           = 1e-4
    EPOCHS       = 10
    EVAL_EVERY   = 200
    SAVE_EVERY   = 500

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                   weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS * (len(train_data) // (BATCH_SIZE * CONTEXT)))

    def get_batch(split_data):
        ix = torch.randint(len(split_data) - CONTEXT - 1,
                           (BATCH_SIZE,))
        x  = torch.stack([split_data[i:i+CONTEXT] for i in ix])
        y  = torch.stack([split_data[i+1:i+CONTEXT+1] for i in ix])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def estimate_val_loss(steps=50):
        model.eval()
        losses = []
        for _ in range(steps):
            xb, yb = get_batch(val_data)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)

    # Training loop
    os.makedirs('models', exist_ok=True)
    print(f"\nStarting training: {EPOCHS} epochs\n")
    print(f"{'Step':>6} | {'Train Loss':>10} | {'Val Loss':>10} | {'Time':>8}")
    print("-" * 45)

    step       = 0
    best_loss  = float('inf')
    start_time = time.time()

    for epoch in range(EPOCHS):
        steps_per_epoch = len(train_data) // (BATCH_SIZE * CONTEXT)

        for i in range(steps_per_epoch):
            xb, yb = get_batch(train_data)
            _, loss = model(xb, yb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            step += 1

            if step % EVAL_EVERY == 0:
                val_loss  = estimate_val_loss()
                elapsed   = time.time() - start_time
                print(f"{step:>6} | {loss.item():>10.4f} | "
                      f"{val_loss:>10.4f} | {elapsed:>6.0f}s")

                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save({
                        'step':       step,
                        'model':      model.state_dict(),
                        'optimizer':  optimizer.state_dict(),
                        'val_loss':   val_loss,
                        'config':     vars(cfg)
                    }, 'models/jarvis_v2_best.pt')
                    print(f"         💾 Best model saved! (loss={val_loss:.4f})")

            if step % SAVE_EVERY == 0:
                torch.save({
                    'step':  step,
                    'model': model.state_dict(),
                    'config': vars(cfg)
                }, f'models/jarvis_v2_step{step}.pt')

        print(f"\n--- Epoch {epoch+1}/{EPOCHS} complete ---\n")

    print(f"\nTraining complete!")
    print(f"Best val loss: {best_loss:.4f}")
    print(f"Model saved:   models/jarvis_v2_best.pt")

if __name__ == "__main__":
    main()