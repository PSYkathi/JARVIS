import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import os, glob, math, time


# ── Config ───────────────────────────────────────────────────
TOKENIZED_DIR  = "data/tokenized_v3"
TOKENIZER_DIR  = "data/tokenizer_v3"
CHECKPOINT_DIR = "checkpoints_v3"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

VOCAB_SIZE   = 8000
CONTEXT_LEN  = 64
BATCH_SIZE   = 32
EMBED_DIM    = 256
N_HEADS      = 8
N_LAYERS     = 6
DROPOUT      = 0.1
LR_MAX       = 2e-4
LR_MIN       = 1e-5
WARMUP_STEPS = 200
MAX_EPOCHS   = 1
SAVE_EVERY   = 1000
EVAL_EVERY   = 200
LOG_EVERY    = 100
VAL_BATCHES  = 50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}\n")


# ── Dataset ──────────────────────────────────────────────────
class TokenDataset(Dataset):
    def __init__(self, folder, context_len):
        self.context_len = context_len
        all_ids = []
        files = sorted(glob.glob(f"{folder}/*.ids"))
        print(f"Loading {len(files)} token files...")
        for fpath in files:
            with open(fpath) as f:
                ids = list(map(int, f.read().split()))
            all_ids.extend(ids)
            print(f"  ✅ {os.path.basename(fpath)}: {len(ids):,} tokens")
        self.data = torch.tensor(all_ids, dtype=torch.long)
        print(f"\nTotal tokens: {len(self.data):,}")
        self.data = self.data[:2_000_000]
        print(f"Using first 2M tokens for speed\n")

    def __len__(self):
        return len(self.data) - self.context_len

    def __getitem__(self, i):
        chunk = self.data[i : i + self.context_len + 1]
        return chunk[:-1], chunk[1:]


# ── Model ────────────────────────────────────────────────────
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1  = nn.LayerNorm(EMBED_DIM)
        self.attn = nn.MultiheadAttention(EMBED_DIM, N_HEADS, dropout=DROPOUT, batch_first=True)
        self.ln2  = nn.LayerNorm(EMBED_DIM)
        self.ff   = nn.Sequential(
            nn.Linear(EMBED_DIM, 4 * EMBED_DIM),
            nn.GELU(),
            nn.Linear(4 * EMBED_DIM, EMBED_DIM),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        T    = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask)
        x    = x + a
        x    = x + self.ff(self.ln2(x))
        return x


class JARVIS(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.pos_emb = nn.Embedding(CONTEXT_LEN, EMBED_DIM)
        self.drop    = nn.Dropout(DROPOUT)
        self.blocks  = nn.Sequential(*[Block() for _ in range(N_LAYERS)])
        self.ln_f    = nn.LayerNorm(EMBED_DIM)
        self.head    = nn.Linear(EMBED_DIM, VOCAB_SIZE, bias=False)
        self.head.weight = self.tok_emb.weight
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T   = idx.shape
        pos    = torch.arange(T, device=idx.device)
        x      = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        x      = self.blocks(x)
        x      = self.ln_f(x)
        logits = self.head(x)
        loss   = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, VOCAB_SIZE), targets.view(-1)
            )
        return logits, loss

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Safe Save ────────────────────────────────────────────────
def safe_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ── Find Latest Checkpoint ───────────────────────────────────
def find_latest_checkpoint():
    # First priority: latest.pt (saved every step on Ctrl+C)
    latest = f"{CHECKPOINT_DIR}/jarvis_v3_latest.pt"
    if os.path.exists(latest):
        return latest

    # Second priority: highest step periodic checkpoint
    ckpts = glob.glob(f"{CHECKPOINT_DIR}/jarvis_v3_step*.pt")
    if ckpts:
        # extract step numbers and pick highest
        def get_step(p):
            try: return int(os.path.basename(p).replace("jarvis_v3_step","").replace(".pt",""))
            except: return -1
        return max(ckpts, key=get_step)

    # No checkpoint found
    return None


# ── LR Schedule ──────────────────────────────────────────────
def get_lr(step, total_steps):
    if step < WARMUP_STEPS:
        return LR_MAX * step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / (total_steps - WARMUP_STEPS)
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * progress))


# ── Validation ───────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, val_loader, max_batches=VAL_BATCHES):
    model.eval()
    total_loss = 0
    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        x, y = x.to(DEVICE), y.to(DEVICE)
        _, loss = model(x, y)
        total_loss += loss.item()
    model.train()
    return total_loss / min(max_batches, len(val_loader))


# ── Train ────────────────────────────────────────────────────
def train():
    full_dataset = TokenDataset(TOKENIZED_DIR, CONTEXT_LEN)
    val_size     = int(0.1 * len(full_dataset))
    train_size   = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    print(f"Train samples: {train_size:,} | Val samples: {val_size:,}\n")

    model     = JARVIS().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=LR_MAX, weight_decay=0.1,
                                  betas=(0.9, 0.95))

    total_steps   = MAX_EPOCHS * len(train_loader)
    step          = 0
    best_val_loss = float('inf')

    # ── Auto Resume ──────────────────────────────────────────
    resume_path = find_latest_checkpoint()
    if resume_path:
        print(f"📂 Resuming from: {resume_path}")
        ckpt = torch.load(resume_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt['model'])
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        step          = ckpt.get('step', 0)
        best_val_loss = ckpt.get('val_loss', float('inf'))
        print(f"✅ Resumed! step={step:,} | best_val={best_val_loss:.4f}\n")
    else:
        print("🆕 No checkpoint found, starting fresh.\n")

    print(f"Model parameters: {model.count_params():,}")
    print(f"Total steps:  {total_steps:,}")
    print(f"Log every:    {LOG_EVERY} steps")
    print(f"Eval every:   {EVAL_EVERY} steps  (max {VAL_BATCHES} batches)")
    print(f"Save every:   {SAVE_EVERY} steps\n")
    print(f"{'Step':>8} | {'Train Loss':>10} | {'Val Loss':>10} | {'LR':>10} | {'Time':>8}")
    print("-" * 60)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        epoch_loss     = 0
        t0             = time.time()
        steps_in_epoch = step % len(train_loader)

        try:
            for i, (x, y) in enumerate(train_loader):

                # ── Skip already-completed steps ──────────────
                if i < steps_in_epoch:
                    continue

                x, y = x.to(DEVICE), y.to(DEVICE)

                lr = get_lr(step, total_steps)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr

                _, loss = model(x, y)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                step += 1

                # ── Log train loss ─────────────────────────────
                if step % LOG_EVERY == 0:
                    elapsed = time.time() - t0
                    print(f"{step:>8,} | {loss.item():>10.4f} | {'─':>10} | {lr:>10.6f} | {elapsed:>6.1f}s")

                # ── Eval + save best ───────────────────────────
                if step % EVAL_EVERY == 0:
                    val_loss = evaluate(model, val_loader)
                    elapsed  = time.time() - t0
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        safe_save({
                            "step":       step,
                            "model":      model.state_dict(),
                            "optimizer":  optimizer.state_dict(),
                            "train_loss": loss.item(),
                            "val_loss":   val_loss,
                        }, f"{CHECKPOINT_DIR}/jarvis_v3_best.pt")
                        print(f"{step:>8,} | {loss.item():>10.4f} | {val_loss:>10.4f} | {lr:>10.6f} | {elapsed:>6.1f}s  ✅ New best!")

                # ── Periodic checkpoint ────────────────────────
                if step % SAVE_EVERY == 0:
                    safe_save({
                        "step":       step,
                        "model":      model.state_dict(),
                        "optimizer":  optimizer.state_dict(),
                        "train_loss": loss.item(),
                        "val_loss":   best_val_loss,
                    }, f"{CHECKPOINT_DIR}/jarvis_v3_step{step}.pt")
                    print(f"\n💾 Checkpoint saved: jarvis_v3_step{step}.pt | best_val={best_val_loss:.4f}\n")

        except KeyboardInterrupt:
            # ── Save exactly where we stopped ─────────────────
            print(f"\n⚠️  Interrupted at step {step:,} — saving latest checkpoint...")
            safe_save({
                "step":       step,
                "model":      model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "train_loss": loss.item(),
                "val_loss":   best_val_loss,
            }, f"{CHECKPOINT_DIR}/jarvis_v3_latest.pt")
            print(f"💾 Saved: jarvis_v3_latest.pt | step={step:,} | best_val={best_val_loss:.4f}")
            print("▶️  Run python train_v3.py again to resume from here.\n")
            return

        avg = epoch_loss / max(1, len(train_loader) - steps_in_epoch)
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{MAX_EPOCHS} | Avg Train Loss: {avg:.4f} | Time: {time.time()-t0:.1f}s")
        print(f"Best Val Loss: {best_val_loss:.4f}")
        print(f"{'='*60}\n")

    # ── Final save ────────────────────────────────────────────
    safe_save({
        "step":  step,
        "model": model.state_dict(),
    }, f"{CHECKPOINT_DIR}/jarvis_v3_final.pt")

    # Clean up latest.pt since training is complete
    latest = f"{CHECKPOINT_DIR}/jarvis_v3_latest.pt"
    if os.path.exists(latest):
        os.remove(latest)

    print("🎉 Training complete!")
    print(f"   Final  → jarvis_v3_final.pt")
    print(f"   Best   → jarvis_v3_best.pt  (val_loss={best_val_loss:.4f})")


if __name__ == "__main__":
    train()