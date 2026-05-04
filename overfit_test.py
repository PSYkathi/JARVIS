import torch
import torch.nn as nn

# ── Tiny config ──────────────────────────
VOCAB_SIZE   = 32000
EMBED_DIM    = 128
N_HEADS      = 4
N_LAYERS     = 3
CONTEXT_LEN  = 64
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# ── Tiny model ───────────────────────────
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1  = nn.LayerNorm(EMBED_DIM)
        self.attn = nn.MultiheadAttention(EMBED_DIM, N_HEADS, batch_first=True)
        self.ln2  = nn.LayerNorm(EMBED_DIM)
        self.ff   = nn.Sequential(
            nn.Linear(EMBED_DIM, 4 * EMBED_DIM),
            nn.GELU(),
            nn.Linear(4 * EMBED_DIM, EMBED_DIM),
        )

    def forward(self, x):
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        att, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask)
        x = x + att
        x = x + self.ff(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.pos_emb = nn.Embedding(CONTEXT_LEN, EMBED_DIM)
        self.blocks  = nn.Sequential(*[Block() for _ in range(N_LAYERS)])
        self.ln_f    = nn.LayerNorm(EMBED_DIM)
        self.head    = nn.Linear(EMBED_DIM, VOCAB_SIZE, bias=False)
        self.head.weight = self.tok_emb.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.ln_f(self.blocks(x))
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
        return logits, loss

# ── Load ONE real batch ───────────────────
import glob
all_ids = []
for f in sorted(glob.glob("data/tokenized/*.ids"))[:5]:
    with open(f) as fp:
        all_ids.extend(list(map(int, fp.read().split())))

data  = torch.tensor(all_ids[:CONTEXT_LEN + 1], dtype=torch.long)
x     = data[:-1].unsqueeze(0).to(DEVICE)
y     = data[1:].unsqueeze(0).to(DEVICE)

# ── Overfit loop ─────────────────────────
model     = TinyGPT().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
print("Overfitting one batch...\n")

for step in range(1, 201):
    _, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 20 == 0:
        print(f"Step {step:3d} | Loss: {loss.item():.4f}")