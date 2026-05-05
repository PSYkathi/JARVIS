import torch
import torch.nn as nn
from tokenizers import ByteLevelBPETokenizer
import math

# ── Config (must match train_v3.py) ─────────────────────────
VOCAB_SIZE   = 8000
CONTEXT_LEN  = 64
EMBED_DIM    = 256
N_HEADS      = 8
N_LAYERS     = 6
DROPOUT      = 0.0   # off during inference
CHECKPOINT = "checkpoints_v3/jarvis_v3_step18000.pt"
TOKENIZER    = "data/tokenizer_v3"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# ── Model (copy from train_v3.py) ────────────────────────────
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

    def forward(self, idx):
        B, T   = idx.shape
        pos    = torch.arange(T, device=idx.device)
        x      = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        x      = self.blocks(x)
        x      = self.ln_f(x)
        return self.head(x)

# ── Generate ─────────────────────────────────────────────────
@torch.no_grad()
def generate(model, tok, prompt, max_new=200, temperature=0.8, top_k=40):
    model.eval()
    ids = tok.encode(prompt).ids
    ids = torch.tensor([ids], dtype=torch.long, device=DEVICE)

    for _ in range(max_new):
        # crop to context length
        ids_crop = ids[:, -CONTEXT_LEN:]
        logits   = model(ids_crop)
        logits   = logits[:, -1, :] / temperature

        # top-k sampling
        if top_k > 0:
            values, _ = torch.topk(logits, top_k)
            logits[logits < values[:, -1:]] = float('-inf')

        probs  = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids    = torch.cat([ids, next_id], dim=1)

    output_ids = ids[0].tolist()
    return tok.decode(output_ids)

# ── Main ─────────────────────────────────────────────────────
def main():
    print(f"Loading tokenizer...")
    tok = ByteLevelBPETokenizer(
        f'{TOKENIZER}/vocab.json',
        f'{TOKENIZER}/merges.txt'
    )

    print(f"Loading checkpoint: {CHECKPOINT}")
    model = JARVIS().to(DEVICE)
    ckpt  = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(ckpt['model'])
    print(f"✅ Loaded! (step {ckpt['step']})\n")

    print("Type a prompt and press Enter. Ctrl+C to quit.\n")
    while True:
        try:
            prompt = input("You: ").strip()
            if not prompt:
                continue
            output = generate(model, tok, prompt)
            print(f"\nJARVIS: {output}\n")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    main()