import torch
import torch.nn as nn
import json
import math

# --- Load tokenizer ---
def load_tokenizer(path):
    with open(f'{path}/vocab.json', 'r') as f:
        vocab = json.load(f)
    with open(f'{path}/merges.json', 'r') as f:
        merges_raw = json.load(f)
    merges = {tuple(k.split(' ', 1)): v for k, v in merges_raw.items()}
    return vocab, merges

def encode(text, vocab, merges):
    unk = vocab.get('<unk>', 0)
    tokens = []
    for word in text.strip().split():
        word_chars = list(word) + ['</w>']
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

def decode(tokens, vocab):
    reverse_vocab = {i: ch for ch, i in vocab.items()}
    result = ""
    for t in tokens:
        piece = reverse_vocab.get(t, '')
        if piece in ('<unk>', '<pad>', '<bos>', '<eos>'):
            result += ' '          # skip special tokens
        elif piece.endswith('</w>'):
            result += piece[:-4] + ' '   # strip </w>, add space = word boundary
        else:
            result += piece              # sub-word piece, no space = glue together
    return result.strip()

# --- Model (same as train_v2.py) ---
class JARVISConfig:
    vocab_size  = 8000
    embed_dim   = 256
    num_heads   = 8
    num_layers  = 6
    context_len = 256
    dropout     = 0.1
    ff_dim      = 1024

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

    def generate(self, idx, max_tokens=100, temperature=0.8, top_k=40):
        self.eval()
        with torch.no_grad():
            for _ in range(max_tokens):
                idx_cond = idx[:, -self.context_len:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature
                # Top-k sampling
                values, _ = torch.topk(logits, top_k)
                logits[logits < values[:, [-1]]] = float('-inf')
                probs  = torch.softmax(logits, dim=-1)
                next_t = torch.multinomial(probs, 1)
                idx    = torch.cat([idx, next_t], dim=1)
        return idx

# --- Main ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load model
cfg   = JARVISConfig()
model = JARVISModel(cfg).to(device)
checkpoint = torch.load('models/jarvis_v2_best.pt',
                         map_location=device)
model.load_state_dict(checkpoint['model'])
model.eval()
print(f"Model loaded! (val loss: {checkpoint['val_loss']:.4f})\n")

# Load tokenizer
vocab, merges = load_tokenizer('data/tokenizer_v2')

# Test prompts
prompts = [
    "The ship sailed",
    "It was a dark",
    "The detective looked",
    "She opened the door",
    "The captain said",
]

print("=" * 55)
print("  JARVIS v0.2.0 — Text Generation Test")
print("=" * 55)

for prompt in prompts:
    tokens = encode(prompt, vocab, merges)
    idx    = torch.tensor([tokens], dtype=torch.long).to(device)
    out    = model.generate(idx, max_tokens=40, temperature=0.8, top_k=40)
    result = decode(out[0].tolist(), vocab)
    print(f"\nPrompt:  '{prompt}'")
    print(f"JARVIS:   {result}")
    print("-" * 55)