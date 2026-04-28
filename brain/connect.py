import torch
import json
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformer import JarvisTransformer
from data.tokenizer import BPETokenizer

# ============================================
# CONNECT TOKENIZER → TRANSFORMER
# Real words in → Real words out
# ============================================

def load_tokenizer():
    tokenizer = BPETokenizer()
    tokenizer.load('data/tokenizer')
    return tokenizer

def text_to_tokens(text, tokenizer):
    encoded = tokenizer.encode(text)
    return torch.tensor([encoded], dtype=torch.long)

def tokens_to_text(tokens, tokenizer):
    token_list = tokens[0].tolist()
    return tokenizer.decode(token_list)

if __name__ == "__main__":
    print("=== Connecting Tokenizer to Transformer ===\n")

    # Load tokenizer
    tokenizer = load_tokenizer()
    vocab_size = len(tokenizer.vocab)
    print(f"Tokenizer vocabulary size: {vocab_size}")

    # Build transformer with correct vocab size
    model = JarvisTransformer(
        vocab_size  = vocab_size,
        d_model     = 128,
        num_heads   = 4,
        num_layers  = 4,
        max_seq_len = 256,
        dropout     = 0.1
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}\n")

    # Test: encode real text
    test_sentences = [
        "It was a dark and stormy night",
        "The detective looked at the clues",
        "Hello my name is"
    ]

    print("Encoding real sentences:\n")
    for sentence in test_sentences:
        tokens = text_to_tokens(sentence, tokenizer)
        print(f"Input:   '{sentence}'")
        print(f"Tokens:  {tokens[0].tolist()}")
        print(f"Length:  {tokens.shape[1]} tokens")

        # Forward pass through transformer
        logits, _ = model(tokens)
        print(f"Logits:  shape {logits.shape}")

        # What token does JARVIS predict next? (untrained = random)
        next_token_logits = logits[0, -1, :]
        predicted_idx = next_token_logits.argmax().item()
        reverse_vocab = {i: ch for ch, i in tokenizer.vocab.items()}
        predicted_token = reverse_vocab.get(predicted_idx, '<unk>')
        print(f"Predicted next token (untrained): '{predicted_token}'")
        print()

    # Full generation test
    print("-" * 40)
    print("Generation test (untrained - output is gibberish):")
    print("This will improve dramatically in Phase 4 when we train!\n")

    seed_text = "The ship sailed"
    seed_tokens = text_to_tokens(seed_text, tokenizer)
    generated_tokens = model.generate(seed_tokens, max_new_tokens=20)
    generated_text = tokens_to_text(generated_tokens, tokenizer)

    print(f"Seed:      '{seed_text}'")
    print(f"Generated: '{generated_text}'")
    print()
    print("Pipeline connected!")
    print("Tokenizer → Transformer → Generation all working ✅")
    print()
    print("Next: TRAIN this model so the output stops being gibberish!")