import torch
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformer import JarvisTransformer
from data.tokenizer import BPETokenizer
from config import load_config

def load_trained_model():
    config    = load_config()
    model_cfg = config['model']

    # Load tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load('data/tokenizer')

    # Build model
    model = JarvisTransformer(**model_cfg)

    # Load trained weights
    checkpoint = torch.load(
        'checkpoints/jarvis_latest.pt',
        map_location='cpu'
    )
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    step = checkpoint['step']
    loss = checkpoint['loss']
    print(f"Model loaded! Trained for {step:,} steps")
    print(f"Last loss: {loss:.4f}\n")

    return model, tokenizer


def generate(model, tokenizer, prompt, max_tokens=200, temperature=0.8):
    # Encode prompt
    tokens = tokenizer.encode(prompt)
    x = torch.tensor([tokens], dtype=torch.long)

    # Generate
    output = model.generate(x, max_new_tokens=max_tokens,
                            temperature=temperature)

    # Decode
    return tokenizer.decode(output[0].tolist())


if __name__ == "__main__":
    print("=" * 55)
    print("   JARVIS — First Words After Training")
    print("=" * 55 + "\n")

    model, tokenizer = load_trained_model()

    # Test prompts
    prompts = [
        "It was a dark and stormy night",
        "The detective examined the",
        "She looked at him and said",
        "The ship sailed towards",
        "My name is"
    ]

    for prompt in prompts:
        print(f"Prompt:    '{prompt}'")
        print("-" * 40)
        result = generate(model, tokenizer, prompt,
                         max_tokens=100, temperature=0.8)
        print(result)
        print("\n")

    # Interactive mode
    print("=" * 55)
    print("INTERACTIVE MODE - Type your own prompts!")
    print("Type 'quit' to exit")
    print("=" * 55 + "\n")

    while True:
        prompt = input("Your prompt: ").strip()
        if prompt.lower() == 'quit':
            break
        if prompt:
            print("\nJARVIS:")
            print("-" * 40)
            result = generate(model, tokenizer, prompt,
                            max_tokens=150, temperature=0.8)
            print(result)
            print()