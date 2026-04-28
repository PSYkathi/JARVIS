import json
import os

# ============================================
# JARVIS BRAIN CONFIGURATION
# Single source of truth for all model settings
# ============================================

JARVIS_CONFIG = {
    "model": {
        "vocab_size":   3000,
        "d_model":      128,
        "num_heads":    4,
        "num_layers":   4,
        "max_seq_len":  256,
        "dropout":      0.1
    },
    "training": {
        "batch_size":       32,
        "learning_rate":    3e-4,
        "max_epochs":       50,
        "eval_every":       500,
        "save_every":       1000,
        "warmup_steps":     100,
        "grad_clip":        1.0
    },
    "data": {
        "train_split":      0.9,
        "val_split":        0.1,
        "tokenizer_path":   "data/tokenizer",
        "clean_data_path":  "data/clean"
    },
    "paths": {
        "checkpoints":  "checkpoints",
        "brain":        "brain"
    },
    "version": "0.1.0",
    "name":    "JARVIS"
}

def save_config():
    os.makedirs('brain', exist_ok=True)
    with open('brain/config.json', 'w') as f:
        json.dump(JARVIS_CONFIG, f, indent=4)
    print("Config saved to brain/config.json ✅")

def load_config():
    with open('brain/config.json', 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    print("=== JARVIS Brain Configuration ===\n")

    save_config()

    # Verify it loads back correctly
    config = load_config()

    print("\nModel Architecture:")
    for k, v in config['model'].items():
        print(f"  {k}: {v}")

    print("\nTraining Settings:")
    for k, v in config['training'].items():
        print(f"  {k}: {v}")

    print(f"\nJARVIS Version: {config['version']}")
    print(f"Name: {config['name']}")

    # Calculate model complexity
    d     = config['model']['d_model']
    h     = config['model']['num_heads']
    L     = config['model']['num_layers']
    vocab = config['model']['vocab_size']

    approx_params = (
        vocab * d +          # embedding
        L * (4 * d * d) +    # attention
        L * (8 * d * d) +    # feedforward
        vocab * d            # output projection
    )

    print(f"\nEstimated parameters: ~{approx_params:,}")
    print(f"Estimated model size: ~{approx_params * 4 / 1024 / 1024:.2f} MB")
    print(f"\nThis config is JARVIS's DNA.")
    print(f"Every training run reads from here.")