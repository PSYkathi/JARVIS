import torch
import torch.nn as nn
import os
import sys
import json
import time
import math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformer import JarvisTransformer
from dataset import build_dataset
from config import load_config

# ============================================
# JARVIS TRAINING ENGINE
# Full training loop with:
# - Warmup learning rate schedule
# - Gradient clipping
# - Checkpointing (save & resume)
# - Evaluation
# - Progress logging
# ============================================

def get_lr(step, d_model, warmup_steps):
    """
    Warmup then decay learning rate.
    From the original Transformer paper.
    """
    if step == 0:
        step = 1
    return (d_model ** -0.5) * min(
        step ** -0.5,
        step * warmup_steps ** -1.5
    )

def evaluate(model, val_loader, loss_fn, device, max_batches=50):
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = loss_fn(
                logits.view(-1, logits.shape[-1]),
                y.view(-1)
            )
            total_loss += loss.item()
            count += 1
    model.train()
    return total_loss / count if count > 0 else 0

def save_checkpoint(model, optimizer, step, epoch, loss, config):
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint = {
        'step':           step,
        'epoch':          epoch,
        'model_state':    model.state_dict(),
        'optimizer_state':optimizer.state_dict(),
        'loss':           loss,
        'config':         config
    }
    path = f'checkpoints/jarvis_step_{step}.pt'
    torch.save(checkpoint, path)
    # Also save as latest
    torch.save(checkpoint, 'checkpoints/jarvis_latest.pt')
    print(f"  💾 Checkpoint saved: {path}")

def load_checkpoint(model, optimizer):
    path = 'checkpoints/jarvis_latest.pt'
    if os.path.exists(path):
        print(f"Found checkpoint! Loading from {path}")
        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        return checkpoint['step'], checkpoint['epoch'], checkpoint['loss']
    return 0, 0, None


def train():
    print("=" * 55)
    print("   JARVIS BRAIN TRAINING")
    print("=" * 55 + "\n")

    # Load config
    config      = load_config()
    model_cfg   = config['model']
    train_cfg   = config['training']

    device = 'cpu'
    print(f"Device: {device}")
    print(f"Model: {model_cfg['num_layers']} layers, "
          f"{model_cfg['d_model']} dims, "
          f"{model_cfg['num_heads']} heads\n")

    # Build dataset
    train_loader, val_loader, tokenizer = build_dataset(
        seq_len=model_cfg['max_seq_len']
    )

    # Build model
    model = JarvisTransformer(**model_cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    # Loss function
    loss_fn = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0,  # LR controlled by scheduler
        betas=(0.9, 0.98),
        eps=1e-9
    )

    # Load checkpoint if exists
    start_step, start_epoch, last_loss = load_checkpoint(model, optimizer)
    if start_step > 0:
        print(f"Resuming from step {start_step}, epoch {start_epoch}")

    # Training log
    log = {
        'steps': [], 'train_loss': [],
        'val_loss': [], 'time': []
    }

    print("\n" + "=" * 55)
    print("Starting training...")
    print("Press Ctrl+C at any time - progress is saved!")
    print("=" * 55 + "\n")

    step        = start_step
    start_time  = time.time()
    model.train()

    try:
        for epoch in range(start_epoch, train_cfg['max_epochs']):
            epoch_loss  = 0
            batch_count = 0

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)

                # Update learning rate
                lr = get_lr(
                    step + 1,
                    model_cfg['d_model'],
                    train_cfg['warmup_steps']
                ) * train_cfg['learning_rate'] * 1000

                for g in optimizer.param_groups:
                    g['lr'] = lr

                # Forward pass
                logits, _ = model(x)

                # Calculate loss
                loss = loss_fn(
                    logits.view(-1, logits.shape[-1]),
                    y.view(-1)
                )

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    train_cfg['grad_clip']
                )

                # Update weights
                optimizer.step()

                step        += 1
                epoch_loss  += loss.item()
                batch_count += 1

                # Print progress every 100 steps
                if step % 100 == 0:
                    elapsed = time.time() - start_time
                    avg_loss = epoch_loss / batch_count
                    print(f"Epoch {epoch+1} | Step {step:,} | "
                          f"Loss: {avg_loss:.4f} | "
                          f"LR: {lr:.6f} | "
                          f"Time: {elapsed:.0f}s")

                # Evaluate every 500 steps
                if step % train_cfg['eval_every'] == 0:
                    val_loss = evaluate(
                        model, val_loader, loss_fn, device
                    )
                    print(f"\n  📊 Validation Loss: {val_loss:.4f}\n")
                    log['steps'].append(step)
                    log['train_loss'].append(epoch_loss / batch_count)
                    log['val_loss'].append(val_loss)
                    log['time'].append(time.time() - start_time)

                    # Save log
                    with open('checkpoints/training_log.json', 'w') as f:
                        json.dump(log, f, indent=2)

                # Save checkpoint every 1000 steps
                if step % train_cfg['save_every'] == 0:
                    save_checkpoint(
                        model, optimizer, step,
                        epoch, loss.item(), config
                    )

            print(f"\n✅ Epoch {epoch+1} complete | "
                  f"Avg Loss: {epoch_loss/batch_count:.4f}\n")

            # Save at end of every epoch
            save_checkpoint(
                model, optimizer, step,
                epoch + 1, epoch_loss/batch_count, config
            )

    except KeyboardInterrupt:
        print("\n\nTraining paused by user.")
        print("Saving checkpoint...")
        save_checkpoint(
            model, optimizer, step,
            epoch, loss.item(), config
        )
        print("✅ Progress saved! Run train.py again to resume.")

    print("\n" + "=" * 55)
    print("Training session complete!")
    print(f"Total steps: {step:,}")
    print(f"Checkpoints saved in: checkpoints/")
    print("=" * 55)


if __name__ == "__main__":
    train()