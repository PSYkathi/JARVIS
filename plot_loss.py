import os
import glob
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

CHECKPOINT_DIR = "D:/JARVIS/checkpoints_v3"
OUTPUT_FILE    = "D:/JARVIS/loss_curve.png"

def load_losses():
    train_steps, train_losses = [], []
    val_steps,   val_losses   = [], []

    # Load all step checkpoints
    ckpts = sorted(
        glob.glob(f"{CHECKPOINT_DIR}/jarvis_v3_step*.pt"),
        key=lambda p: int(os.path.basename(p)
                         .replace("jarvis_v3_step","")
                         .replace(".pt",""))
    )

    for path in ckpts:
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            step = ckpt.get("step")
            t    = ckpt.get("train_loss")
            v    = ckpt.get("val_loss")

            if step and t:
                train_steps.append(step)
                train_losses.append(t)
            if step and v:
                val_steps.append(step)
                val_losses.append(v)
        except Exception as e:
            print(f"⚠️  Skipping {os.path.basename(path)}: {e}")

    # Also load best checkpoint
    best_path = f"{CHECKPOINT_DIR}/jarvis_v3_best.pt"
    best_step, best_val = None, None
    if os.path.exists(best_path):
        try:
            ckpt     = torch.load(best_path, map_location="cpu", weights_only=True)
            best_step = ckpt.get("step")
            best_val  = ckpt.get("val_loss")
        except:
            pass

    return train_steps, train_losses, val_steps, val_losses, best_step, best_val


def plot(train_steps, train_losses, val_steps, val_losses, best_step, best_val):
    fig, ax = plt.subplots(figsize=(12, 6))

    # Background
    fig.patch.set_facecolor("#0f0f0f")
    ax.set_facecolor("#1a1a1a")

    # Plot lines
    ax.plot(train_steps, train_losses,
            color="#4f98a3", linewidth=1.5,
            alpha=0.7, label="Train Loss")

    ax.plot(val_steps, val_losses,
            color="#f0a500", linewidth=2.0,
            label="Val Loss")

    # Mark best val loss
    if best_step and best_val:
        ax.scatter([best_step], [best_val],
                   color="#ff4f4f", s=100, zorder=5,
                   label=f"Best Val: {best_val:.4f} @ step {best_step:,}")
        ax.axhline(y=best_val, color="#ff4f4f",
                   linestyle="--", linewidth=0.8, alpha=0.5)

    # Phase target lines
    ax.axhline(y=2.0, color="#6daa45", linestyle=":",
               linewidth=1.0, alpha=0.7, label="Phase 3 Target (2.0)")
    ax.axhline(y=1.5, color="#a86fdf", linestyle=":",
               linewidth=1.0, alpha=0.7, label="Phase 4 Target (1.5)")

    # Labels & styling
    ax.set_title("JARVIS v3 — Training Loss Curve",
                 color="white", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Training Step", color="#aaaaaa", fontsize=12)
    ax.set_ylabel("Loss",          color="#aaaaaa", fontsize=12)

    ax.tick_params(colors="#aaaaaa")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"))

    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    ax.grid(color="#2a2a2a", linewidth=0.8)
    ax.legend(facecolor="#1a1a1a", edgecolor="#333333",
              labelcolor="white", fontsize=10)

    # Stats box
    if val_losses:
        stats = (f"Current Best Val: {min(val_losses):.4f}\n"
                 f"Total Steps Logged: {len(train_steps)}\n"
                 f"Latest Step: {train_steps[-1]:,}")
        ax.text(0.98, 0.97, stats,
                transform=ax.transAxes,
                fontsize=9, color="#aaaaaa",
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(facecolor="#111111",
                          edgecolor="#333333",
                          boxstyle="round,pad=0.5"))

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150,
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"✅ Loss curve saved: {OUTPUT_FILE}")


def main():
    print("📊 Loading checkpoints...")
    train_steps, train_losses, val_steps, val_losses, best_step, best_val = load_losses()

    if not train_steps:
        print("❌ No checkpoints found!")
        return

    print(f"   Found {len(train_steps)} train points")
    print(f"   Found {len(val_steps)} val points")
    if best_step:
        print(f"   Best val: {best_val:.4f} @ step {best_step:,}")

    print("🎨 Plotting...")
    plot(train_steps, train_losses, val_steps, val_losses, best_step, best_val)


if __name__ == "__main__":
    main()