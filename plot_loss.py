import re, glob, matplotlib.pyplot as plt

log_files = {"train_v1": "train.log", "train_v2": "train_v2.log"}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("JARVIS Training Loss", fontsize=14, fontweight='bold')

for ax, (name, logfile) in zip(axes, log_files.items()):
    if not __import__('os').path.exists(logfile):
        ax.set_title(f"{name} — no log yet")
        continue
    steps, losses = [], []
    with open(logfile) as f:
        for line in f:
            m = re.search(r'Step\s+([\d,]+).*Loss\s+([\d.]+)', line)
            if m:
                steps.append(int(m.group(1).replace(',', '')))
                losses.append(float(m.group(2)))
    if steps:
        ax.plot(steps, losses, linewidth=1.5)
        ax.set_title(name)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
    else:
        ax.set_title(f"{name} — no data yet")

plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150)
print("Saved loss_curve.png")
plt.show()