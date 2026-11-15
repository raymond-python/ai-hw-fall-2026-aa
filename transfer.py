"""Transfer (black-box) test: adversarial images crafted against one model
(the source) are shown to a DIFFERENT model (the target), whose gradients the
attacker never saw. Uses the images test.py saved at eps = 0.2.

    python transfer.py
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from attacks import ATTACKS
from data import get_device
from test import TRANSFER_EPS, load, predict

ORDER = ["mlp", "cnn", "transformer", "cnn_advtrained"]
LABEL = {"mlp": "MLP", "cnn": "CNN", "transformer": "Transformer",
         "cnn_advtrained": "CNN (adv-trained)"}


def main():
    device = get_device()
    tags = [t for t in ORDER if os.path.exists(f"saved_models/{t}.pth")]
    models = {t: load(f"saved_models/{t}.pth", device)[0] for t in tags}
    clean = torch.load("adv_cache/clean.pt")
    y = clean["y"]
    right = {t: predict(models[t], clean["x"], device) == y for t in tags}

    out = {"eps": TRANSFER_EPS, "models": tags, "asr": {}}
    for atk in ATTACKS:
        mat = np.full((len(tags), len(tags)), np.nan)
        for i, src in enumerate(tags):
            path = f"adv_cache/{src}_{atk}.pt"
            if not os.path.exists(path):
                continue
            x_adv = torch.load(path)
            for j, tgt in enumerate(tags):
                pred = predict(models[tgt], x_adv, device)
                fooled = right[tgt] & (pred != y)
                mat[i, j] = 100.0 * fooled.sum().item() / max(right[tgt].sum().item(), 1)
        out["asr"][atk] = [[None if np.isnan(v) else round(float(v), 2) for v in r]
                           for r in mat]

    os.makedirs("results", exist_ok=True)
    with open("results/transfer.json", "w") as f:
        json.dump(out, f, indent=2)

    n = len(tags)
    fig, axes = plt.subplots(1, len(ATTACKS), figsize=(4.2 * len(ATTACKS), 4.4))
    for ax, atk in zip(axes, ATTACKS):
        mat = np.array([[np.nan if v is None else v for v in r] for r in out["asr"][atk]])
        ax.imshow(mat, cmap="Reds", vmin=0, vmax=100)
        for i in range(n):
            for j in range(n):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=10,
                            color="white" if mat[i, j] > 60 else "black",
                            fontweight="bold" if i == j else "normal")
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels([LABEL[t] for t in tags], rotation=35, ha="right", fontsize=8)
        ax.set_yticklabels([LABEL[t] for t in tags], fontsize=8)
        ax.set_title(atk, fontsize=11)
        ax.set_xlabel("target (attacked blind)", fontsize=8)
    axes[0].set_ylabel("source (gradients used)", fontsize=9)
    fig.suptitle(f"Attack success rate (%) when transferring adversarial images, ε = {TRANSFER_EPS}"
                 "   (diagonal = white-box)", fontsize=11)
    fig.tight_layout()
    fig.savefig("results/transfer_heatmap.png", dpi=150)
    plt.close(fig)
    for atk in ATTACKS:
        print(atk, out["asr"][atk])


if __name__ == "__main__":
    main()
