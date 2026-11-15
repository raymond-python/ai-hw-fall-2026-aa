"""Test a trained model on the MNIST TEST split, before and after attacks.

Metrics
  * Recognition rate (clean accuracy) on all 10,000 test images.
  * Attack success rate (ASR) for FGSM, I-FGSM, PGD and MI-FGSM at several
    epsilons:  ASR = (images the model got RIGHT before the attack and WRONG
    after it) / (images the model got right before the attack).
  * Accuracy under attack (robust accuracy) on the same images.

Attacks run on the first --n-attack test images (default 2,000) to keep CPU
time reasonable; every model is attacked on exactly the same images.

    python test.py --model-path saved_models/cnn.pth
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from attacks import ATTACKS
from data import attack_set, get_device, set_seed, tensor_loader, test_loader
from models import build_model

EPSILONS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
TRANSFER_EPS = 0.2      # adversarial images saved at this eps for the transfer test
SHOW_EPS = 0.2          # eps used in the example picture


def load(path, device):
    ckpt = torch.load(path, map_location=device)
    model = build_model(ckpt["arch"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)       # we need gradients w.r.t. the INPUT only
    return model, ckpt


@torch.no_grad()
def predict(model, x, device, bs=1000):
    return torch.cat([model(x[i:i + bs].to(device)).argmax(1).cpu()
                      for i in range(0, len(x), bs)])


def run_attack(model, attack, x, y, eps, device, steps):
    out = []
    for xb, yb in tensor_loader(x, y):
        out.append(attack(model, xb.to(device), yb.to(device), eps, steps=steps).cpu())
    return torch.cat(out)


def example_figure(model, x, y, device, tag, path, steps, k=8):
    set_seed(1)
    xs, ys = x[:k], y[:k]
    rows = [("original", xs)]
    for name, atk in ATTACKS.items():
        rows.append((name, atk(model, xs.to(device), ys.to(device), SHOW_EPS,
                               steps=steps).cpu()))
    fig, axes = plt.subplots(len(rows), k, figsize=(k * 1.15, len(rows) * 1.35))
    for r, (name, imgs) in enumerate(rows):
        preds = predict(model, imgs, device)
        for c in range(k):
            ax = axes[r, c]
            ax.imshow(imgs[c, 0].numpy(), cmap="gray_r", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            ok = preds[c].item() == ys[c].item()
            ax.set_title(f"pred {preds[c].item()}", fontsize=8,
                         color="#1a7f37" if ok else "#c62828",
                         fontweight="normal" if ok else "bold")
        axes[r, 0].set_ylabel(name, fontsize=9, rotation=0, ha="right", va="center")
    fig.suptitle(f"{tag}: adversarial examples at ε = {SHOW_EPS}  "
                 "(green = still correct, red = fooled)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--n-attack", type=int, default=2000)
    p.add_argument("--steps", type=int, default=10, help="iterations for I-FGSM/PGD/MI-FGSM")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()

    device = get_device()
    model, ckpt = load(args.model_path, device)
    tag = ckpt["tag"]

    # 1. Recognition rate before attacks: full test set
    correct = total = 0
    with torch.no_grad():
        for xb, yb in test_loader(args.data_dir, workers=args.workers):
            correct += (model(xb.to(device)).argmax(1).cpu() == yb).sum().item()
            total += yb.size(0)
    clean_acc = 100.0 * correct / total
    print(f"[{tag}] clean test accuracy (recognition rate): {clean_acc:.2f}% on {total} images")

    # 2. Attacks
    x, y = attack_set(args.data_dir, args.n_attack)
    clean_pred = predict(model, x, device)
    was_right = clean_pred == y
    n_right = int(was_right.sum())
    print(f"[{tag}] attack set: {len(y)} test images, {n_right} classified correctly before attack")

    os.makedirs("results", exist_ok=True)
    os.makedirs("adv_cache", exist_ok=True)
    rows = []
    for eps in EPSILONS:
        for name, atk in ATTACKS.items():
            set_seed(0)
            t0 = time.time()
            x_adv = run_attack(model, atk, x, y, eps, device, args.steps)
            secs = time.time() - t0
            adv_pred = predict(model, x_adv, device)
            fooled = was_right & (adv_pred != y)
            delta = (x_adv - x).flatten(1)
            row = {"attack": name, "eps": eps,
                   "asr": round(100.0 * int(fooled.sum()) / max(n_right, 1), 2),
                   "robust_acc": round(100.0 * (adv_pred == y).float().mean().item(), 2),
                   "linf": round(delta.abs().max(1).values.mean().item(), 4),
                   "l2": round(delta.norm(dim=1).mean().item(), 3),
                   "seconds": round(secs, 1)}
            rows.append(row)
            print(f"  eps {eps:.2f}  {name:8s} ASR {row['asr']:6.2f}%  "
                  f"acc-under-attack {row['robust_acc']:6.2f}%  ({secs:.0f}s)", flush=True)
            if abs(eps - TRANSFER_EPS) < 1e-9:
                torch.save(x_adv, f"adv_cache/{tag}_{name}.pt")

    torch.save({"x": x, "y": y}, "adv_cache/clean.pt")
    example_figure(model, x, y, device, tag, f"results/examples_{tag}.png", args.steps)

    result = {"tag": tag, "arch": ckpt["arch"], "adv_train": bool(ckpt.get("adv_train")),
              "clean_test_acc": round(clean_acc, 2), "n_test": total,
              "n_attack": len(y), "n_attack_correct": n_right,
              "attack_set_clean_acc": round(100.0 * n_right / len(y), 2),
              "steps": args.steps, "epsilons": EPSILONS, "results": rows}
    with open(f"results/attack_{tag}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{tag}] saved results/attack_{tag}.json")


if __name__ == "__main__":
    main()
