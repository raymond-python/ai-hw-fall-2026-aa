"""Train a model on the MNIST TRAIN split.

Normal training:
    python train.py --model cnn
Adversarial training (defence): PGD examples are generated on the fly and the
model learns on them. Usually started from the normally trained model:
    python train.py --model cnn --adv-train --init saved_models/cnn.pth
"""
import argparse
import json
import os
import time

import torch
import torch.nn as nn

from attacks import pgd
from data import get_device, set_seed, train_val_loaders
from models import build_model, count_params

DEFAULTS = {"mlp": (10, 1e-3), "cnn": (8, 2e-3), "transformer": (15, 1e-3)}


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


def robust_accuracy(model, loader, device, eps, n=1000):
    """Accuracy under PGD on the first n validation images."""
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = pgd(model, x, y, eps, steps=10)
        with torch.no_grad():
            correct += (model(x_adv).argmax(1) == y).sum().item()
        total += y.size(0)
        if total >= n:
            break
    return 100.0 * correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(DEFAULTS))
    p.add_argument("--epochs", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--adv-train", action="store_true", help="PGD adversarial training")
    p.add_argument("--adv-eps", type=float, default=0.3)
    p.add_argument("--adv-steps", type=int, default=5)
    p.add_argument("--init", help="checkpoint to start from (fine-tuning)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()

    epochs, lr = DEFAULTS[args.model]
    if args.adv_train:
        epochs, lr = 4, 1e-3
    epochs = args.epochs or epochs
    lr = args.lr or lr
    tag = args.model + ("_advtrained" if args.adv_train else "")

    set_seed(args.seed)
    device = get_device()
    train_loader, val_loader = train_val_loaders(args.data_dir, args.batch_size, args.workers)
    model = build_model(args.model).to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device)["state_dict"])
    print(f"[{tag}] device={device} params={count_params(model):,} epochs={epochs} lr={lr}"
          + (f" adv-train eps={args.adv_eps} steps={args.adv_steps}" if args.adv_train else ""))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader))
    crit = nn.CrossEntropyLoss()

    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    ckpt_path = f"saved_models/{tag}.pth"
    history, best, t0 = [], -1.0, time.time()

    for epoch in range(1, epochs + 1):
        correct = total = 0
        loss_sum = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if args.adv_train:
                model.eval()                       # craft the attack with fixed BN/dropout
                x = pgd(model, x, y, args.adv_eps, steps=args.adv_steps)
            model.train()
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

        row = {"epoch": epoch, "train_loss": loss_sum / total,
               "train_acc": 100.0 * correct / total,
               "val_acc": accuracy(model, val_loader, device)}
        # adversarially trained model: keep the epoch that is most ROBUST on validation
        score = row["val_acc"]
        if args.adv_train:
            row["val_pgd_acc"] = robust_accuracy(model, val_loader, device, args.adv_eps)
            score = row["val_pgd_acc"]
        history.append(row)
        flag = ""
        if score > best:
            best = score
            torch.save({"arch": args.model, "tag": tag, "state_dict": model.state_dict(),
                        "epoch": epoch, "adv_train": args.adv_train}, ckpt_path)
            flag = "  *saved"
        extra = f"  val-PGD {row['val_pgd_acc']:6.2f}%" if args.adv_train else ""
        print(f"  epoch {epoch:2d}/{epochs}  train {row['train_acc']:6.2f}%  "
              f"val {row['val_acc']:6.2f}%{extra}{flag}", flush=True)

    info = {"tag": tag, "arch": args.model, "adv_train": args.adv_train,
            "adv_eps": args.adv_eps if args.adv_train else None,
            "params": count_params(model), "epochs": epochs, "lr": lr,
            "init": args.init, "train_time_sec": round(time.time() - t0, 1),
            "device": str(device), "history": history}
    with open(f"results/train_{tag}.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"[{tag}] done in {info['train_time_sec']:.0f}s -> {ckpt_path}")


if __name__ == "__main__":
    main()
