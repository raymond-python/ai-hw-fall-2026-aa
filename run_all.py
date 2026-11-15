"""Full pipeline: train the models, attack them, run the transfer test and
build the summary tables / charts in results/ (and the table in README.md).

    python run_all.py            # everything
    python run_all.py --quick    # tiny run to check the pipeline works
    python run_all.py --summary  # only rebuild tables/charts from results/
"""
import argparse
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["mlp", "cnn", "transformer", "cnn_advtrained"]
LABEL = {"mlp": "MLP", "cnn": "CNN", "transformer": "Transformer",
         "cnn_advtrained": "CNN (adv-trained)"}
ATTACK_NAMES = ["FGSM", "I-FGSM", "PGD", "MI-FGSM"]
ATTACK_COLORS = {"FGSM": "#f2a541", "I-FGSM": "#4c8bf5", "PGD": "#c62828", "MI-FGSM": "#2e7d32"}
MODEL_COLORS = {"mlp": "#8e8e8e", "cnn": "#1f4e8c", "transformer": "#7b3fa0",
                "cnn_advtrained": "#2e7d32"}
TABLE_EPS = [0.1, 0.3]


def run(cmd):
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def load_results():
    res = {}
    for t in ORDER:
        a, tr = f"results/attack_{t}.json", f"results/train_{t}.json"
        if os.path.exists(a):
            res[t] = json.load(open(a))
            res[t]["train"] = json.load(open(tr)) if os.path.exists(tr) else {}
    return res


def lookup(r, attack, eps):
    for row in r["results"]:
        if row["attack"] == attack and abs(row["eps"] - eps) < 1e-9:
            return row
    return None


def summarize():
    res = load_results()
    if not res:
        print("No results yet."); return
    tags = [t for t in ORDER if t in res]

    # full sweep as CSV
    with open("results/summary.csv", "w") as f:
        f.write("model,clean_test_acc,attack,eps,asr,robust_acc,mean_linf,mean_l2\n")
        for t in tags:
            for row in res[t]["results"]:
                f.write(f"{t},{res[t]['clean_test_acc']},{row['attack']},{row['eps']},"
                        f"{row['asr']},{row['robust_acc']},{row['linf']},{row['l2']}\n")

    # markdown tables
    any_r = res[tags[0]]
    md = [f"Recognition rate = accuracy on all {any_r['n_test']:,} test images before any attack. "
          f"ASR = share of correctly classified images that the attack turns into a wrong "
          f"answer, measured on the first {any_r['n_attack']:,} test images "
          f"({any_r['steps']} iterations for I-FGSM / PGD / MI-FGSM).\n"]
    for eps in TABLE_EPS:
        md.append(f"\n**Attack success rate at ε = {eps}** (pixels in [0, 1]; "
                  f"{eps} ≈ {round(eps * 255)}/255 grey levels)\n")
        md.append("| Model | Recognition rate (clean) | " + " | ".join(ATTACK_NAMES) + " |")
        md.append("|---|---:|" + "---:|" * len(ATTACK_NAMES))
        for t in tags:
            cells = []
            for a in ATTACK_NAMES:
                row = lookup(res[t], a, eps)
                cells.append(f"{row['asr']:.1f}%" if row else "–")
            md.append(f"| {LABEL[t]} | {res[t]['clean_test_acc']:.2f}% | " + " | ".join(cells) + " |")
    table = "\n".join(md) + "\n"
    with open("results/summary.md", "w") as f:
        f.write(table)
    with open("results/summary.json", "w") as f:
        json.dump({t: {k: v for k, v in res[t].items()} for t in tags}, f, indent=2)

    # chart 1: ASR vs eps, one panel per model
    fig, axes = plt.subplots(1, len(tags), figsize=(4.2 * len(tags), 3.8), sharey=True)
    axes = axes if len(tags) > 1 else [axes]
    for ax, t in zip(axes, tags):
        for a in ATTACK_NAMES:
            pts = sorted((r["eps"], r["asr"]) for r in res[t]["results"] if r["attack"] == a)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", ms=4, lw=2,
                    color=ATTACK_COLORS[a], label=a)
        ax.set_title(f"{LABEL[t]}  (clean {res[t]['clean_test_acc']:.2f}%)", fontsize=10)
        ax.set_xlabel("ε (max change per pixel)")
        ax.set_ylim(-3, 103); ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("attack success rate (%)")
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig("results/asr_vs_eps.png", dpi=150); plt.close(fig)

    # chart 2: accuracy under attack vs eps, one panel per attack, lines = models
    fig, axes = plt.subplots(1, len(ATTACK_NAMES), figsize=(16, 3.8), sharey=True)
    for ax, a in zip(axes, ATTACK_NAMES):
        for t in tags:
            pts = sorted((r["eps"], r["robust_acc"]) for r in res[t]["results"] if r["attack"] == a)
            ax.plot([0] + [p[0] for p in pts],
                    [res[t]["attack_set_clean_acc"]] + [p[1] for p in pts],
                    "-o", ms=4, lw=2, color=MODEL_COLORS[t], label=LABEL[t])
        ax.set_title(a); ax.set_xlabel("ε"); ax.set_ylim(-3, 103); ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("accuracy under attack (%)")
    axes[-1].legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout(); fig.savefig("results/accuracy_vs_eps.png", dpi=150); plt.close(fig)

    # README table
    if os.path.exists("README.md"):
        readme = open("README.md").read()
        a, b = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
        if a in readme and b in readme:
            readme = readme[:readme.index(a) + len(a)] + "\n" + table + readme[readme.index(b):]
            open("README.md", "w").write(readme)
    print("\n" + table)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--summary", action="store_true")
    p.add_argument("--workers", default="2")
    args = p.parse_args()
    py = sys.executable
    os.makedirs("results", exist_ok=True)

    if not args.summary:
        q_train = ["--epochs", "1"] if args.quick else []
        q_test = ["--n-attack", "200", "--steps", "3"] if args.quick else []
        for m in ["mlp", "cnn", "transformer"]:
            run([py, "train.py", "--model", m, "--workers", args.workers] + q_train)
        run([py, "train.py", "--model", "cnn", "--adv-train", "--init", "saved_models/cnn.pth",
             "--workers", args.workers] + q_train)
        for t in ORDER:
            run([py, "test.py", "--model-path", f"saved_models/{t}.pth",
                 "--workers", args.workers] + q_test)
        run([py, "transfer.py"])
    summarize()


if __name__ == "__main__":
    main()
