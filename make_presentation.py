"""Build presentation/Adversarial_Attacks_MNIST.pdf from the real numbers in
results/. Run after run_all.py.

    python make_presentation.py
"""
import json
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

W, H = 13.333, 7.5
INK, MUTED, ACCENT, RED, BG = "#1b1f24", "#5b6470", "#1f4e8c", "#c62828", "#ffffff"
ORDER = ["mlp", "cnn", "transformer", "cnn_advtrained"]
LABEL = {"mlp": "MLP", "cnn": "CNN", "transformer": "Transformer",
         "cnn_advtrained": "CNN (adv-trained)"}
ATTACKS = ["FGSM", "I-FGSM", "PGD", "MI-FGSM"]
OUT = "presentation/Adversarial_Attacks_MNIST.pdf"


def slide(title=None, subtitle=None):
    fig = plt.figure(figsize=(W, H), facecolor=BG)
    if title:
        fig.text(0.05, 0.90, title, fontsize=28, fontweight="bold", color=INK)
        fig.add_artist(plt.Line2D([0.05, 0.12], [0.875, 0.875], color=ACCENT, lw=4))
    if subtitle:
        fig.text(0.05, 0.82, subtitle, fontsize=14, color=MUTED)
    return fig


def bullets(fig, items, x=0.06, y=0.72, size=17, width=60):
    line_h = size * 1.4 / 72 / H
    for it in items:
        lines = textwrap.wrap(it, width)
        fig.text(x, y, "•", fontsize=size, color=ACCENT, fontweight="bold", va="top")
        fig.text(x + 0.022, y, "\n".join(lines), fontsize=size, color=INK, va="top",
                 linespacing=1.4)
        y -= line_h * len(lines) + line_h * 0.7
    return y


def image(fig, path, rect):
    if not os.path.exists(path):
        return
    ax = fig.add_axes(rect)
    ax.imshow(mpimg.imread(path))
    ax.axis("off")


def footer(fig, n):
    fig.text(0.95, 0.03, str(n), fontsize=11, color=MUTED, ha="right")
    fig.text(0.05, 0.03, "ai-hw-fall-2026-aa · adversarial attacks on MNIST",
             fontsize=11, color=MUTED)


def asr(res, tag, attack, eps):
    for r in res[tag]["results"]:
        if r["attack"] == attack and abs(r["eps"] - eps) < 1e-9:
            return r["asr"]
    return None


def table(fig, rect, header, cells, highlight_rows=(), size=14):
    ax = fig.add_axes(rect); ax.axis("off")
    tb = ax.table(cellText=cells, colLabels=header, loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(size); tb.scale(1, 2.3)
    for (i, j), c in tb.get_celld().items():
        c.set_edgecolor("#d6dde6")
        if i == 0:
            c.set_facecolor(ACCENT); c.get_text().set_color("white")
            c.get_text().set_fontweight("bold")
        elif (i - 1) in highlight_rows:
            c.set_facecolor("#e6f2e8")


def main():
    res = json.load(open("results/summary.json"))
    tags = [t for t in ORDER if t in res]
    transfer = json.load(open("results/transfer.json")) if os.path.exists("results/transfer.json") else None
    std = [t for t in tags if t != "cnn_advtrained"]
    adv = "cnn_advtrained" in tags
    any_r = res[tags[0]]
    os.makedirs("presentation", exist_ok=True)

    with PdfPages(OUT) as pdf:
        n = 1
        # 1. Title
        fig = slide()
        fig.text(0.06, 0.60, "Fooling a Digit Recognizer", fontsize=46,
                 fontweight="bold", color=INK)
        fig.text(0.06, 0.52, "Adversarial attacks on MNIST: FGSM, I-FGSM / PGD, MI-FGSM",
                 fontsize=20, color=MUTED)
        fig.add_artist(plt.Line2D([0.06, 0.16], [0.46, 0.46], color=ACCENT, lw=5))
        c = asr(res, "cnn", "PGD", 0.1) if "cnn" in res else None
        if c is not None:
            fig.text(0.06, 0.36, f"Changing each pixel by at most 0.1 (≈26/255), PGD fools "
                     f"our {res['cnn']['clean_test_acc']:.2f}%-accurate CNN on {c:.1f}% of digits.",
                     fontsize=17, color=RED, fontweight="bold")
        fig.text(0.06, 0.12, "AI homework · Fall 2026", fontsize=14, color=MUTED)
        pdf.savefig(fig); plt.close(fig); n += 1

        # 2. Setup & metrics
        fig = slide("Setup and metrics", "White-box, untargeted attacks with a pixel budget ε")
        bullets(fig, [
            "Models trained on the 60,000-image MNIST train split; everything below is "
            "measured on the test split.",
            f"Recognition rate = accuracy on all {any_r['n_test']:,} clean test images, "
            "before any attack.",
            "Attack success rate (ASR) = of the images the model gets right, the share "
            "the attack turns into a wrong answer.",
            "Threat model: the attacker knows the model and its gradients, and may change "
            "every pixel by at most ε (L∞ norm); pixels stay in [0, 1].",
            f"Attacks evaluated on the first {any_r['n_attack']:,} test images, ε from 0.05 "
            f"to 0.3, {any_r['steps']} iterations for the iterative attacks.",
        ], width=88, size=17)
        footer(fig, n); pdf.savefig(fig); plt.close(fig); n += 1

        # 3. Attacks
        fig = slide("Four attacks", "All push the image in the direction that increases "
                    "the loss of the true class")
        cards = [
            ("FGSM", "one step", r"$x' = x + \epsilon\cdot \mathrm{sign}(\nabla_x L)$",
             ["Goodfellow et al. 2015", "Fast: 1 gradient", "Assumes the loss is", "roughly linear"]),
            ("I-FGSM", "many small steps", r"$x_{t+1} = \mathrm{clip}_\epsilon(x_t + \alpha\,\mathrm{sign}(\nabla L))$",
             ["Kurakin et al. 2017", "α = ε / steps", "Follows the curved", "loss surface"]),
            ("PGD", "random start", r"$x_0 = x + U(-\epsilon,\epsilon)$, then I-FGSM",
             ["Madry et al. 2018", "α = 2.5 ε / steps", "Standard strongest", "first-order attack"]),
            ("MI-FGSM", "momentum", r"$g_{t+1} = \mu g_t + \nabla L / \|\nabla L\|_1$",
             ["Dong et al. 2018", "μ = 1, α = ε / steps", "Stable direction →", "transfers better"]),
        ]
        for i, (name, sub, formula, pts) in enumerate(cards):
            x = 0.05 + i * 0.23
            fig.add_artist(plt.Rectangle((x, 0.14), 0.215, 0.62, transform=fig.transFigure,
                                         facecolor="#f3f6fa", edgecolor="#d6dde6"))
            fig.text(x + 0.015, 0.69, name, fontsize=22, fontweight="bold", color=ACCENT)
            fig.text(x + 0.015, 0.645, sub, fontsize=13, color=MUTED)
            fig.text(x + 0.015, 0.57, formula, fontsize=12.5, color=INK)
            for j, t in enumerate(pts):
                fig.text(x + 0.015, 0.48 - j * 0.07, t, fontsize=13.5, color=INK)
        footer(fig, n); pdf.savefig(fig); plt.close(fig); n += 1

        # 4. Models
        fig = slide("Models under attack", "Same architectures as the previous homework")
        header = ["Model", "Params", "Training", "Recognition rate (clean test)"]
        cells = []
        for t in tags:
            tr = res[t].get("train", {})
            how = (f"PGD adv. training (ε={tr.get('adv_eps', 0.3)})"
                   if res[t]["adv_train"] else "normal")
            cells.append([LABEL[t], f"{tr.get('params', 0):,}", how,
                          f"{res[t]['clean_test_acc']:.2f}%"])
        table(fig, [0.06, 0.30, 0.88, 0.45], header, cells,
              highlight_rows=[i for i, t in enumerate(tags) if t == "cnn_advtrained"], size=15)
        if adv:
            fig.text(0.06, 0.20, "The adversarially trained CNN is the normal CNN fine-tuned on PGD "
                     "images generated on the fly — the defence we test at the end.",
                     fontsize=14, color=MUTED)
        footer(fig, n); pdf.savefig(fig); plt.close(fig); n += 1

        # 5. Examples
        fig = slide("What the attack looks like", "CNN, ε = 0.2 — a human still reads "
                    "every digit; the model does not")
        image(fig, "results/examples_cnn.png", [0.10, 0.06, 0.80, 0.74])
        footer(fig, n); pdf.savefig(fig); plt.close(fig); n += 1

        # 6. Tables
        fig = slide("Attack success rate", "Share of correctly classified test digits "
                    "that the attack flips")
        for k, eps in enumerate([0.1, 0.3]):
            cells = [[LABEL[t], f"{res[t]['clean_test_acc']:.2f}%"] +
                     [f"{asr(res, t, a, eps):.1f}%" if asr(res, t, a, eps) is not None else "–"
                      for a in ATTACKS] for t in tags]
            fig.text(0.06, 0.76 - k * 0.36, f"ε = {eps}  (≈{round(eps * 255)}/255 per pixel)",
                     fontsize=15, fontweight="bold", color=INK)
            table(fig, [0.06, 0.45 - k * 0.36, 0.88, 0.29], ["Model", "Clean acc"] + ATTACKS,
                  cells, highlight_rows=[i for i, t in enumerate(tags) if t == "cnn_advtrained"],
                  size=13)
        footer(fig, n); pdf.savefig(fig); plt.close(fig); n += 1

        # 7. ASR vs eps
        fig = slide("Bigger budget, more fooled", "Attack success rate vs ε for each model")
        image(fig, "results/asr_vs_eps.png", [0.03, 0.12, 0.94, 0.66])
        footer(fig, n); pdf.savefig(fig); plt.close(fig); n += 1

        # 8. Accuracy under attack
        fig = slide("Accuracy under attack", "Same data seen from the defender's side")
        image(fig, "results/accuracy_vs_eps.png", [0.03, 0.14, 0.94, 0.62])
        footer(fig, n); pdf.savefig(fig); plt.close(fig); n += 1

        # 9. Transfer
        if transfer and os.path.exists("results/transfer_heatmap.png"):
            fig = slide("Black-box transfer", f"Adversarial images made on one model, "
                        f"shown to another (ε = {transfer['eps']})")
            image(fig, "results/transfer_heatmap.png", [0.03, 0.10, 0.94, 0.68])
            footer(fig, n); pdf.savefig(fig); plt.close(fig); n += 1

        # 10. Defence
        if adv and "cnn" in res:
            fig = slide("Defence: adversarial training", "Train on PGD images instead of "
                        "clean ones")
            image(fig, "results/examples_cnn_advtrained.png", [0.46, 0.08, 0.52, 0.72])
            pts = [f"Clean accuracy: {res['cnn']['clean_test_acc']:.2f}% → "
                   f"{res['cnn_advtrained']['clean_test_acc']:.2f}%."]
            for eps in [0.1, 0.3]:
                a, b = asr(res, "cnn", "PGD", eps), asr(res, "cnn_advtrained", "PGD", eps)
                if a is not None and b is not None:
                    pts.append(f"PGD success at ε = {eps}: {a:.1f}% → {b:.1f}%.")
            pts.append("Costs ~6× more compute per epoch (5 PGD steps per batch).")
            bullets(fig, pts, width=40, size=17)
            footer(fig, n); pdf.savefig(fig); plt.close(fig); n += 1

        # 11. Takeaways, derived from the numbers
        fig = slide("Takeaways")
        pts = []
        if "cnn" in res:
            f1, p1 = asr(res, "cnn", "FGSM", 0.1), asr(res, "cnn", "PGD", 0.1)
            if f1 is not None and p1 is not None:
                pts.append(f"Iterative attacks beat one-step FGSM: on the CNN at ε = 0.1, "
                           f"FGSM {f1:.1f}% vs PGD {p1:.1f}% success.")
        weakest = max(std, key=lambda t: asr(res, t, "FGSM", 0.1) or 0)
        pts.append(f"Easiest to fool with FGSM at ε = 0.1: {LABEL[weakest]} "
                   f"({asr(res, weakest, 'FGSM', 0.1):.1f}%). High clean accuracy does not "
                   "mean robustness.")
        if transfer:
            names = transfer["models"]
            def off_diag(atk):
                m = transfer["asr"][atk]
                v = [m[i][j] for i in range(len(names)) for j in range(len(names))
                     if i != j and m[i][j] is not None
                     and names[i] != "cnn_advtrained" and names[j] != "cnn_advtrained"]
                return sum(v) / len(v) if v else None
            tr = {a: off_diag(a) for a in ATTACKS}
            tr = {a: v for a, v in tr.items() if v is not None}
            if tr:
                best = max(tr, key=tr.get)
                pts.append("Average black-box transfer between the normal models: "
                           + ", ".join(f"{a} {v:.0f}%" for a, v in tr.items())
                           + f" — best: {best}.")
        if adv and "cnn" in res:
            a, b = asr(res, "cnn", "PGD", 0.3), asr(res, "cnn_advtrained", "PGD", 0.3)
            if a is not None and b is not None:
                pts.append(f"Adversarial training is the defence that works: PGD success at "
                           f"ε = 0.3 drops from {a:.1f}% to {b:.1f}%.")
        bullets(fig, pts, size=17, width=80)
        footer(fig, n); pdf.savefig(fig); plt.close(fig); n += 1

    print(f"Saved {OUT} ({n - 1} slides)")


if __name__ == "__main__":
    main()
