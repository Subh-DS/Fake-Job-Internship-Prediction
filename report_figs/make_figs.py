"""Generate report figures from verified notebook outputs. No invention: all values
copied from executed cell outputs in noteX.ipynb (Exp1) and
Experiment_2_Multimodal_ResNet50.ipynb (Exp2)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

FIG = r"C:\DEV\FAKE ML\report_figs"
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "figure.dpi": 200, "savefig.bbox": "tight",
})
BLUE, RED, GREEN, ORANGE, GREY = "#2E5AAC", "#C0392B", "#1E8449", "#D4851A", "#5D6D7E"

# ---- Fig 1: class distribution ----
fig, ax = plt.subplots(figsize=(4.6, 2.8))
cats = ["Genuine (0)", "Fake (1)"]
vals = [17014, 866]
bars = ax.bar(cats, vals, color=[BLUE, RED], width=0.55)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, b.get_height()*1.02,
            f"{v:,}\n({v/17880*100:.2f}%)", ha="center", fontsize=9)
ax.set_title("Class distribution (n = 17,880)")
ax.set_ylabel("Postings")
fig.savefig(f"{FIG}/fig1_class_dist.png"); plt.close(fig)

# ---- Fig 2: classical baselines + Exp1 deep (FAKE-class P/R/F1) ----
methods = ["LR-Normal", "LR-SMOTE", "LR-ADASYN", "Exp1 Deep"]
prec = [0.9375, 0.5205, 0.4258, 0.7120]
rec  = [0.3468, 0.8786, 0.8960, 0.7861]
f1   = [0.5063, 0.6538, 0.5773, 0.7473]
x = np.arange(len(methods)); w = 0.24
fig, ax = plt.subplots(figsize=(5.6, 3.0))
ax.bar(x-w, prec, w, label="FAKE precision", color=BLUE)
ax.bar(x,   rec,  w, label="FAKE recall", color=RED)
ax.bar(x+w, f1,   w, label="FAKE F1", color=GREEN)
ax.set_xticks(x, methods, fontsize=8)
ax.set_ylim(0, 1.05); ax.set_ylabel("Score"); ax.legend(fontsize=8, ncol=3)
ax.set_title("Classical baselines vs Experiment 1 deep model (test set)")
fig.savefig(f"{FIG}/fig2_baselines.png"); plt.close(fig)

# ---- Fig 3 & 4: confusion matrices ----
def plot_cm(cm, title, path):
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Pred Genuine", "Pred Fake"])
    ax.set_yticks([0, 1], ["True Genuine", "True Fake"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="white" if cm[i, j] > cm.max()/2 else "black")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(path); plt.close(fig)

plot_cm(np.array([[3348, 55], [37, 136]]),
        "Experiment 1 — test CM (thr 0.50)", f"{FIG}/fig3_cm_exp1.png")
plot_cm(np.array([[3397, 6], [57, 116]]),
        "Experiment 2 — test CM (thr 0.40)", f"{FIG}/fig4_cm_exp2.png")

# ---- Fig 5: Exp1 vs Exp2 metrics ----
labels = ["Accuracy", "FAKE Prec.", "FAKE Rec.", "FAKE F1", "Macro F1"]
e1 = [0.9743, 0.7120, 0.7861, 0.7473, 0.8700]
e2 = [0.9824, 0.9508, 0.6705, 0.7864, 0.8886]
x = np.arange(len(labels)); w = 0.35
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.2, 2.9))
a1.bar(x-w/2, e1, w, label="Experiment 1", color=GREY)
a1.bar(x+w/2, e2, w, label="Experiment 2", color=BLUE)
a1.set_xticks(x, labels, fontsize=7); a1.set_ylim(0.5, 1.02)
a1.legend(fontsize=8); a1.set_title("Test metrics")
a2.bar(["Exp1 FP", "Exp2 FP", "Exp1 FN", "Exp2 FN"], [55, 6, 37, 57],
       color=[ORANGE, BLUE, ORANGE, RED], width=0.6)
a2.set_title("Error counts (test, n=3,576)")
for i, v in enumerate([55, 6, 37, 57]):
    a2.text(i, v + 1, str(v), ha="center", fontsize=9, fontweight="bold")
fig.suptitle("Experiment 1 vs Experiment 2 (test set)", fontweight="bold")
fig.savefig(f"{FIG}/fig5_exp1_exp2.png"); plt.close(fig)

# ---- Fig 6: Exp2 threshold search (validation) ----
thr = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
vp = [0.8938, 0.9018, 0.9099, 0.9099, 0.9174, 0.9174, 0.9167, 0.9159, 0.9327]
vr = [0.7266, 0.7266, 0.7266, 0.7266, 0.7194, 0.7194, 0.7122, 0.7050, 0.6978]
vf = [0.8016, 0.8048, 0.8080, 0.8080, 0.8065, 0.8065, 0.8016, 0.7967, 0.7984]
fig, ax = plt.subplots(figsize=(5.4, 2.9))
ax.plot(thr, vp, "o-", label="FAKE precision", color=BLUE, ms=3)
ax.plot(thr, vr, "s-", label="FAKE recall", color=RED, ms=3)
ax.plot(thr, vf, "D-", label="FAKE F1", color=GREEN, ms=3)
ax.axvline(0.40, color="k", ls="--", lw=1)
ax.text(0.405, 0.73, "selected 0.40", fontsize=8)
ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
ax.legend(fontsize=8); ax.set_title("Exp2 threshold search (validation, n=2,861)")
fig.savefig(f"{FIG}/fig6_threshold.png"); plt.close(fig)

# ---- Fig 7: Exp2 training history ----
ep = np.arange(1, 12)
tl = [1.0284, 1.3141, 1.2673, 1.1228, 1.0052, 0.8976, 0.7799, 0.6789, 0.6423, 0.5698, 0.5074]
vl = [0.5956, 0.5563, 0.5517, 0.6699, 0.4748, 0.4471, 0.5310, 0.5446, 0.5888, 1.1329, 0.6518]
vf1 = [0.3820, 0.6256, 0.7265, 0.6426, 0.7717, 0.8065, 0.7786, 0.8000, 0.7833, 0.7260, 0.7043]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.2, 2.9))
a1.plot(ep, tl, "o-", ms=3, label="Train loss", color=GREY)
a1.plot(ep, vl, "s-", ms=3, label="Val loss", color=BLUE)
a1.set_xlabel("Epoch"); a1.legend(fontsize=8); a1.set_title("Loss")
a1.grid(alpha=0.3)
a2.plot(ep, vf1, "D-", ms=3, color=GREEN)
a2.axvline(6, color="k", ls="--", lw=1)
a2.text(6.15, 0.42, "best ep.6", fontsize=8)
a2.set_xlabel("Epoch"); a2.set_ylabel("Val FAKE F1"); a2.set_title("Val FAKE F1 (early stop ep.11)")
a2.grid(alpha=0.3)
fig.suptitle("Experiment 2 training history", fontweight="bold")
fig.savefig(f"{FIG}/fig7_training.png"); plt.close(fig)

print("figures done:", sorted(os.listdir(FIG)))
