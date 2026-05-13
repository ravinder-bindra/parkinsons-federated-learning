"""
src/evaluate.py
---------------
Generates the full evaluation suite from the trained multi-modal
federated model. Reads results/global_model.npz + the per-modality
test prediction CSVs, then writes plots to results/figures/.

Run:
    python -m src.evaluate
"""

from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (auc, brier_score_loss, confusion_matrix,
                             precision_recall_curve, roc_auc_score,
                             roc_curve)

from src.model import forward
from src.federated_trainer import load_voice_test, load_hw_test


RES = Path("results")
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def load_model():
    return dict(np.load(RES / "global_model.npz"))


# ------------------------------------------------------------------ #
# Plots (per modality)
# ------------------------------------------------------------------ #

def plot_training_curves():
    h = pd.read_csv(RES / "round_history.csv")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Voice row
    axes[0, 0].plot(h["round"], h["voice_loss"], "o-", color="#d62728")
    axes[0, 0].set_title("Voice — BCE loss"); axes[0, 0].grid(alpha=0.3)
    axes[0, 1].plot(h["round"], h["voice_auc"], "o-", color="#1f77b4")
    axes[0, 1].set_title("Voice — ROC AUC"); axes[0, 1].set_ylim(0.5, 1.0)
    axes[0, 1].grid(alpha=0.3)
    axes[0, 2].plot(h["round"], h["voice_balanced_acc"], "o-",
                    color="#2ca02c", label="Balanced acc")
    axes[0, 2].plot(h["round"], h["voice_acc"], "s--", color="#888",
                    alpha=0.7, label="Accuracy")
    axes[0, 2].set_title("Voice — accuracy"); axes[0, 2].grid(alpha=0.3)
    axes[0, 2].legend(); axes[0, 2].set_ylim(0.4, 1.0)

    # Handwriting row
    axes[1, 0].plot(h["round"], h["hw_loss"], "o-", color="#d62728")
    axes[1, 0].set_title("Handwriting — BCE loss"); axes[1, 0].grid(alpha=0.3)
    axes[1, 1].plot(h["round"], h["hw_auc"], "o-", color="#1f77b4")
    axes[1, 1].set_title("Handwriting — ROC AUC"); axes[1, 1].set_ylim(0.5, 1.0)
    axes[1, 1].grid(alpha=0.3)
    axes[1, 2].plot(h["round"], h["hw_balanced_acc"], "o-",
                    color="#2ca02c", label="Balanced acc")
    axes[1, 2].plot(h["round"], h["hw_acc"], "s--", color="#888",
                    alpha=0.7, label="Accuracy")
    axes[1, 2].set_title("Handwriting — accuracy"); axes[1, 2].grid(alpha=0.3)
    axes[1, 2].legend(); axes[1, 2].set_ylim(0.4, 1.05)

    for ax in axes.flat:
        ax.set_xlabel("FL round")

    fig.suptitle("Multi-Modal Federated Training History — 4 clients, 15 rounds",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG / "training_curves.png", dpi=140, bbox_inches="tight")
    plt.close()


def plot_confusion(modality):
    preds = pd.read_csv(RES / f"predictions_{modality}_test.csv")
    y, p = preds["y_true"].values, preds["y_pred"].values
    cm = confusion_matrix(y, p)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black",
                    fontsize=16, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Healthy", "PD"]); ax.set_yticklabels(["Healthy", "PD"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {modality}")
    plt.colorbar(im, ax=ax); plt.tight_layout()
    plt.savefig(FIG / f"confusion_{modality}.png", dpi=140, bbox_inches="tight")
    plt.close()


def plot_roc_pr():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for modality, color in (("voice", "#1f77b4"), ("hw", "#d62728")):
        preds = pd.read_csv(RES / f"predictions_{modality}_test.csv")
        y, p = preds["y_true"].values, preds["y_prob"].values
        fpr, tpr, _ = roc_curve(y, p); a = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, "-", lw=2, color=color,
                     label=f"{modality} (AUC={a:.3f})")
        prec, rec, _ = precision_recall_curve(y, p); ap = auc(rec, prec)
        axes[1].plot(rec, prec, "-", lw=2, color=color,
                     label=f"{modality} (AP={ap:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC — both modalities"); axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall — both modalities"); axes[1].legend(loc="lower left")
    axes[1].grid(alpha=0.3)

    plt.suptitle("Held-Out Test Set Performance", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG / "roc_pr_combined.png", dpi=140, bbox_inches="tight")
    plt.close()


def plot_calibration():
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
    for modality, color in (("voice", "#1f77b4"), ("hw", "#d62728")):
        preds = pd.read_csv(RES / f"predictions_{modality}_test.csv")
        y, p = preds["y_true"].values, preds["y_prob"].values
        # Few-point datasets need few bins
        n_bins = min(8, max(3, len(y) // 5))
        try:
            frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins,
                                                   strategy="quantile")
            brier = brier_score_loss(y, p)
            ax.plot(mean_pred, frac_pos, "o-", color=color,
                    label=f"{modality} (Brier={brier:.3f})")
        except ValueError:
            continue
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical positive fraction")
    ax.set_title("Calibration (Reliability) Curve")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "calibration.png", dpi=140, bbox_inches="tight")
    plt.close()


def plot_prob_histograms():
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, modality in zip(axes, ("voice", "hw")):
        preds = pd.read_csv(RES / f"predictions_{modality}_test.csv")
        p_pd = preds.loc[preds["y_true"] == 1, "y_prob"].values
        p_he = preds.loc[preds["y_true"] == 0, "y_prob"].values
        bins = np.linspace(0, 1, 20)
        ax.hist(p_he, bins=bins, alpha=0.6, label=f"Healthy (n={len(p_he)})",
                color="#2ca02c")
        ax.hist(p_pd, bins=bins, alpha=0.6, label=f"PD (n={len(p_pd)})",
                color="#d62728")
        ax.axvline(0.5, color="k", ls="--", alpha=0.5)
        ax.set_xlabel("Predicted P(PD)"); ax.set_ylabel("Count")
        ax.set_title(f"{modality}")
        ax.legend(); ax.grid(alpha=0.3)
    plt.suptitle("Predicted-Probability Distribution by True Class",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG / "prob_histograms.png", dpi=140, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------------ #
# Permutation feature importance per modality
# ------------------------------------------------------------------ #

def permutation_importance(p, X, y, modality, n_repeats=30, seed=0):
    rng = np.random.default_rng(seed)
    base_probs, _ = forward(p, X, modality, dropout_p=0.0)
    base_auc = roc_auc_score(y, base_probs)
    importances = np.zeros((X.shape[1], n_repeats))
    for j in range(X.shape[1]):
        for r in range(n_repeats):
            X_shuf = X.copy(); rng.shuffle(X_shuf[:, j])
            probs, _ = forward(p, X_shuf, modality, dropout_p=0.0)
            try:
                a = roc_auc_score(y, probs)
            except ValueError:
                a = base_auc
            importances[j, r] = base_auc - a
    return base_auc, importances


def plot_feature_importance(p, X, y, names, modality, k=15):
    base, imp = permutation_importance(p, X, y, modality)
    means, stds = imp.mean(axis=1), imp.std(axis=1)
    order = np.argsort(means)[::-1]
    top = order[:k]
    pd.DataFrame({"feature": [names[i] for i in order],
                  "mean_importance": means[order],
                  "std": stds[order]}).to_csv(
        RES / f"feature_importance_{modality}.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, max(5, 0.4*k)))
    yy = np.arange(len(top))
    ax.barh(yy, means[top], xerr=stds[top], color="#1f77b4", alpha=0.85)
    ax.set_yticks(yy); ax.set_yticklabels([names[i] for i in top])
    ax.invert_yaxis()
    ax.set_xlabel("Δ AUC when feature is permuted")
    ax.set_title(f"Top-{k} feature importance — {modality}  "
                 f"(baseline AUC={base:.3f})")
    ax.grid(alpha=0.3, axis="x"); plt.tight_layout()
    plt.savefig(FIG / f"feature_importance_{modality}.png",
                dpi=140, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    p = load_model()

    plot_training_curves()
    plot_confusion("voice"); plot_confusion("hw")
    plot_roc_pr()
    plot_calibration()
    plot_prob_histograms()

    Xv, yv, vfeat = load_voice_test()
    Xh, yh, hfeat = load_hw_test()
    plot_feature_importance(p, Xv, yv, vfeat, "voice", k=15)
    plot_feature_importance(p, Xh, yh, hfeat, "handwriting", k=15)

    print("Wrote figures to", FIG)
    for f in sorted(FIG.glob("*.png")):
        print(" -", f.name)


if __name__ == "__main__":
    main()
