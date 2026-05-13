"""
src/updrs_regression.py
-----------------------
A second model: predicts motor_UPDRS and total_UPDRS from voice
features in the Oxford Telemonitoring dataset (5,875 recordings from
42 PD patients, longitudinal).

This is a *regression* problem (continuous severity score), not the
binary classification of the main FL run. Useful clinically because
binary "PD vs healthy" undersells what voice biomarkers can do --
they actually track disease progression.

Setup
-----
   * Same encoder architecture as the federated voice encoder (so a
     transfer-learning extension is straightforward).
   * Local training only (single-cohort dataset, no federation needed).
   * Patient-aware split (no subject in both train and test) is already
     baked into the preprocessed CSVs.

Outputs (results/updrs/)
------------------------
   updrs_model.npz                   trained parameters
   predictions_test.csv              per-recording predictions
   metrics.json                      MAE, RMSE, R² for both targets
   curves.png                        train/test loss across epochs
"""

from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.model import init_params, forward, mse_loss, backward_mse, Adam


PROCESSED = Path("data/processed")
OUT = Path("results/updrs")
OUT.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------- #
# Patch the model for 2-output regression
# -------------------------------------------------------------------- #
# The model in src/model.py defaults to head_out=1 (binary classification).
# For regression we re-init with head_out=2 and use forward(... head_out=2)
# so the head's final layer has shape (16, 2) instead of (16, 1).

UPDRS_FEATURES = ["age", "sex",
                  "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5",
                  "Jitter:DDP", "Shimmer", "Shimmer(dB)", "Shimmer:APQ3",
                  "Shimmer:APQ5", "Shimmer:APQ11", "Shimmer:DDA",
                  "NHR", "HNR", "RPDE", "DFA", "PPE"]
TARGETS = ["motor_UPDRS", "total_UPDRS"]


def load_updrs():
    train = pd.read_csv(PROCESSED / "updrs_train.csv")
    test  = pd.read_csv(PROCESSED / "updrs_test.csv")
    Xtr = train[UPDRS_FEATURES].values.astype(np.float64)
    ytr = train[TARGETS].values.astype(np.float64)
    Xte = test[UPDRS_FEATURES].values.astype(np.float64)
    yte = test[TARGETS].values.astype(np.float64)
    return Xtr, ytr, Xte, yte


def _pad_features_to_22(X):
    """The voice encoder expects 22 features. Telemonitoring has 18.
    Pad with zeros so the same architecture works on both."""
    pad = np.zeros((X.shape[0], 22 - X.shape[1]))
    return np.concatenate([X, pad], axis=1)


def predict(p, X, head_out=2):
    Xp = _pad_features_to_22(X)
    out, _ = forward(p, Xp, "voice", dropout_p=0.0, head_out=head_out)
    return out  # shape (N, 2)


def evaluate(p, X, y):
    y_pred = predict(p, X)
    metrics = {}
    for i, name in enumerate(TARGETS):
        metrics[name] = {
            "MAE":  float(mean_absolute_error(y[:, i], y_pred[:, i])),
            "RMSE": float(np.sqrt(mean_squared_error(y[:, i], y_pred[:, i]))),
            "R2":   float(r2_score(y[:, i], y_pred[:, i])),
        }
    metrics["mean_RMSE"] = float(np.mean([m["RMSE"] for m in metrics.values()
                                          if isinstance(m, dict)]))
    return metrics, y_pred


def train(epochs: int = 30, batch_size: int = 64, lr: float = 1e-3,
          seed: int = 42):
    rng = np.random.default_rng(seed)
    p = init_params(rng, head_out=2)
    opt = Adam(p, lr=lr)

    Xtr, ytr, Xte, yte = load_updrs()
    Xtr_p = _pad_features_to_22(Xtr)
    print(f"Training on {Xtr.shape[0]} recordings, testing on {Xte.shape[0]}")

    history = {"epoch": [], "train_loss": [], "test_loss": [],
               "motor_MAE": [], "total_MAE": []}

    for epoch in range(1, epochs + 1):
        idx = rng.permutation(Xtr_p.shape[0])
        epoch_loss = 0.0; nb = 0
        for s in range(0, Xtr_p.shape[0], batch_size):
            sel = idx[s:s + batch_size]
            xb, yb = Xtr_p[sel], ytr[sel]
            pred, cache = forward(p, xb, "voice", dropout_p=0.2, rng=rng,
                                   head_out=2)
            loss = mse_loss(pred, yb)
            grads = backward_mse(p, cache, yb, pred)
            opt.step(p, grads)
            epoch_loss += loss * xb.shape[0]; nb += xb.shape[0]
        train_loss = epoch_loss / nb

        test_pred = predict(p, Xte)
        test_loss = mse_loss(test_pred, yte)
        m = evaluate(p, Xte, yte)[0]
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["motor_MAE"].append(m["motor_UPDRS"]["MAE"])
        history["total_MAE"].append(m["total_UPDRS"]["MAE"])

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"  epoch {epoch:>2}: train_loss={train_loss:.3f}  "
                  f"test_loss={test_loss:.3f}  "
                  f"motor MAE={m['motor_UPDRS']['MAE']:.2f}  "
                  f"total MAE={m['total_UPDRS']['MAE']:.2f}")

    # Final eval and save
    final_metrics, y_pred = evaluate(p, Xte, yte)
    np.savez(OUT / "updrs_model.npz", **p)
    pd.DataFrame({
        "y_motor_true": yte[:, 0], "y_motor_pred": y_pred[:, 0],
        "y_total_true": yte[:, 1], "y_total_pred": y_pred[:, 1],
    }).to_csv(OUT / "predictions_test.csv", index=False)

    with open(OUT / "metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    # Plots: training curves + scatter
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["epoch"], history["train_loss"], "-",
                 label="train", color="#1f77b4")
    axes[0].plot(history["epoch"], history["test_loss"], "-",
                 label="test", color="#d62728")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE")
    axes[0].set_title("UPDRS regression: training curves")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].scatter(yte[:, 0], y_pred[:, 0], s=8, alpha=0.4,
                    color="#1f77b4")
    lo, hi = yte[:, 0].min(), yte[:, 0].max()
    axes[1].plot([lo, hi], [lo, hi], "k--", alpha=0.5)
    axes[1].set_xlabel("True motor_UPDRS"); axes[1].set_ylabel("Predicted")
    axes[1].set_title(f"motor_UPDRS  (R² = "
                      f"{final_metrics['motor_UPDRS']['R2']:.3f})")
    axes[1].grid(alpha=0.3)

    axes[2].scatter(yte[:, 1], y_pred[:, 1], s=8, alpha=0.4,
                    color="#d62728")
    lo, hi = yte[:, 1].min(), yte[:, 1].max()
    axes[2].plot([lo, hi], [lo, hi], "k--", alpha=0.5)
    axes[2].set_xlabel("True total_UPDRS"); axes[2].set_ylabel("Predicted")
    axes[2].set_title(f"total_UPDRS  (R² = "
                      f"{final_metrics['total_UPDRS']['R2']:.3f})")
    axes[2].grid(alpha=0.3)

    plt.suptitle("Telemonitoring UPDRS Regression (42 PD patients, "
                 "patient-aware split)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "curves.png", dpi=140, bbox_inches="tight")
    plt.close()

    print(f"\n== Final metrics ==")
    print(json.dumps(final_metrics, indent=2))
    return final_metrics


if __name__ == "__main__":
    train()
