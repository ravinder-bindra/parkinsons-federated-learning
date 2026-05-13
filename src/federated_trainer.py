"""
src/federated_trainer.py
------------------------
Multi-modal federated training driver.

Clients
-------
   Hospital 1, 2, 3  : voice modality        (UCI hospital_{1,2,3}.csv)
   Hospital 4        : handwriting modality  (NewHandPD aggregated)

Aggregation rule (modality-aware FedProx)
-----------------------------------------
   * voice encoder keys: averaged over H1, H2, H3 (weighted by samples)
   * hw encoder keys   : taken from H4 directly (only owner)
   * classifier head   : averaged over ALL clients
                         (this is the shared knowledge that crosses
                         modalities -- both kinds of clients are
                         training the same downstream task)

Outputs (in results/)
---------------------
   global_model.npz               final aggregated parameters
   round_history.csv              per-round metrics (voice & hw test sets)
   predictions_voice_test.csv     per-patient predictions on voice test
   predictions_hw_test.csv        per-patient predictions on handwriting test
   run_config.json                reproducibility config

Run
---
   python -m src.federated_trainer --rounds 15
   python -m src.federated_trainer --rounds 15 --dp        # with DP-SGD
"""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, balanced_accuracy_score,
                             accuracy_score, f1_score)

from src.model import (init_params, forward, bce_loss, backward_bce, Adam,
                       VOICE_KEYS, HW_KEYS, HEAD_KEYS)


PROCESSED = Path("data/processed")
RESULTS = Path("results")


# -------------------------------------------------------------------- #
# Client data loaders
# -------------------------------------------------------------------- #

def load_voice_client(hid: int) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(PROCESSED / f"voice_h{hid}.csv")
    X = df.drop("status", axis=1).values.astype(np.float64)
    y = df["status"].values.astype(np.float64)
    return X, y


def load_handwriting_client() -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(PROCESSED / "handwriting_train.csv")
    feat = [c for c in df.columns if c not in ("patient_id", "status")]
    X = df[feat].values.astype(np.float64)
    y = df["status"].values.astype(np.float64)
    return X, y


def load_voice_test():
    df = pd.read_csv(PROCESSED / "voice_test.csv")
    X = df.drop("status", axis=1).values.astype(np.float64)
    y = df["status"].values.astype(int)
    return X, y, [c for c in df.columns if c != "status"]


def load_hw_test():
    df = pd.read_csv(PROCESSED / "handwriting_test.csv")
    feat = [c for c in df.columns if c not in ("patient_id", "status")]
    X = df[feat].values.astype(np.float64)
    y = df["status"].values.astype(int)
    return X, y, feat


# -------------------------------------------------------------------- #
# DP-SGD primitives (per-sample clip + Gaussian noise)
# -------------------------------------------------------------------- #

def per_sample_clipped_grads(p, X, y, modality, clip_norm):
    grads_list = []
    for i in range(X.shape[0]):
        probs, cache = forward(p, X[i:i+1], modality)
        g_i = backward_bce(p, cache, y[i:i+1], probs)
        flat = np.concatenate([v.flatten() for v in g_i.values()])
        norm = np.linalg.norm(flat) + 1e-12
        scale = min(1.0, clip_norm / norm)
        grads_list.append({k: v * scale for k, v in g_i.items()})
    return grads_list


def dp_aggregate(grads_list, clip_norm, noise_mult, rng):
    n = len(grads_list)
    summed = {k: sum(g[k] for g in grads_list) for k in grads_list[0]}
    return {k: (s + rng.normal(0, noise_mult * clip_norm, s.shape)) / n
            for k, s in summed.items()}


# -------------------------------------------------------------------- #
# One client's local update
# -------------------------------------------------------------------- #

def local_train(p_global, X, y, modality, *, local_epochs, batch_size, lr,
                prox_mu, dropout_p, dp_enabled, dp_clip, dp_noise, rng):
    """Returns updated copy of the global parameters (for this hospital)."""
    p = {k: v.copy() for k, v in p_global.items()}
    opt = Adam(p, lr=lr)

    owned = set(p.keys())  # all params trainable; aggregation handles ownership
    n = X.shape[0]
    for _ in range(local_epochs):
        idx = rng.permutation(n)
        for s in range(0, n, batch_size):
            sel = idx[s:s + batch_size]
            xb, yb = X[sel], y[sel]
            if dp_enabled:
                gps = per_sample_clipped_grads(p, xb, yb, modality, dp_clip)
                grads = dp_aggregate(gps, dp_clip, dp_noise, rng)
                if prox_mu > 0:
                    for k in grads:
                        grads[k] = grads[k] + prox_mu * (p[k] - p_global[k])
            else:
                probs, cache = forward(p, xb, modality, dropout_p, rng)
                grads = backward_bce(p, cache, yb, probs,
                                     prox_mu=prox_mu, p_global=p_global)
            opt.step(p, grads)
    return p


# -------------------------------------------------------------------- #
# Modality-aware FedProx aggregation
# -------------------------------------------------------------------- #

def aggregate(p_global, voice_updates, voice_weights,
              hw_update, hw_weight):
    """
    voice_updates: list of param-dicts from voice clients
    voice_weights: list of sample counts (one per voice client)
    hw_update:     param-dict from the handwriting client
    hw_weight:     sample count for handwriting client
    """
    p_new = {k: v.copy() for k, v in p_global.items()}
    voice_total = float(sum(voice_weights))
    head_total  = voice_total + float(hw_weight)

    # voice encoder = weighted avg over voice clients only
    for k in VOICE_KEYS:
        p_new[k] = sum(w * cp[k] for w, cp in zip(voice_weights,
                                                   voice_updates)) / voice_total

    # handwriting encoder = direct copy from the single owner
    for k in HW_KEYS:
        p_new[k] = hw_update[k].copy()

    # classifier head = weighted avg across ALL clients
    for k in HEAD_KEYS:
        head_sum = sum(w * cp[k] for w, cp in zip(voice_weights, voice_updates))
        head_sum = head_sum + hw_weight * hw_update[k]
        p_new[k] = head_sum / head_total

    return p_new


# -------------------------------------------------------------------- #
# Evaluation
# -------------------------------------------------------------------- #

def evaluate(p, X, y, modality):
    probs, _ = forward(p, X, modality, dropout_p=0.0)
    preds = (probs > 0.5).astype(int)
    metrics = {
        "loss":         bce_loss(probs, y.astype(float)),
        "accuracy":     accuracy_score(y, preds),
        "balanced_acc": balanced_accuracy_score(y, preds),
        "f1":           f1_score(y, preds, zero_division=0),
    }
    try:
        metrics["auc"] = roc_auc_score(y, probs)
    except ValueError:
        metrics["auc"] = float("nan")
    return metrics, probs, preds


# -------------------------------------------------------------------- #
# Main FL run
# -------------------------------------------------------------------- #

def federated_run(*, num_rounds=15, local_epochs=3, batch_size=16, lr=1e-3,
                  prox_mu=0.1, dropout_p=0.3, dp_enabled=False,
                  dp_clip=1.0, dp_noise=0.5, seed=42,
                  results_dir: str = "results"):
    results_dir = Path(results_dir); results_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    p_global = init_params(rng, head_out=1)

    voice_data = {hid: load_voice_client(hid) for hid in (1, 2, 3)}
    hw_data = load_handwriting_client()
    X_v_test, y_v_test, voice_features = load_voice_test()
    X_h_test, y_h_test, hw_features   = load_hw_test()

    history = []
    print(f"\n=== Multi-modal Federated Run: rounds={num_rounds}, "
          f"local_epochs={local_epochs}, prox_mu={prox_mu}, "
          f"DP={'ON' if dp_enabled else 'OFF'} ===\n")

    for rnd in range(1, num_rounds + 1):
        # ---- Voice clients local training ----
        voice_updates, voice_weights = [], []
        for hid, (X, y) in voice_data.items():
            up = local_train(p_global, X, y, "voice",
                             local_epochs=local_epochs, batch_size=batch_size,
                             lr=lr, prox_mu=prox_mu, dropout_p=dropout_p,
                             dp_enabled=dp_enabled, dp_clip=dp_clip,
                             dp_noise=dp_noise, rng=rng)
            voice_updates.append(up)
            voice_weights.append(X.shape[0])

        # ---- Handwriting client local training ----
        Xh, yh = hw_data
        hw_update = local_train(p_global, Xh, yh, "handwriting",
                                local_epochs=local_epochs, batch_size=batch_size,
                                lr=lr, prox_mu=prox_mu, dropout_p=dropout_p,
                                dp_enabled=dp_enabled, dp_clip=dp_clip,
                                dp_noise=dp_noise, rng=rng)
        hw_weight = Xh.shape[0]

        # ---- Aggregate ----
        p_global = aggregate(p_global, voice_updates, voice_weights,
                             hw_update, hw_weight)

        # ---- Evaluate on both modality test sets ----
        m_v, _, _ = evaluate(p_global, X_v_test, y_v_test, "voice")
        m_h, _, _ = evaluate(p_global, X_h_test, y_h_test, "handwriting")
        history.append({"round": rnd,
                        "voice_loss": m_v["loss"], "voice_auc": m_v["auc"],
                        "voice_acc":  m_v["accuracy"],
                        "voice_balanced_acc": m_v["balanced_acc"],
                        "voice_f1":   m_v["f1"],
                        "hw_loss":    m_h["loss"], "hw_auc": m_h["auc"],
                        "hw_acc":     m_h["accuracy"],
                        "hw_balanced_acc": m_h["balanced_acc"],
                        "hw_f1":      m_h["f1"]})
        print(f"[r{rnd:>2}] voice: AUC={m_v['auc']:.3f} acc={m_v['accuracy']:.3f} "
              f"bal_acc={m_v['balanced_acc']:.3f} | "
              f"handwriting: AUC={m_h['auc']:.3f} acc={m_h['accuracy']:.3f} "
              f"bal_acc={m_h['balanced_acc']:.3f}")

    # ---- Save artefacts ----
    np.savez(results_dir / "global_model.npz", **p_global)
    pd.DataFrame(history).to_csv(results_dir / "round_history.csv", index=False)

    m_v, v_probs, v_preds = evaluate(p_global, X_v_test, y_v_test, "voice")
    m_h, h_probs, h_preds = evaluate(p_global, X_h_test, y_h_test, "handwriting")
    pd.DataFrame({"y_true": y_v_test, "y_prob": v_probs,
                  "y_pred": v_preds}).to_csv(
        results_dir / "predictions_voice_test.csv", index=False)
    pd.DataFrame({"y_true": y_h_test, "y_prob": h_probs,
                  "y_pred": h_preds}).to_csv(
        results_dir / "predictions_hw_test.csv", index=False)

    cfg = dict(num_rounds=num_rounds, local_epochs=local_epochs,
               batch_size=batch_size, lr=lr, prox_mu=prox_mu,
               dropout_p=dropout_p, dp_enabled=dp_enabled,
               dp_clip=dp_clip, dp_noise=dp_noise, seed=seed,
               final_voice=m_v, final_handwriting=m_h)
    with open(results_dir / "run_config.json", "w") as f:
        json.dump(cfg, f, indent=2, default=float)

    print(f"\nFinal voice:        {m_v}")
    print(f"Final handwriting:  {m_h}")
    return {"history": history, "params": p_global,
            "voice_features": voice_features, "hw_features": hw_features}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds",       type=int,   default=15)
    p.add_argument("--local-epochs", type=int,   default=3)
    p.add_argument("--batch",        type=int,   default=16)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--prox-mu",      type=float, default=0.1)
    p.add_argument("--dropout",      type=float, default=0.3)
    p.add_argument("--dp",           action="store_true")
    p.add_argument("--dp-clip",      type=float, default=1.0)
    p.add_argument("--dp-noise",     type=float, default=0.5)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--results-dir",  default="results")
    args = p.parse_args()

    federated_run(num_rounds=args.rounds, local_epochs=args.local_epochs,
                  batch_size=args.batch, lr=args.lr, prox_mu=args.prox_mu,
                  dropout_p=args.dropout, dp_enabled=args.dp,
                  dp_clip=args.dp_clip, dp_noise=args.dp_noise,
                  seed=args.seed, results_dir=args.results_dir)


if __name__ == "__main__":
    main()
