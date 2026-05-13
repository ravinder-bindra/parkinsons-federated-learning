"""
src/preprocess.py
-----------------
Preprocesses the three REAL datasets uploaded to data/raw/ into
clean, per-client CSVs in data/processed/:

  Voice (binary classification):
    - data/raw/hospital_{1,2,3}.csv      UCI Parkinson's (existing splits)
    - data/raw/global_test.csv           held-out test
    -> data/processed/voice_h{1,2,3}.csv, voice_test.csv

  Handwriting (binary classification):
    - data/raw/NewSpiral.csv             NewHandPD spiral kinematics
    - data/raw/NewMeander.csv            NewHandPD meander kinematics
    Aggregated per patient (mean over 4 exams), combined to 18 features,
    split 80/20 by patient_id (no patient leakage).
    -> data/processed/handwriting_train.csv, handwriting_test.csv

  UPDRS regression:
    - data/raw/parkinsons_updrs.data     Oxford Telemonitoring
    Patient-aware split (no subject in both train and test).
    -> data/processed/updrs_train.csv, updrs_test.csv

This script is deterministic (seed=42) and produces a manifest.json
summarising row counts, class balance, and feature lists.
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)


# ====================================================================
# 1) Voice (already split, just copy and standardise the label dtype)
# ====================================================================

def preprocess_voice() -> dict:
    info = {"clients": {}, "feature_count": None}
    for hid in (1, 2, 3):
        df = pd.read_csv(RAW / f"hospital_{hid}.csv")
        df["status"] = df["status"].astype(int)
        out_path = OUT / f"voice_h{hid}.csv"
        df.to_csv(out_path, index=False)
        info["clients"][f"h{hid}"] = {
            "rows": len(df),
            "PD": int((df["status"] == 1).sum()),
            "healthy": int((df["status"] == 0).sum()),
            "path": str(out_path),
        }
    test = pd.read_csv(RAW / "global_test.csv")
    test["status"] = test["status"].astype(int)
    test.to_csv(OUT / "voice_test.csv", index=False)
    info["test"] = {
        "rows": len(test),
        "PD": int((test["status"] == 1).sum()),
        "healthy": int((test["status"] == 0).sum()),
        "path": str(OUT / "voice_test.csv"),
    }
    info["feature_count"] = test.shape[1] - 1
    info["features"] = [c for c in test.columns if c != "status"]
    return info


# ====================================================================
# 2) Handwriting — patient-level aggregation, no patient leakage
# ====================================================================

HW_FEATURES = [
    "RMS", "MAX_BETWEEN_ET_HT", "MIN_BETWEEN_ET_HT", "STD_DEVIATION_ET_HT",
    "MRT", "MAX_HT", "MIN_HT", "STD_HT",
    "CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_ET_HT",
]


def preprocess_handwriting(seed: int = 42) -> dict:
    spiral = pd.read_csv(RAW / "NewSpiral.csv")
    meander = pd.read_csv(RAW / "NewMeander.csv")

    # Average all exams per patient (each patient has ~4 exams per task)
    sp_agg = spiral.groupby("ID_PATIENT")[HW_FEATURES].mean()
    me_agg = meander.groupby("ID_PATIENT")[HW_FEATURES].mean()
    sp_agg.columns = [f"spiral_{c}" for c in sp_agg.columns]
    me_agg.columns = [f"meander_{c}" for c in me_agg.columns]

    # Class label (constant per patient)
    cls = spiral.groupby("ID_PATIENT")["CLASS_TYPE"].first()
    # NewHandPD: CLASS_TYPE 1 = healthy control, 2 = Parkinson's patient
    status = (cls == 2).astype(int)
    status.name = "status"

    # Demographics (constant per patient)
    demo = spiral.groupby("ID_PATIENT")[["AGE", "GENDER", "RIGH/LEFT-HANDED"]].first()
    # Encode demographics numerically (M=1, F=0; R=1, L=0)
    demo["GENDER"] = (demo["GENDER"].astype(str).str.strip().str.upper() == "M").astype(int)
    demo = demo.rename(columns={"RIGH/LEFT-HANDED": "RIGHT_HANDED"})
    demo["RIGHT_HANDED"] = (demo["RIGHT_HANDED"].astype(str).str.strip().str.upper() == "R").astype(int)

    full = pd.concat([sp_agg, me_agg, demo, status], axis=1).reset_index()
    full = full.rename(columns={"ID_PATIENT": "patient_id"})

    # Standardise features (z-score) using training-set stats only
    feature_cols = [c for c in full.columns
                    if c not in ("patient_id", "status")]

    # Patient-level split: stratified by status
    train_df, test_df = train_test_split(
        full, test_size=0.20, random_state=seed,
        stratify=full["status"])

    # Fit scaler on train only (NaN-safe), apply to test
    means = train_df[feature_cols].mean()
    stds = train_df[feature_cols].std().replace(0, 1.0)
    train_df[feature_cols] = (train_df[feature_cols] - means) / stds
    test_df[feature_cols] = (test_df[feature_cols] - means) / stds

    train_df.to_csv(OUT / "handwriting_train.csv", index=False)
    test_df.to_csv(OUT / "handwriting_test.csv", index=False)

    # Save scaler stats for inference reuse
    pd.DataFrame({"feature": feature_cols, "mean": means.values,
                  "std": stds.values}).to_csv(
        OUT / "handwriting_scaler.csv", index=False)

    return {
        "train": {"rows": len(train_df),
                  "PD": int((train_df["status"] == 1).sum()),
                  "healthy": int((train_df["status"] == 0).sum())},
        "test":  {"rows": len(test_df),
                  "PD": int((test_df["status"] == 1).sum()),
                  "healthy": int((test_df["status"] == 0).sum())},
        "feature_count": len(feature_cols),
        "features": feature_cols,
    }


# ====================================================================
# 3) UPDRS regression on Telemonitoring (longitudinal, all 42 PD subjects)
# ====================================================================

# Acoustic features per recording (matches Oxford telemonitoring schema)
TELE_ACOUSTIC = [
    "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
    "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:APQ11",
    "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "PPE",
]
TELE_DEMO = ["age", "sex"]
TELE_TARGETS = ["motor_UPDRS", "total_UPDRS"]


def preprocess_updrs(seed: int = 42) -> dict:
    df = pd.read_csv(RAW / "parkinsons_updrs.data")
    subjects = df["subject#"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)

    # 80/20 split by SUBJECT (so a patient never appears in both sets)
    n_train = int(0.8 * len(subjects))
    train_subj = set(subjects[:n_train])
    test_subj  = set(subjects[n_train:])

    train_df = df[df["subject#"].isin(train_subj)].copy()
    test_df  = df[df["subject#"].isin(test_subj)].copy()

    # z-score acoustic + demographic features using train-only stats
    feature_cols = TELE_DEMO + TELE_ACOUSTIC
    means = train_df[feature_cols].mean()
    stds  = train_df[feature_cols].std().replace(0, 1.0)
    train_df[feature_cols] = (train_df[feature_cols] - means) / stds
    test_df[feature_cols]  = (test_df[feature_cols]  - means) / stds

    train_df.to_csv(OUT / "updrs_train.csv", index=False)
    test_df.to_csv(OUT / "updrs_test.csv", index=False)
    pd.DataFrame({"feature": feature_cols, "mean": means.values,
                  "std": stds.values}).to_csv(OUT / "updrs_scaler.csv", index=False)

    return {
        "train": {"recordings": len(train_df),
                  "subjects": len(train_subj)},
        "test":  {"recordings": len(test_df),
                  "subjects": len(test_subj)},
        "feature_count": len(feature_cols),
        "features": feature_cols,
        "targets": TELE_TARGETS,
    }


# ====================================================================
# Manifest
# ====================================================================

def main():
    print("=" * 60)
    print("Preprocessing REAL multi-modal Parkinson's data")
    print("=" * 60)

    voice = preprocess_voice()
    print(f"\nVoice (UCI binary):  test set {voice['test']['rows']} rows")
    for cid, v in voice["clients"].items():
        print(f"  hospital_{cid}: {v['rows']} rows  "
              f"(PD={v['PD']}, healthy={v['healthy']})")

    hw = preprocess_handwriting()
    print(f"\nHandwriting (NewHandPD):")
    print(f"  train: {hw['train']['rows']} patients  "
          f"(PD={hw['train']['PD']}, healthy={hw['train']['healthy']})")
    print(f"  test : {hw['test']['rows']} patients  "
          f"(PD={hw['test']['PD']}, healthy={hw['test']['healthy']})")
    print(f"  features: {hw['feature_count']} (spiral × 9 + meander × 9 + 3 demographics)")

    up = preprocess_updrs()
    print(f"\nUPDRS regression (Oxford Telemonitoring, ALL PD):")
    print(f"  train: {up['train']['recordings']} recordings "
          f"from {up['train']['subjects']} subjects")
    print(f"  test : {up['test']['recordings']} recordings "
          f"from {up['test']['subjects']} subjects")
    print(f"  features: {up['feature_count']}  targets: {up['targets']}")

    manifest = {"voice": voice, "handwriting": hw, "updrs": up}
    with open(OUT / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote manifest → {OUT/'manifest.json'}")


if __name__ == "__main__":
    main()
