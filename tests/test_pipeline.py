"""
tests/test_pipeline.py
----------------------
End-to-end tests for the multi-modal federated pipeline.

Run:
    python tests/test_pipeline.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import (init_params, forward, bce_loss, backward_bce,
                       VOICE_KEYS, HW_KEYS, HEAD_KEYS,
                       VOICE_IN, HW_IN, EMBED_DIM)
from src.federated_trainer import (aggregate, federated_run,
                                   load_voice_client, load_handwriting_client,
                                   load_voice_test, load_hw_test)


PROC = Path(__file__).resolve().parents[1] / "data" / "processed"


# ----------------------------------------------------------------- #
# Data
# ----------------------------------------------------------------- #

def test_processed_files_exist():
    for f in ("voice_h1.csv", "voice_h2.csv", "voice_h3.csv",
              "voice_test.csv", "handwriting_train.csv",
              "handwriting_test.csv", "manifest.json"):
        assert (PROC / f).exists(), f"missing {PROC / f}"


def test_voice_shapes():
    for hid in (1, 2, 3):
        X, y = load_voice_client(hid)
        assert X.shape[1] == VOICE_IN
        assert X.shape[0] == y.shape[0]
    Xt, yt, names = load_voice_test()
    assert Xt.shape[1] == VOICE_IN
    assert len(names) == VOICE_IN


def test_hw_shapes():
    X, y = load_handwriting_client()
    Xt, yt, names = load_hw_test()
    assert X.shape[1] == HW_IN
    assert Xt.shape[1] == HW_IN
    assert len(names) == HW_IN


# ----------------------------------------------------------------- #
# Model
# ----------------------------------------------------------------- #

def test_init_param_shapes():
    rng = np.random.default_rng(0)
    p = init_params(rng)
    assert p["voice_W1"].shape == (VOICE_IN, 64)
    assert p["voice_W2"].shape == (64, EMBED_DIM)
    assert p["hw_W1"].shape == (HW_IN, 64)
    assert p["hw_W2"].shape == (64, EMBED_DIM)
    assert p["head_W1"].shape == (EMBED_DIM, 16)
    assert p["head_W2"].shape == (16, 1)


def test_forward_voice():
    rng = np.random.default_rng(0)
    p = init_params(rng)
    X = rng.standard_normal((5, VOICE_IN))
    out, _ = forward(p, X, "voice")
    assert out.shape == (5,)
    assert (out >= 0).all() and (out <= 1).all()


def test_forward_hw():
    rng = np.random.default_rng(0)
    p = init_params(rng)
    X = rng.standard_normal((5, HW_IN))
    out, _ = forward(p, X, "handwriting")
    assert out.shape == (5,)
    assert (out >= 0).all() and (out <= 1).all()


def test_backward_finite_difference_voice():
    rng = np.random.default_rng(42)
    p = init_params(rng)
    X = rng.standard_normal((4, VOICE_IN))
    y = np.array([1.0, 0.0, 1.0, 0.0])
    probs, cache = forward(p, X, "voice")
    grads = backward_bce(p, cache, y, probs)
    # Check the head's last layer (smallest, fastest)
    W = p["head_W2"]; eps = 1e-5
    num_grad = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            pp = {k: v.copy() for k, v in p.items()}
            pm = {k: v.copy() for k, v in p.items()}
            pp["head_W2"][i, j] += eps; pm["head_W2"][i, j] -= eps
            l_p = bce_loss(forward(pp, X, "voice")[0], y)
            l_m = bce_loss(forward(pm, X, "voice")[0], y)
            num_grad[i, j] = (l_p - l_m) / (2 * eps)
    rel_err = (np.linalg.norm(grads["head_W2"] - num_grad)
               / (np.linalg.norm(grads["head_W2"]) +
                  np.linalg.norm(num_grad) + 1e-12))
    assert rel_err < 1e-3, f"voice grad check failed rel_err={rel_err}"


def test_backward_finite_difference_hw():
    rng = np.random.default_rng(42)
    p = init_params(rng)
    X = rng.standard_normal((4, HW_IN))
    y = np.array([1.0, 0.0, 1.0, 0.0])
    probs, cache = forward(p, X, "handwriting")
    grads = backward_bce(p, cache, y, probs)
    W = p["hw_W2"]; eps = 1e-5
    num_grad = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            pp = {k: v.copy() for k, v in p.items()}
            pm = {k: v.copy() for k, v in p.items()}
            pp["hw_W2"][i, j] += eps; pm["hw_W2"][i, j] -= eps
            l_p = bce_loss(forward(pp, X, "handwriting")[0], y)
            l_m = bce_loss(forward(pm, X, "handwriting")[0], y)
            num_grad[i, j] = (l_p - l_m) / (2 * eps)
    rel_err = (np.linalg.norm(grads["hw_W2"] - num_grad)
               / (np.linalg.norm(grads["hw_W2"]) +
                  np.linalg.norm(num_grad) + 1e-12))
    assert rel_err < 1e-3, f"hw grad check failed rel_err={rel_err}"


# ----------------------------------------------------------------- #
# Modality-aware aggregation
# ----------------------------------------------------------------- #

def test_aggregate_voice_only_averages_voice_keys():
    rng = np.random.default_rng(0)
    pg = init_params(rng)
    vu1 = {k: pg[k] * 1.0 for k in pg}
    vu2 = {k: pg[k] * 3.0 for k in pg}
    vu3 = {k: pg[k] * 5.0 for k in pg}
    hu  = {k: pg[k] * 7.0 for k in pg}
    out = aggregate(pg, [vu1, vu2, vu3], [10, 10, 10], hu, 10)
    # Voice keys should be the simple average of 1, 3, 5 = 3
    expected_voice_factor = (10*1 + 10*3 + 10*5) / 30.0
    assert np.allclose(out["voice_W1"], pg["voice_W1"] * expected_voice_factor)
    # HW keys should be exactly the HW update
    assert np.allclose(out["hw_W1"], pg["hw_W1"] * 7.0)
    # Head keys = (10*1 + 10*3 + 10*5 + 10*7) / 40 = 4
    expected_head_factor = (10*1 + 10*3 + 10*5 + 10*7) / 40.0
    assert np.allclose(out["head_W1"], pg["head_W1"] * expected_head_factor)


def test_keys_sets_disjoint_modality_specific():
    assert set(VOICE_KEYS).isdisjoint(set(HW_KEYS))
    assert set(VOICE_KEYS).isdisjoint(set(HEAD_KEYS))
    assert set(HW_KEYS).isdisjoint(set(HEAD_KEYS))


# ----------------------------------------------------------------- #
# Integration: smoke training
# ----------------------------------------------------------------- #

def test_federated_run_reduces_loss():
    """3 rounds of multi-modal training should clearly reduce loss."""
    out = federated_run(num_rounds=3, local_epochs=2, batch_size=16,
                        seed=0, results_dir="results_test")
    h = out["history"]
    assert h[-1]["voice_loss"] < 0.65, "voice loss not learning"
    assert h[-1]["hw_loss"]    < 0.65, "hw loss not learning"


def test_deterministic_seed():
    o1 = federated_run(num_rounds=2, local_epochs=1, seed=123,
                       results_dir="results_d1")
    o2 = federated_run(num_rounds=2, local_epochs=1, seed=123,
                       results_dir="results_d2")
    for k in o1["params"]:
        assert np.allclose(o1["params"][k], o2["params"][k]), \
            f"mismatch on {k}"


# ----------------------------------------------------------------- #
# Runner
# ----------------------------------------------------------------- #

if __name__ == "__main__":
    tests = [
        test_processed_files_exist,
        test_voice_shapes,
        test_hw_shapes,
        test_init_param_shapes,
        test_forward_voice,
        test_forward_hw,
        test_backward_finite_difference_voice,
        test_backward_finite_difference_hw,
        test_aggregate_voice_only_averages_voice_keys,
        test_keys_sets_disjoint_modality_specific,
        test_federated_run_reduces_loss,
        test_deterministic_seed,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  OK   {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERR  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    # Cleanup test artifacts
    import shutil
    for d in ("results_test", "results_d1", "results_d2"):
        shutil.rmtree(d, ignore_errors=True)
    sys.exit(1 if failed else 0)
