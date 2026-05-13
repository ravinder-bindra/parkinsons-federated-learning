"""
src/model.py
------------
Multi-modal Parkinson's model implemented in pure numpy.

Architecture
------------
                                   ┌──────────────────────┐
   voice_features (22)  ───►   VoiceEncoder (22 → 32)
                                   │                      │
   handwriting (21)     ───►   HwEncoder    (21 → 32)
                                   └──────────┬───────────┘
                                              │
                                       shared embedding (32)
                                              │
                                      ClassifierHead (32 → 16 → 1)
                                              │
                                            sigmoid (PD risk)

For UPDRS regression we keep the same encoders but swap the head's
final layer to a 2-unit linear output (motor_UPDRS, total_UPDRS).

The federated training treats:
  * Voice clients (hospitals 1-3): train voice_encoder + classifier_head
  * Handwriting client (hospital 4): trains hw_encoder + classifier_head
  * The classifier head aggregates across ALL clients (it's the
    *shared knowledge* about PD).
  * Each encoder aggregates only across clients of its modality.

This is what lets two clients with different feature spaces collaborate
on the same task — the realistic federated setting where Hospital A
has voice equipment but no tablet, and Hospital B has the tablet but
no recording booth.
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional

import numpy as np


# -------------------------------------------------------------------- #
# Activations
# -------------------------------------------------------------------- #

def relu(x):  return np.maximum(0.0, x)
def drelu(x): return (x > 0.0).astype(x.dtype)


def sigmoid(x):
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


# -------------------------------------------------------------------- #
# Architectural constants
# -------------------------------------------------------------------- #

EMBED_DIM = 32                      # shared embedding dim across modalities
VOICE_IN = 22                       # UCI Parkinson's voice features
HW_IN    = 21                       # NewHandPD: 9*2 kinematic + 3 demographics


# -------------------------------------------------------------------- #
# Parameter scopes
# -------------------------------------------------------------------- #

VOICE_KEYS = ("voice_W1", "voice_b1", "voice_W2", "voice_b2")
HW_KEYS    = ("hw_W1",    "hw_b1",    "hw_W2",    "hw_b2")
HEAD_KEYS  = ("head_W1",  "head_b1",  "head_W2",  "head_b2")

ALL_KEYS = VOICE_KEYS + HW_KEYS + HEAD_KEYS


def init_params(rng: np.random.Generator,
                head_out: int = 1) -> Dict[str, np.ndarray]:
    """He init for ReLU layers, zero biases."""
    def he(fan_in, shape):
        return rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)

    return {
        # Voice encoder: 22 -> 64 -> 32
        "voice_W1": he(VOICE_IN, (VOICE_IN, 64)),
        "voice_b1": np.zeros(64),
        "voice_W2": he(64, (64, EMBED_DIM)),
        "voice_b2": np.zeros(EMBED_DIM),
        # Handwriting encoder: 21 -> 64 -> 32
        "hw_W1": he(HW_IN, (HW_IN, 64)),
        "hw_b1": np.zeros(64),
        "hw_W2": he(64, (64, EMBED_DIM)),
        "hw_b2": np.zeros(EMBED_DIM),
        # Classifier head: 32 -> 16 -> head_out
        "head_W1": he(EMBED_DIM, (EMBED_DIM, 16)),
        "head_b1": np.zeros(16),
        "head_W2": he(16, (16, head_out)),
        "head_b2": np.zeros(head_out),
    }


def keys_for_modality(modality: str) -> Tuple[str, ...]:
    """Returns the parameter keys a hospital with this modality owns."""
    if modality == "voice":
        return VOICE_KEYS + HEAD_KEYS
    if modality == "handwriting":
        return HW_KEYS + HEAD_KEYS
    raise ValueError(modality)


# -------------------------------------------------------------------- #
# Encoders
# -------------------------------------------------------------------- #

def encode_voice(p, X, dropout_p=0.0, rng=None):
    z1 = X @ p["voice_W1"] + p["voice_b1"]
    a1 = relu(z1)
    mask1 = _dropout_mask(a1, dropout_p, rng)
    if mask1 is not None: a1 = a1 * mask1
    z2 = a1 @ p["voice_W2"] + p["voice_b2"]
    emb = relu(z2)
    return emb, ("voice", X, z1, a1, mask1, z2, emb)


def encode_hw(p, X, dropout_p=0.0, rng=None):
    z1 = X @ p["hw_W1"] + p["hw_b1"]
    a1 = relu(z1)
    mask1 = _dropout_mask(a1, dropout_p, rng)
    if mask1 is not None: a1 = a1 * mask1
    z2 = a1 @ p["hw_W2"] + p["hw_b2"]
    emb = relu(z2)
    return emb, ("hw", X, z1, a1, mask1, z2, emb)


def _dropout_mask(a, p, rng):
    if p <= 0 or rng is None:
        return None
    return (rng.random(a.shape) > p).astype(a.dtype) / (1 - p)


# -------------------------------------------------------------------- #
# Classifier head
# -------------------------------------------------------------------- #

def head_forward(p, emb, dropout_p=0.0, rng=None,
                 head_out: int = 1) -> Tuple[np.ndarray, tuple]:
    z1 = emb @ p["head_W1"] + p["head_b1"]
    a1 = relu(z1)
    mask = _dropout_mask(a1, dropout_p, rng)
    if mask is not None: a1 = a1 * mask
    z2 = a1 @ p["head_W2"] + p["head_b2"]  # logits (or regression output)
    if head_out == 1:
        out = sigmoid(z2.squeeze(-1))
    else:
        out = z2  # raw values for regression
    return out, (emb, z1, a1, mask, z2)


# -------------------------------------------------------------------- #
# Full forward pass per modality
# -------------------------------------------------------------------- #

def forward(p, X, modality: str, dropout_p: float = 0.0,
            rng: Optional[np.random.Generator] = None,
            head_out: int = 1):
    if modality == "voice":
        emb, enc_cache = encode_voice(p, X, dropout_p, rng)
    elif modality == "handwriting":
        emb, enc_cache = encode_hw(p, X, dropout_p, rng)
    else:
        raise ValueError(modality)
    out, head_cache = head_forward(p, emb, dropout_p, rng, head_out)
    return out, (enc_cache, head_cache)


# -------------------------------------------------------------------- #
# Losses
# -------------------------------------------------------------------- #

def bce_loss(p_pred, y_true, eps: float = 1e-9) -> float:
    p_pred = np.clip(p_pred, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p_pred)
                          + (1 - y_true) * np.log(1 - p_pred)))


def mse_loss(y_pred, y_true) -> float:
    return float(np.mean((y_pred - y_true) ** 2))


# -------------------------------------------------------------------- #
# Backward (BCE classification path)
# -------------------------------------------------------------------- #

def backward_bce(p, cache, y_true, y_pred,
                 prox_mu: float = 0.0,
                 p_global: Optional[Dict[str, np.ndarray]] = None
                 ) -> Dict[str, np.ndarray]:
    enc_cache, head_cache = cache
    modality, X, z1, a1, mask1, z2, emb = enc_cache
    emb_h, h_z1, h_a1, h_mask, h_z2 = head_cache
    n = X.shape[0]

    # ---- Head backward ----
    dz_head_out = ((y_pred - y_true) / n).reshape(-1, 1)   # (B,1)
    dW_head2 = h_a1.T @ dz_head_out
    db_head2 = dz_head_out.sum(axis=0)

    da_h1 = dz_head_out @ p["head_W2"].T
    if h_mask is not None: da_h1 = da_h1 * h_mask
    dz_h1 = da_h1 * drelu(h_z1)
    dW_head1 = emb_h.T @ dz_h1
    db_head1 = dz_h1.sum(axis=0)
    d_emb = dz_h1 @ p["head_W1"].T

    # ---- Encoder backward ----
    dz_enc2 = d_emb * drelu(z2)
    if modality == "voice":
        dW_enc2 = a1.T @ dz_enc2
        db_enc2 = dz_enc2.sum(axis=0)
        da_enc1 = dz_enc2 @ p["voice_W2"].T
        if mask1 is not None: da_enc1 = da_enc1 * mask1
        dz_enc1 = da_enc1 * drelu(z1)
        dW_enc1 = X.T @ dz_enc1
        db_enc1 = dz_enc1.sum(axis=0)

        grads = {
            "voice_W1": dW_enc1, "voice_b1": db_enc1,
            "voice_W2": dW_enc2, "voice_b2": db_enc2,
            "head_W1": dW_head1, "head_b1": db_head1,
            "head_W2": dW_head2, "head_b2": db_head2,
        }
    else:  # handwriting
        dW_enc2 = a1.T @ dz_enc2
        db_enc2 = dz_enc2.sum(axis=0)
        da_enc1 = dz_enc2 @ p["hw_W2"].T
        if mask1 is not None: da_enc1 = da_enc1 * mask1
        dz_enc1 = da_enc1 * drelu(z1)
        dW_enc1 = X.T @ dz_enc1
        db_enc1 = dz_enc1.sum(axis=0)

        grads = {
            "hw_W1": dW_enc1, "hw_b1": db_enc1,
            "hw_W2": dW_enc2, "hw_b2": db_enc2,
            "head_W1": dW_head1, "head_b1": db_head1,
            "head_W2": dW_head2, "head_b2": db_head2,
        }

    # FedProx proximal term on the keys this hospital owns
    if prox_mu > 0 and p_global is not None:
        for k in grads:
            grads[k] = grads[k] + prox_mu * (p[k] - p_global[k])
    return grads


# -------------------------------------------------------------------- #
# Backward (MSE regression path) - same shape; just different dz_out
# -------------------------------------------------------------------- #

def backward_mse(p, cache, y_true, y_pred,
                 prox_mu: float = 0.0,
                 p_global: Optional[Dict[str, np.ndarray]] = None
                 ) -> Dict[str, np.ndarray]:
    enc_cache, head_cache = cache
    modality, X, z1, a1, mask1, z2, emb = enc_cache
    emb_h, h_z1, h_a1, h_mask, h_z2 = head_cache
    n = X.shape[0]
    head_out = y_pred.shape[1] if y_pred.ndim > 1 else 1

    # dMSE/d(output) = 2*(y_pred - y_true)/n
    dz_head_out = (2.0 * (y_pred - y_true) / n)
    if dz_head_out.ndim == 1:
        dz_head_out = dz_head_out.reshape(-1, 1)

    dW_head2 = h_a1.T @ dz_head_out
    db_head2 = dz_head_out.sum(axis=0)

    da_h1 = dz_head_out @ p["head_W2"].T
    if h_mask is not None: da_h1 = da_h1 * h_mask
    dz_h1 = da_h1 * drelu(h_z1)
    dW_head1 = emb_h.T @ dz_h1
    db_head1 = dz_h1.sum(axis=0)
    d_emb = dz_h1 @ p["head_W1"].T

    dz_enc2 = d_emb * drelu(z2)
    if modality == "voice":
        dW_enc2 = a1.T @ dz_enc2; db_enc2 = dz_enc2.sum(axis=0)
        da_enc1 = dz_enc2 @ p["voice_W2"].T
        if mask1 is not None: da_enc1 = da_enc1 * mask1
        dz_enc1 = da_enc1 * drelu(z1)
        dW_enc1 = X.T @ dz_enc1; db_enc1 = dz_enc1.sum(axis=0)
        grads = {
            "voice_W1": dW_enc1, "voice_b1": db_enc1,
            "voice_W2": dW_enc2, "voice_b2": db_enc2,
            "head_W1": dW_head1, "head_b1": db_head1,
            "head_W2": dW_head2, "head_b2": db_head2,
        }
    else:
        dW_enc2 = a1.T @ dz_enc2; db_enc2 = dz_enc2.sum(axis=0)
        da_enc1 = dz_enc2 @ p["hw_W2"].T
        if mask1 is not None: da_enc1 = da_enc1 * mask1
        dz_enc1 = da_enc1 * drelu(z1)
        dW_enc1 = X.T @ dz_enc1; db_enc1 = dz_enc1.sum(axis=0)
        grads = {
            "hw_W1": dW_enc1, "hw_b1": db_enc1,
            "hw_W2": dW_enc2, "hw_b2": db_enc2,
            "head_W1": dW_head1, "head_b1": db_head1,
            "head_W2": dW_head2, "head_b2": db_head2,
        }

    if prox_mu > 0 and p_global is not None:
        for k in grads:
            grads[k] = grads[k] + prox_mu * (p[k] - p_global[k])
    return grads


# -------------------------------------------------------------------- #
# Adam optimiser
# -------------------------------------------------------------------- #

class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=1e-4):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, b1, b2, eps, wd
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in grads:
            g = grads[k] + self.wd * params[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            m_hat = self.m[k] / (1 - self.b1 ** self.t)
            v_hat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
