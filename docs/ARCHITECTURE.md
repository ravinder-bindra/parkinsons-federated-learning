# Architecture Details

## The federated math

### Modality-aware FedProx

Standard FedAvg averages all parameters across all clients uniformly.
That breaks when clients have different modalities — Hospital 4 has no
voice data, so it shouldn't contribute to voice encoder weights. The
modality-aware aggregator we implement:

```
For each parameter key k in the global model:
    contributing_clients = clients_that_train_k
    weight_total = sum(sample_count[c] for c in contributing_clients)
    p_global[k] = sum(sample_count[c] * p_client[c][k] for c in contributing_clients) / weight_total
```

Specifically:
```
voice_encoder_keys → averaged over {H1, H2, H3}              (sample-weighted)
handwriting_encoder_keys → directly from H4                  (sole owner)
classifier_head_keys → averaged over {H1, H2, H3, H4}        (sample-weighted)
```

This is what enables hybrid horizontal+vertical federated learning. The
*horizontal* part is the three voice clients agreeing on voice-encoder
weights. The *vertical* part is the cross-modality classifier head
benefiting from both kinds of training signal.

### FedProx proximal term

Each client adds μ/2 · ||w − w_global||² to its local loss, with gradient
contribution μ · (w − w_global). This keeps client updates close to the
global model and helps with non-IID data (which we have: hospitals 1, 2, 3
have different class balances). Default μ = 0.1.

### Differential Privacy (DP-SGD)

When `--dp` is set, each local training step does:

1. For each sample i in the batch, compute its individual gradient gᵢ.
2. Clip each gᵢ to L2 norm ≤ C (default C = 1.0).
3. Sum the clipped gradients, add Gaussian noise N(0, σ²C²I) (σ = 0.5).
4. Divide by batch size and apply optimizer step.

This is the recipe from Abadi et al. (2016). It gives provable
differential privacy guarantees once you account for the composition
across batches and rounds (compute ε with a privacy accountant — we
log the operations but don't compute the budget automatically; use
`opacus` for that in production).

## Why a numpy-only implementation

The sandbox where this was developed had no internet (so no PyTorch
install), and the federated logic is small enough that a clean numpy
reference implementation is more useful than a PyTorch one for
understanding:

- Forward and backward passes are explicit, line-by-line.
- The gradient finite-difference test in `tests/` directly verifies the
  backprop math.
- No hidden behavior from `nn.Module` or autograd.
- Trains in seconds on a CPU.

For production deployment, swap the numpy MLP for a PyTorch model and
wrap each hospital in a Flower client. The aggregator logic in
`federated_trainer.aggregate()` translates directly to a Flower
`Strategy.aggregate_fit()` method.

## Why this architecture instead of late-fusion concat

A standard multimodal model concatenates per-modality embeddings then
passes them through a shared head:
```
y = head(concat(voice_emb, hw_emb))
```
That requires every sample to have both modalities. Our patients don't —
voice patients aren't in NewHandPD and vice versa. So instead each
modality has its own encoder feeding its own forward pass through the
shared head:
```
y_voice = head(voice_encoder(X_voice))
y_hw    = head(hw_encoder(X_hw))
```
The head's weights still get updated by both, which is the cross-modal
knowledge transfer. This is sometimes called "shared backbone, separate
inputs" or modality-specific encoding with a unified task head.

## What to swap when adding a third modality

Adding (say) gait time-series from PhysioNet GaitPDB:

1. Add `GaitEncoder` to `src/model.py` with input shape matching gait
   features (e.g. 16-channel time-series → 32-d embedding via 1D-CNN).
2. Add `GAIT_KEYS` tuple and update `init_params` to include the new
   parameters.
3. Add a `forward` branch for `modality == "gait"`.
4. Add a `backward_bce` branch handling the gait encoder gradients.
5. Add `keys_for_modality("gait")` returning `GAIT_KEYS + HEAD_KEYS`.
6. Add a gait client to `federated_trainer.federated_run` and update
   `aggregate` to handle the new key set.

Everything else (evaluation, dashboard, tests) reuses the same
infrastructure.
