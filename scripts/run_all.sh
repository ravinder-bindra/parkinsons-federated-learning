#!/usr/bin/env bash
# scripts/run_all.sh — full multi-modal pipeline
#
# Usage:
#   bash scripts/run_all.sh           # standard run
#   bash scripts/run_all.sh --dp      # with DP-SGD

set -euo pipefail
cd "$(dirname "$0")/.."

EXTRA=""
if [[ "${1:-}" == "--dp" ]]; then EXTRA="--dp"; echo ">> DP-SGD enabled"; fi

echo "=== 1/5  Preprocess (UCI Voice + NewHandPD + Telemonitoring) ==="
python -m src.preprocess

echo ""
echo "=== 2/5  Multi-modal federated training (15 rounds) ==="
python -m src.federated_trainer --rounds 15 $EXTRA

echo ""
echo "=== 3/5  Evaluation plots ==="
python -m src.evaluate

echo ""
echo "=== 4/5  UPDRS regression (Telemonitoring) ==="
python -m src.updrs_regression

echo ""
echo "=== 5/5  Tests ==="
python tests/test_pipeline.py

echo ""
echo "Done. Artefacts:"
echo "  results/global_model.npz"
echo "  results/round_history.csv"
echo "  results/figures/*.png"
echo "  results/updrs/*"
echo ""
echo "Launch dashboard:  streamlit run app/app.py"
