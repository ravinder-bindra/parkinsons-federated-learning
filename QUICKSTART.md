# Quickstart

Three commands to run everything:

```bash
pip install -r requirements.txt
bash scripts/run_all.sh          # or scripts\run_all.bat on Windows
streamlit run app/app.py
```

That's it. The first command installs dependencies (~1 min). The second
trains and evaluates the federated model end-to-end (~30 sec). The
third opens the dashboard at http://localhost:8501.

## Folder map

| Folder | Contents |
|---|---|
| `data/raw/` | Original datasets — UCI voice, NewHandPD handwriting, Telemonitoring UPDRS |
| `data/processed/` | Cleaned, patient-aware splits ready for training |
| `src/` | Source code (preprocess, model, trainer, evaluate, UPDRS) |
| `app/` | Streamlit dashboard |
| `scripts/` | Bash + batch runners |
| `tests/` | Test suite (12 tests, run with `python tests/test_pipeline.py`) |
| `docs/` | Architecture documentation |
| `results/` | Trained model, plots, predictions (already populated) |

## If `bash` isn't available (Windows without WSL)

Use the batch script:
```cmd
scripts\run_all.bat
```

Or run the steps manually:
```cmd
python -m src.preprocess
python -m src.federated_trainer --rounds 15
python -m src.evaluate
python -m src.updrs_regression
python tests\test_pipeline.py
```

## What to look at

After running, the most useful artefacts:
- `results/figures/training_curves.png` — see the model learn round by round
- `results/figures/roc_pr_combined.png` — performance on both modalities
- `results/figures/feature_importance_voice.png` and `_handwriting.png` — what the model uses
- `results/run_config.json` — final metrics
- The Streamlit dashboard — interactive predictions on uploaded CSVs

## Already populated

You don't actually have to run anything to see the results — `results/`
already contains the trained model, all plots, and prediction CSVs
from when this project was built. Re-running just reproduces them.

See `README.md` for the full documentation.
