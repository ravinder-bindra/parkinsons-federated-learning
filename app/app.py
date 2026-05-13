"""
app/app.py
----------
Multi-modal clinical dashboard. Loads the trained federated model from
results/global_model.npz and offers:

  * Voice screening tab    -- upload a 22-feature voice CSV
  * Handwriting tab        -- upload a NewHandPD-format CSV
  * Model performance tab  -- shows the training plots from results/figures/

Run:
    streamlit run app/app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import forward


st.set_page_config(
    page_title="Multi-Modal Federated PD Dashboard",
    page_icon="🩺",
    layout="wide",
)


@st.cache_resource
def load_model():
    path = Path("results/global_model.npz")
    if not path.exists():
        return None
    return dict(np.load(path))


@st.cache_data
def load_history():
    p = Path("results/round_history.csv")
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def load_hw_scaler():
    p = Path("data/processed/handwriting_scaler.csv")
    if not p.exists():
        return None
    return pd.read_csv(p)


def predict_voice(model, X):
    out, _ = forward(model, X, "voice", dropout_p=0.0)
    return out


def predict_hw(model, X):
    out, _ = forward(model, X, "handwriting", dropout_p=0.0)
    return out


# ---------- UI ----------
st.title("🩺 Multi-Modal Parkinson's Diagnostic Dashboard")
st.caption("Federated learning across 4 hospital clients on 2 modalities — "
           "voice (22 features) and handwriting (21 features).")

model = load_model()
history = load_history()
hw_scaler = load_hw_scaler()

with st.sidebar:
    st.header("📊 Model card")
    if history is not None:
        last = history.iloc[-1]
        st.metric("Voice AUC", f"{last['voice_auc']:.3f}")
        st.metric("Voice bal-acc", f"{last['voice_balanced_acc']:.3f}")
        st.markdown("---")
        st.metric("Handwriting AUC", f"{last['hw_auc']:.3f}")
        st.metric("Handwriting bal-acc", f"{last['hw_balanced_acc']:.3f}")
    st.markdown("---")
    st.markdown(
        "**Federated setup**\n"
        "- 3 voice clients (UCI Parkinson's, 52 patients each)\n"
        "- 1 handwriting client (NewHandPD, 50 patients)\n"
        "- 15 rounds, FedProx μ=0.1\n"
        "- Modality-aware aggregation\n"
        "- Raw data never leaves hospitals"
    )

if model is None:
    st.error("Model not found. Run: `bash scripts/run_all.sh`")
    st.stop()

tab_voice, tab_hw, tab_perf = st.tabs([
    "🗣️ Voice Screening",
    "✍️ Handwriting Screening",
    "📈 Model Performance",
])

# ======= Voice tab =======
with tab_voice:
    st.subheader("Voice-based Screening")
    st.write("Upload a CSV with 22 normalised acoustic features "
             "(MDVP family). Use `data/processed/voice_test.csv` for a demo.")
    f = st.file_uploader("Voice CSV", type="csv", key="voice")
    if f is not None:
        df = pd.read_csv(f)
        feat = df.drop("status", axis=1) if "status" in df.columns else df
        if feat.shape[1] != 22:
            st.error(f"Expected 22 features, got {feat.shape[1]}")
        else:
            X = feat.values.astype(np.float64)
            probs = predict_voice(model, X)

            mode = st.radio("Mode", ["Individual", "Bulk triage"],
                            horizontal=True, key="vmode")
            if mode == "Individual":
                idx = st.slider("Patient row", 0, len(feat)-1, 0, key="vidx")
                st.dataframe(feat.iloc[[idx]])
                if st.button("🧠 Diagnose", type="primary",
                             use_container_width=True, key="vbtn"):
                    p = float(probs[idx])
                    if p > 0.5:
                        st.error(f"⚠️ HIGH RISK — P(PD) = {p:.1%}")
                    else:
                        st.success(f"✅ CLEAR — P(PD) = {p:.1%}")
                    st.progress(min(max(p, 0.0), 1.0))
            else:
                if st.button("🚨 Bulk scan", use_container_width=True, key="vbulk"):
                    out = pd.DataFrame({
                        "Patient ID": [f"V-{i:04d}" for i in range(len(feat))],
                        "Risk score": probs,
                        "Verdict": np.where(probs > 0.5, "🚨 High risk", "✅ Clear"),
                    }).sort_values("Risk score", ascending=False)
                    st.dataframe(out, use_container_width=True, hide_index=True)
                    st.download_button("📥 Download report",
                        data=out.to_csv(index=False).encode(),
                        file_name="voice_triage.csv", mime="text/csv")

# ======= Handwriting tab =======
with tab_hw:
    st.subheader("Handwriting-based Screening")
    st.write("Upload a CSV with handwriting features. Use "
             "`data/processed/handwriting_test.csv` for a demo. The model "
             "expects 18 kinematic features (9 from spiral + 9 from meander) "
             "plus AGE, GENDER, RIGHT_HANDED.")
    f = st.file_uploader("Handwriting CSV", type="csv", key="hw")
    if f is not None:
        df = pd.read_csv(f)
        hw_feat_names = ([f"spiral_{c}" for c in
                          ["RMS","MAX_BETWEEN_ET_HT","MIN_BETWEEN_ET_HT",
                           "STD_DEVIATION_ET_HT","MRT","MAX_HT","MIN_HT",
                           "STD_HT","CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_ET_HT"]]
                         + [f"meander_{c}" for c in
                            ["RMS","MAX_BETWEEN_ET_HT","MIN_BETWEEN_ET_HT",
                             "STD_DEVIATION_ET_HT","MRT","MAX_HT","MIN_HT",
                             "STD_HT","CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_ET_HT"]]
                         + ["AGE","GENDER","RIGHT_HANDED"])
        missing = [c for c in hw_feat_names if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing[:5]}...")
        else:
            feat = df[hw_feat_names]
            X = feat.values.astype(np.float64)
            probs = predict_hw(model, X)

            idx = st.slider("Patient row", 0, len(feat)-1, 0, key="hidx")
            st.dataframe(feat.iloc[[idx]])
            if st.button("🧠 Diagnose", type="primary",
                         use_container_width=True, key="hbtn"):
                p = float(probs[idx])
                if p > 0.5:
                    st.error(f"⚠️ HIGH RISK — P(PD) = {p:.1%}")
                else:
                    st.success(f"✅ CLEAR — P(PD) = {p:.1%}")
                st.progress(min(max(p, 0.0), 1.0))

            if st.button("🚨 Bulk scan all", use_container_width=True, key="hbulk"):
                out = pd.DataFrame({
                    "Patient ID": (df["patient_id"].values if "patient_id" in df.columns
                                   else [f"H-{i:04d}" for i in range(len(feat))]),
                    "Risk score": probs,
                    "Verdict": np.where(probs > 0.5, "🚨 High risk", "✅ Clear"),
                }).sort_values("Risk score", ascending=False)
                st.dataframe(out, use_container_width=True, hide_index=True)

# ======= Performance tab =======
with tab_perf:
    st.subheader("Federated training performance")
    if history is not None:
        c1, c2 = st.columns(2)
        c1.markdown("**Voice**")
        c1.line_chart(history.set_index("round")[["voice_auc","voice_balanced_acc","voice_acc"]])
        c2.markdown("**Handwriting**")
        c2.line_chart(history.set_index("round")[["hw_auc","hw_balanced_acc","hw_acc"]])

    fig_dir = Path("results/figures")
    if fig_dir.exists():
        st.markdown("### Generated plots")
        for fp in sorted(fig_dir.glob("*.png")):
            st.image(str(fp), caption=fp.stem.replace("_"," ").title())
