"""Mizan - Arabic Sentiment dashboard.

Run:  streamlit run app/streamlit_app.py
"""
import json
import os
import sys
from pathlib import Path

# Streamlit puts the script's own directory (not the project root) on sys.path,
# so make the project root importable no matter how the app is launched.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
import streamlit as st

from app.preprocessing import normalize_arabic

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE, "models", "sentiment_pipeline.joblib")
METRICS_PATH = os.path.join(BASE, "reports", "metrics.json")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)


def predict(model, texts):
    frame = pd.DataFrame({"clean": [normalize_arabic(t) for t in texts]})
    proba = model.predict_proba(frame)
    classes = list(model.classes_)
    rows = []
    for text, p in zip(texts, proba):
        i = int(p.argmax())
        rows.append({"text": text, "label": classes[i],
                     "confidence": round(float(p[i]), 4)})
    return rows


st.set_page_config(page_title="Mizan | Arabic Sentiment", page_icon="⚖️", layout="wide")
st.title("⚖️ Mizan - Arabic Sentiment Analysis")
st.caption("Binary sentiment (positive / negative) for Modern Standard Arabic "
           "and Egyptian-dialect text. TF-IDF (word + char n-grams) + Logistic "
           "Regression, trained on 32k+ Arabic tweets.")

model = load_model()
metrics = load_metrics()

tab_single, tab_batch, tab_metrics = st.tabs(["Analyze text", "Batch (CSV)", "Model report"])

with tab_single:
    text = st.text_area("Arabic text", height=120,
                        placeholder="اكتب أو الصق نص عربي هنا…")
    if st.button("Analyze", type="primary"):
        if not text.strip():
            st.warning("Enter some text first.")
        else:
            r = predict(model, [text])[0]
            label, conf = r["label"], r["confidence"]
            (st.success if label == "POS" else st.error)(
                f"**{label}** — confidence {conf:.1%}")
            st.progress(conf)
            if conf < 0.65:
                st.info("Low confidence: the text is likely mixed or neutral. "
                        "The model is binary by design (see Model report).")

with tab_batch:
    st.write("Upload a CSV with a `text` column; get back one prediction per row.")
    up = st.file_uploader("CSV file", type=["csv"])
    if up and st.button("Run batch", type="primary"):
        dfb = pd.read_csv(up)
        if "text" not in dfb.columns:
            st.error("CSV must contain a 'text' column.")
        else:
            rows = predict(model, dfb.text.fillna("").astype(str).tolist())
            out = pd.DataFrame(rows)
            st.dataframe(out, width='stretch')
            st.download_button("Download predictions", out.to_csv(index=False),
                               "predictions.csv", "text/csv")

with tab_metrics:
    tm, dm = metrics["test_metrics"], metrics["egyptian_dialect_metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test accuracy", f"{tm['accuracy']:.1%}")
    c2.metric("Test macro F1", f"{tm['macro_f1']:.1%}")
    c3.metric("Dialect accuracy*", f"{dm['accuracy']:.1%}")
    c4.metric("Dialect macro F1*", f"{dm['macro_f1']:.1%}")
    st.caption("*Evaluated on held-out Egyptian-dialect tweets (ASTD) the model never saw in training.")
    st.subheader("Per-class metrics (test set)")
    pc = tm["per_class"]
    st.dataframe(pd.DataFrame({
        "class": ["NEG", "POS"],
        "precision": [pc["NEG"]["precision"], pc["POS"]["precision"]],
        "recall": [pc["NEG"]["recall"], pc["POS"]["recall"]],
        "f1": [pc["NEG"]["f1-score"], pc["POS"]["f1-score"]],
    }).round(4), width='stretch')
    st.subheader("Confusion matrix (test set)")
    cm = pd.DataFrame(tm["confusion_matrix"],
                      index=["true NEG", "true POS"],
                      columns=["pred NEG", "pred POS"])
    st.dataframe(cm, width='stretch')
    st.subheader("Model comparison (macro F1)")
    st.dataframe(pd.DataFrame(
        [{"model": k, **v} for k, v in metrics["candidates"].items()]
    ), width='stretch')
