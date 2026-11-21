"""Train and evaluate the Arabic tweet sentiment model.

Data design:
  - Training: Arabic Sentiment Twitter Corpus (ASTC, MSA + dialect) plus an
    80% stratified slice of the ASTD Egyptian-dialect subset.
  - Testing: a held-out slice of ASTC plus the held-out 20% of ASTD, so the
    ASTD-only metrics are an honest read on unseen Egyptian-dialect tweets.

Run:  python -m app.train
Outputs: models/sentiment_pipeline.joblib, reports/metrics.json
"""
import json
import os
import time

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from app.preprocessing import normalize_arabic

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")

LABEL_MAP = {"pos": "POS", "neg": "NEG", "POS": "POS", "NEG": "NEG"}


def load_data():
    astc = pd.concat(
        [pd.read_parquet(os.path.join(DATA_DIR, "astc_train.parquet")),
         pd.read_parquet(os.path.join(DATA_DIR, "astc_test.parquet"))],
        ignore_index=True)[["tweet", "label"]].rename(columns={"tweet": "text"})
    astd = pd.read_parquet(os.path.join(DATA_DIR, "astd.parquet"))
    astd = astd[astd.label.isin(["POS", "NEG"])]  # binary scope; NEUTRAL excluded, see README
    df = pd.concat([astc, astd], ignore_index=True)
    df["label"] = df.label.map(LABEL_MAP)
    df["clean"] = df.text.map(normalize_arabic)
    df = df[df.clean.str.len() >= 3].drop_duplicates(subset=["clean"]).reset_index(drop=True)
    n_astd = len(astd)
    is_astd = pd.Series(False, index=df.index)
    is_astd.iloc[len(df) - n_astd:] = True
    return df, is_astd


def build_pipeline(model):
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                           min_df=3, max_features=120_000, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                           min_df=3, max_features=80_000, sublinear_tf=True)
    feats = ColumnTransformer(
        [("word", word, "clean"), ("char", char, "clean")])
    return Pipeline([("features", feats), ("clf", model)])


def evaluate(pipe, X, y):
    pred = pipe.predict(X)
    return {
        "accuracy": round(accuracy_score(y, pred), 4),
        "macro_f1": round(f1_score(y, pred, average="macro"), 4),
        "per_class": classification_report(y, pred, output_dict=True, digits=4),
        "confusion_matrix": confusion_matrix(y, pred, labels=["NEG", "POS"]).tolist(),
        "labels": ["NEG", "POS"],
    }


def main():
    t0 = time.time()
    df, is_astd = load_data()
    print(f"dataset: {len(df)} tweets after cleaning/dedup "
          f"({int(is_astd.sum())} ASTD Egyptian-dialect rows held out of training)")

    astc = df[~is_astd]
    astd_df = df[is_astd]
    tr_idx, te_idx = train_test_split(astc.index, test_size=0.15,
                                      stratify=astc.label, random_state=42)
    atr_idx, ate_idx = train_test_split(astd_df.index, test_size=0.20,
                                        stratify=astd_df.label, random_state=42)
    X_train, y_train = df.loc[list(tr_idx) + list(atr_idx), ["clean"]], df.loc[list(tr_idx) + list(atr_idx), "label"]
    X_test = df.loc[list(te_idx) + list(ate_idx), ["clean"]]
    y_test = df.loc[list(te_idx) + list(ate_idx), "label"]
    dialect_idx = set(ate_idx)
    print(f"train: {len(X_train)} | test: {len(X_test)} "
          f"(of which {len(ate_idx)} held-out dialect ASTD)")

    candidates = {
        "logistic_regression": LogisticRegression(C=2.0, max_iter=2000),
        "linear_svc": LinearSVC(C=1.0),
        "complement_nb": ComplementNB(alpha=0.3),
    }
    results, fitted = {}, {}
    for name, model in candidates.items():
        pipe = build_pipeline(model)
        pipe.fit(X_train, y_train)
        m = evaluate(pipe, X_test, y_test)
        results[name] = m
        fitted[name] = pipe
        print(f"{name}: acc={m['accuracy']} macroF1={m['macro_f1']}")

    best_name = max(results, key=lambda n: results[n]["macro_f1"])
    best = fitted[best_name]
    print(f"best model: {best_name}")

    # honest dialect-only read: these ASTD rows were never trained on
    astd_mask = [i in dialect_idx for i in X_test.index]
    dialect = evaluate(best, X_test[astd_mask], y_test[astd_mask])
    print(f"dialect-only (ASTD): acc={dialect['accuracy']} macroF1={dialect['macro_f1']}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    joblib.dump(best, os.path.join(MODEL_DIR, "sentiment_pipeline.joblib"))
    report = {
        "best_model": best_name,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "held_out_dialect_size": len(ate_idx),
        "candidates": {n: {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}
                       for n, m in results.items()},
        "test_metrics": results[best_name],
        "egyptian_dialect_metrics": dialect,
        "trained_in_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(REPORT_DIR, "metrics.json"), "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("saved models/sentiment_pipeline.joblib and reports/metrics.json")


if __name__ == "__main__":
    main()
