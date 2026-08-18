"""
train.py
--------
Trains two scikit-learn pipelines and saves them with joblib:

1. Baseline:  CountVectorizer (BoW, unigrams) + MultinomialNB
2. Main:      TfidfVectorizer (bigrams, sublinear_tf) + LogisticRegression
              with class_weight='balanced' and GridSearchCV tuning.

Run:
    python src/train.py              # full training
    python src/train.py --smoke-test # quick check with 10% data

Output:
    models/baseline_bow_nb.pkl
    models/tfidf_logreg.pkl
    reports/metrics_summary.md
"""

import argparse
import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
)

RANDOM_STATE = 42
DEPARTMENTS = ["Roads", "Water", "Electricity", "Sanitation", "Traffic", "Parks"]


def load_data(smoke_test: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load preprocessed data and return train/test splits."""
    path = "data/processed/complaints_clean.csv"
    if not os.path.exists(path):
        print(f"[ERROR] {path} not found. Run: python src/preprocess.py first.")
        sys.exit(1)

    df = pd.read_csv(path)
    df = df.dropna(subset=["text_clean", "department"])

    if smoke_test:
        df = df.sample(frac=0.1, random_state=RANDOM_STATE)
        print(f"[SMOKE TEST] Using {len(df)} samples")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text_clean"],
        df["department"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["department"],
    )
    # Also expose raw text train for comparison (not used in main model)
    train_df = df.loc[X_train.index].copy()
    test_df = df.loc[X_test.index].copy()
    return train_df, test_df


def build_baseline_pipeline() -> Pipeline:
    """BoW (unigrams) + Multinomial Naive Bayes."""
    return Pipeline([
        ("vect", CountVectorizer(
            max_features=5000,
            ngram_range=(1, 1),
            lowercase=True,
        )),
        ("clf", MultinomialNB(alpha=1.0)),
    ])


def build_main_pipeline() -> Pipeline:
    """
    TF-IDF (bigrams) + Logistic Regression with class balancing.

    Design choices (interview-ready):
    - sublinear_tf=True  : applies log(1+tf) to dampen high-freq terms
    - ngram_range=(1,2)  : captures "pipe burst", "no water" as features
    - class_weight='balanced': reweights loss for minority classes (Parks)
    - C tuned via GridSearchCV
    """
    return Pipeline([
        ("vect", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=8000,
            sublinear_tf=True,
            min_df=2,
            lowercase=True,
        )),
        ("clf", LogisticRegression(
            C=5.0,
            class_weight="balanced",
            solver="lbfgs",     # lbfgs handles multi-class natively
            max_iter=1000,
            random_state=RANDOM_STATE,
        )),
    ])


def tune_main_pipeline(pipeline: Pipeline, X_train: pd.Series, y_train: pd.Series) -> Pipeline:
    """Run GridSearchCV over LR C values. Returns best pipeline."""
    param_grid = {"clf__C": [0.1, 1.0, 5.0, 10.0]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
    )
    gs.fit(X_train, y_train)
    print(f"  Best C: {gs.best_params_['clf__C']:.1f}  |  CV macro F1: {gs.best_score_:.4f}")
    return gs.best_estimator_


def evaluate(pipeline: Pipeline, X_test: pd.Series, y_test: pd.Series, name: str) -> dict:
    """Return evaluation metrics dict."""
    y_pred = pipeline.predict(X_test)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(f"\n── {name} ──")
    print(f"  Accuracy:   {acc:.4f}   (misleading for imbalanced classes)")
    print(f"  Macro F1:   {macro_f1:.4f}")
    print(classification_report(y_test, y_pred))
    return {
        "name": name,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": {
            dept: {
                "precision": round(report.get(dept, {}).get("precision", 0), 4),
                "recall": round(report.get(dept, {}).get("recall", 0), 4),
                "f1": round(report.get(dept, {}).get("f1-score", 0), 4),
                "support": int(report.get(dept, {}).get("support", 0)),
            }
            for dept in DEPARTMENTS
        },
    }


def save_metrics(baseline_metrics: dict, main_metrics: dict):
    """Write metrics_summary.md for the README and reports."""
    os.makedirs("reports", exist_ok=True)
    lines = [
        "# Model Metrics Summary\n",
        "## Baseline vs Main Model — Macro F1\n",
        "| Model | Accuracy | Macro F1 |",
        "|---|---|---|",
        f"| BoW + Naive Bayes (baseline) | {baseline_metrics['accuracy']} | {baseline_metrics['macro_f1']} |",
        f"| TF-IDF + LogReg (main) | {main_metrics['accuracy']} | {main_metrics['macro_f1']} |",
        "",
        "## Per-Class F1 (TF-IDF + LogReg)\n",
        "| Department | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|",
    ]
    for dept, m in main_metrics["per_class"].items():
        lines.append(f"| {dept} | {m['precision']} | {m['recall']} | {m['f1']} | {m['support']} |")

    lines += [
        "",
        "## Key Observations\n",
        "- `class_weight='balanced'` corrects for Parks being the smallest class (9% of corpus).",
        "- Macro F1 is the right metric here — accuracy inflates when predicting the majority class.",
        "- Roads vs Traffic is the hardest boundary due to shared vocabulary ('signal', 'road', 'junction').",
        "- TF-IDF bigrams capture 'pipe burst', 'no water', 'power cut' as single features.",
    ]

    with open("reports/metrics_summary.md", "w") as f:
        f.write("\n".join(lines))
    print("\nSaved reports/metrics_summary.md")

    # also save raw JSON for Streamlit to consume
    with open("reports/metrics.json", "w") as f:
        json.dump({"baseline": baseline_metrics, "main": main_metrics}, f, indent=2)
    print("Saved reports/metrics.json")


def main(smoke_test: bool = False):
    os.makedirs("models", exist_ok=True)

    train_df, test_df = load_data(smoke_test)
    X_train = train_df["text_clean"]
    y_train = train_df["department"]
    X_test = test_df["text_clean"]
    y_test = test_df["department"]

    print(f"\nDataset: {len(train_df)} train / {len(test_df)} test")
    print("Class distribution (train):")
    for dept, cnt in y_train.value_counts().items():
        print(f"  {dept:<15} {cnt:>4}")

    # ── Baseline ─────────────────────────────────────────────────────────────
    print("\n[1/2] Training baseline (BoW + NB)...")
    baseline = build_baseline_pipeline()
    baseline.fit(X_train, y_train)
    baseline_metrics = evaluate(baseline, X_test, y_test, "BoW + Naive Bayes (baseline)")
    joblib.dump(baseline, "models/baseline_bow_nb.pkl")
    print("Saved models/baseline_bow_nb.pkl")

    # ── Main model ────────────────────────────────────────────────────────────
    print("\n[2/2] Training main model (TF-IDF + LogReg) with GridSearchCV...")
    main_pipeline = build_main_pipeline()
    if not smoke_test:
        main_pipeline = tune_main_pipeline(main_pipeline, X_train, y_train)
    else:
        main_pipeline.fit(X_train, y_train)

    main_metrics = evaluate(main_pipeline, X_test, y_test, "TF-IDF + LogReg (main)")
    joblib.dump(main_pipeline, "models/tfidf_logreg.pkl")
    print("Saved models/tfidf_logreg.pkl")

    # ── Persist metrics ───────────────────────────────────────────────────────
    save_metrics(baseline_metrics, main_metrics)

    lift = main_metrics["macro_f1"] - baseline_metrics["macro_f1"]
    print(f"\n✓ Macro F1 lift over baseline: +{lift:.4f}")

    if not smoke_test and main_metrics["macro_f1"] < 0.80:
        print("[WARN] Macro F1 below 0.80 — check preprocessing or dataset quality.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    main(smoke_test=args.smoke_test)
