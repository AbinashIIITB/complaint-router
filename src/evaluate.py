"""
evaluate.py
-----------
Generates evaluation plots and confusion matrix from saved models.

Outputs:
    reports/confusion_matrix.png    — seaborn heatmap, normalized
    reports/per_class_metrics.png   — grouped bar chart: precision/recall/F1
    reports/baseline_cm.png         — baseline confusion matrix

Run:
    python src/evaluate.py
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

DEPARTMENTS = ["Roads", "Water", "Electricity", "Sanitation", "Traffic", "Parks"]
PALETTE = {
    "Roads":       "#E76F51",
    "Water":       "#2A9D8F",
    "Electricity": "#F4A261",
    "Sanitation":  "#6A0572",
    "Traffic":     "#264653",
    "Parks":       "#57CC99",
}
DEPT_COLORS = [PALETTE[d] for d in DEPARTMENTS]


def load_test_data() -> tuple[pd.Series, pd.Series]:
    from sklearn.model_selection import train_test_split
    df = pd.read_csv("data/processed/complaints_clean.csv").dropna(subset=["text_clean", "department"])
    _, X_test, _, y_test = train_test_split(
        df["text_clean"], df["department"], test_size=0.2, random_state=42, stratify=df["department"]
    )
    return X_test, y_test


def plot_confusion_matrix(y_true, y_pred, model_name: str, out_path: str):
    cm = confusion_matrix(y_true, y_pred, labels=DEPARTMENTS, normalize="true")
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=DEPARTMENTS, yticklabels=DEPARTMENTS,
        ax=ax, linewidths=0.5, linecolor="#dee2e6",
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(f"Confusion Matrix — {model_name}\n(Normalized by true label)", fontsize=14, pad=15)
    ax.set_ylabel("True Department", fontsize=12)
    ax.set_xlabel("Predicted Department", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_per_class_metrics(y_true, y_pred, out_path: str):
    report = classification_report(y_true, y_pred, labels=DEPARTMENTS, output_dict=True)
    metrics_data = {
        dept: {
            "Precision": report[dept]["precision"],
            "Recall":    report[dept]["recall"],
            "F1-Score":  report[dept]["f1-score"],
        }
        for dept in DEPARTMENTS
    }
    df_metrics = pd.DataFrame(metrics_data).T

    x = np.arange(len(DEPARTMENTS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(13, 6))

    bars_p = ax.bar(x - width,   df_metrics["Precision"], width, label="Precision", color="#264653", alpha=0.85)
    bars_r = ax.bar(x,           df_metrics["Recall"],    width, label="Recall",    color="#2A9D8F", alpha=0.85)
    bars_f = ax.bar(x + width,   df_metrics["F1-Score"],  width, label="F1-Score",  color="#E76F51", alpha=0.85)

    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(DEPARTMENTS, fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Per-Class Precision / Recall / F1 — TF-IDF + Logistic Regression", fontsize=14, pad=15)
    ax.legend(fontsize=11)
    ax.axhline(0.85, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="0.85 reference")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bars in [bars_p, bars_r, bars_f]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_f1_comparison(metrics_json_path: str, out_path: str):
    """Macro F1 comparison bar chart: baseline vs main model."""
    with open(metrics_json_path) as f:
        data = json.load(f)

    models = ["BoW + Naive Bayes\n(baseline)", "TF-IDF + LogReg\n(main model)"]
    f1s = [data["baseline"]["macro_f1"], data["main"]["macro_f1"]]
    colors = ["#adb5bd", "#E76F51"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(models, f1s, color=colors, width=0.4, edgecolor="white", linewidth=0.5)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Macro F1 Score", fontsize=12)
    ax.set_title("Macro F1 — Baseline vs Main Model", fontsize=14, pad=15)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, f1 in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, f1 + 0.02,
                f"{f1:.4f}", ha="center", va="bottom", fontsize=13, fontweight="bold")

    lift = f1s[1] - f1s[0]
    ax.annotate(
        f"+{lift:.4f} lift",
        xy=(1, f1s[1]), xytext=(0.5, f1s[1] + 0.06),
        fontsize=11, color="#2A9D8F", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#2A9D8F"),
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def assert_f1(threshold: float = 0.82):
    """Exit with error if main model macro F1 < threshold."""
    with open("reports/metrics.json") as f:
        data = json.load(f)
    f1 = data["main"]["macro_f1"]
    if f1 < threshold:
        print(f"[FAIL] Macro F1 {f1:.4f} < {threshold}")
        raise SystemExit(1)
    print(f"[PASS] Macro F1 {f1:.4f} >= {threshold}")


def main():
    os.makedirs("reports", exist_ok=True)

    print("Loading test data...")
    X_test, y_test = load_test_data()

    print("Loading baseline model...")
    baseline = joblib.load("models/baseline_bow_nb.pkl")
    y_pred_baseline = baseline.predict(X_test)

    print("Loading main model...")
    main_model = joblib.load("models/tfidf_logreg.pkl")
    y_pred_main = main_model.predict(X_test)

    # Confusion matrices
    plot_confusion_matrix(y_test, y_pred_baseline, "BoW + Naive Bayes", "reports/baseline_cm.png")
    plot_confusion_matrix(y_test, y_pred_main,     "TF-IDF + LogReg",   "reports/confusion_matrix.png")

    # Per-class bar chart
    plot_per_class_metrics(y_test, y_pred_main, "reports/per_class_metrics.png")

    # F1 comparison
    if os.path.exists("reports/metrics.json"):
        plot_f1_comparison("reports/metrics.json", "reports/f1_comparison.png")

    print("\nAll evaluation plots saved to reports/")


if __name__ == "__main__":
    import sys
    if "--assert-f1" in sys.argv:
        idx = sys.argv.index("--assert-f1")
        thresh = float(sys.argv[idx + 1]) if idx + 1 < len(sys.argv) else 0.82
        assert_f1(thresh)
    else:
        main()
