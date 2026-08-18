"""
predict.py
----------
Inference wrapper used by the Streamlit UI and smoke tests.

Loads the saved TF-IDF + Logistic Regression pipeline once (lazily, on
first call) and runs it on a complaint string.  Also calls the rule-based
priority scorer to produce a severity label and matched signals.

Usage (from Python):
    from src.predict import predict
    result = predict("There has been no water supply in Rasulgarh for three days.")
    print(result.summary())

Usage (from the command line):
    python src/predict.py --test          # run 6 smoke tests
    python src/predict.py --text "..."   # route a single complaint
"""

import os
import sys
import joblib
from dataclasses import dataclass, field
from typing import List, Dict

# ---------------------------------------------------------------------------
# Model cache — loaded once, reused on every subsequent call.
# ---------------------------------------------------------------------------
_main_model = None


def _load_main():
    """Load the TF-IDF + LogReg pipeline from disk (cached after first load)."""
    global _main_model
    if _main_model is None:
        path = "models/tfidf_logreg.pkl"
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found.  Run:  python src/train.py"
            )
        _main_model = joblib.load(path)
    return _main_model


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
DEPT_EMOJI = {
    "Roads":       "🛣️",
    "Water":       "💧",
    "Electricity": "⚡",
    "Sanitation":  "🗑️",
    "Traffic":     "🚦",
    "Parks":       "🌳",
}

PRIORITY_EMOJI = {
    "Low":    "🟢",
    "Medium": "🟡",
    "High":   "🔴",
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class PredictionResult:
    """Holds everything the UI needs after routing one complaint."""
    department: str
    confidence: float          # probability 0–1 from the classifier
    top_alternatives: List[Dict]  # next two most likely departments
    priority_score: int        # 1–10 from the rule-based scorer
    priority_label: str        # "Low" / "Medium" / "High"
    matched_signals: List[str] # keywords that drove the priority score

    # These are filled automatically after __init__
    dept_emoji: str = field(default="", init=False)
    priority_emoji: str = field(default="", init=False)

    def __post_init__(self):
        self.dept_emoji     = DEPT_EMOJI.get(self.department, "🏛️")
        self.priority_emoji = PRIORITY_EMOJI.get(self.priority_label, "⚪")

    def summary(self) -> str:
        """One-line human-readable summary."""
        return (
            f"{self.dept_emoji} {self.department} | "
            f"Confidence: {self.confidence * 100:.1f}% | "
            f"{self.priority_emoji} Priority: {self.priority_label} ({self.priority_score}/10)"
        )


# ---------------------------------------------------------------------------
# Core prediction function
# ---------------------------------------------------------------------------
def predict(text: str) -> PredictionResult:
    """
    Route a free-text civic complaint to a department and score its priority.

    Args:
        text: Raw complaint string.  No preprocessing needed — the trained
              TF-IDF pipeline handles tokenisation internally.

    Returns:
        A PredictionResult with department, confidence, alternatives, and
        priority details.
    """
    # Import here so that the priority module is found regardless of
    # whether this script is run from repo root or from app/.
    _setup_path()
    from src.priority import score_priority

    model = _load_main()

    # predict_proba returns a probability for each department
    proba   = model.predict_proba([text])[0]
    classes = model.classes_

    # Sort by probability, highest first
    ranked = proba.argsort()[::-1]

    top_dept       = classes[ranked[0]]
    top_confidence = float(proba[ranked[0]])

    # The next two departments — shown as "alternatives" in the UI
    alternatives = [
        {"department": classes[i], "probability": float(proba[i])}
        for i in ranked[1:3]
    ]

    prio = score_priority(text)

    return PredictionResult(
        department=top_dept,
        confidence=top_confidence,
        top_alternatives=alternatives,
        priority_score=prio.priority_score,
        priority_label=prio.priority_label,
        matched_signals=prio.matched_signals,
    )


def _setup_path():
    """Ensure repo root is on sys.path so 'from src.xxx import yyy' works."""
    # This script can be run from:
    #   repo root  → ROOT == current directory
    #   app/       → ROOT == parent directory
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)


# ---------------------------------------------------------------------------
# Smoke tests — one complaint per department
# ---------------------------------------------------------------------------
SMOKE_TESTS = [
    {
        "text": "The signal near Rasulgarh has been blinking yellow for three days, it's chaos at peak hour.",
        "expected_dept": "Traffic",
        "expected_priority": "Medium",   # chaos(2) + days(2) + base(2) = 6 → Medium
    },
    {
        "text": "Garbage has not been collected in Unit-4 for two weeks now. Rats are appearing.",
        "expected_dept": "Sanitation",
        "expected_priority": "High",     # rats(2) + weeks(3) + base(2) = 7 → High
    },
    {
        "text": "No water supply in our colony near Chandrasekharpur since last Monday. We are suffering.",
        "expected_dept": "Water",
        "expected_priority": "High",
    },
    {
        "text": "There is a huge pothole near Saheed Nagar. Vehicles are getting damaged.",
        "expected_dept": "Roads",
        "expected_priority": "Medium",
    },
    {
        "text": "Live electric wire has fallen on the road near Patia school. Children are in danger.",
        "expected_dept": "Electricity",
        "expected_priority": "High",
    },
    {
        "text": "The park near IRC Village has broken swings and slides. Children are getting injured.",
        "expected_dept": "Parks",
        "expected_priority": "High",
    },
]


def run_tests():
    """Run smoke tests and print pass/fail for each case."""
    print("Running smoke tests...\n")
    passed = 0

    for tc in SMOKE_TESTS:
        result = predict(tc["text"])
        dept_ok  = result.department    == tc["expected_dept"]
        prio_ok  = result.priority_label == tc["expected_priority"]
        all_ok   = dept_ok and prio_ok

        status = "✓ PASS" if all_ok else "✗ FAIL"
        print(f"{status} | {result.summary()}")
        print(f"       Text: {tc['text'][:80]}...")
        if not dept_ok:
            print(f"       Expected dept: {tc['expected_dept']} — got: {result.department}")
        if not prio_ok:
            print(f"       Expected priority: {tc['expected_priority']} — got: {result.priority_label}")
        print()

        if all_ok:
            passed += 1

    print(f"Result: {passed}/{len(SMOKE_TESTS)} tests passed")
    return passed == len(SMOKE_TESTS)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    _setup_path()

    parser = argparse.ArgumentParser(description="Civic Complaint Router — inference CLI")
    parser.add_argument("--test", action="store_true", help="Run smoke tests")
    parser.add_argument("--text", type=str, default=None, help="Route a single complaint")
    args = parser.parse_args()

    if args.test:
        ok = run_tests()
        sys.exit(0 if ok else 1)
    elif args.text:
        r = predict(args.text)
        print(r.summary())
        print(f"Signals:      {r.matched_signals}")
        print(f"Alternatives: {r.top_alternatives}")
    else:
        parser.print_help()
