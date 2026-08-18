"""
priority.py
-----------
Rule-based severity and priority scorer.

Entirely independent of the ML classifier — designed for interpretability.
An auditor can inspect exactly which signals drove a High priority flag,
which is critical in civic/government contexts.

Priority Score: 1–10
Labels:
    1–3  → Low
    4–6  → Medium
    7–10 → High

Score formula:
    severity_score = weighted sum of matched severity keywords (capped at 5)
    temporal_score = weighted sum of matched temporal keywords (capped at 3)
    base           = 2
    raw            = base + severity_score + temporal_score
    priority       = min(10, raw)
"""

import re
from dataclasses import dataclass, field
from typing import List

# ── Keyword dictionaries ─────────────────────────────────────────────────────

SEVERITY_KEYWORDS: dict[str, int] = {
    # Critical (weight 3) — life/safety risks
    "accident": 3,
    "injury": 3,
    "injured": 3,
    "hurt": 3,
    "child": 3,
    "children": 3,
    "school": 3,
    "hospital": 3,
    "flood": 3,
    "collapse": 3,
    "fire": 3,
    "gas leak": 3,
    "live wire": 3,
    "electric shock": 3,
    "fallen": 3,
    "danger": 3,
    "dangerous": 3,
    "serious": 3,
    "emergency": 3,
    "life": 3,
    "death": 3,
    "fatal": 3,
    "outbreak": 3,
    "disease": 3,
    "contaminated": 3,
    "sewage mix": 3,
    # Significant (weight 2)
    "chaos": 2,
    "blocked": 2,
    "overflow": 2,
    "overflowing": 2,
    "broken": 2,
    "no supply": 2,
    "no water": 2,
    "power cut": 2,
    "blackout": 2,
    "damaged": 2,
    "hazard": 2,
    "unsafe": 2,
    "risk": 2,
    "suffer": 2,
    "suffering": 2,
    "sick": 2,
    "ill": 2,
    "stuck": 2,
    "accident prone": 2,
    "mosquito": 2,
    "stagnant": 2,
    "rats": 2,
    "rat": 2,
    # Minor (weight 1)
    "inconvenience": 1,
    "request": 1,
    "minor": 1,
    "nuisance": 1,
    "untidy": 1,
    "overgrown": 1,
    "foul smell": 1,
    "bad smell": 1,
}

TEMPORAL_KEYWORDS: dict[str, int] = {
    "month": 3,
    "months": 3,
    "weeks": 3,
    "week": 2,
    "days": 2,
    "daily": 2,
    "since": 2,
    "still": 2,
    "again": 2,
    "repeatedly": 2,
    "multiple": 2,
    "long time": 2,
    "yesterday": 1,
    "today": 1,
    "hours": 1,
}


@dataclass
class PriorityResult:
    priority_score: int
    priority_label: str           # Low / Medium / High
    severity_score: int
    temporal_score: int
    matched_signals: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "priority_score": self.priority_score,
            "priority_label": self.priority_label,
            "severity_score": self.severity_score,
            "temporal_score": self.temporal_score,
            "matched_signals": self.matched_signals,
        }


def _label(score: int) -> str:
    if score <= 3:
        return "Low"
    elif score <= 6:
        return "Medium"
    return "High"


def score_priority(text: str) -> PriorityResult:
    """
    Compute priority score for a complaint.

    Args:
        text: Raw complaint text (preprocessing not required).

    Returns:
        PriorityResult with score, label, and matched signals.
    """
    text_lower = text.lower()
    matched = []

    # Severity pass
    sev_total = 0
    for kw, weight in sorted(SEVERITY_KEYWORDS.items(), key=lambda x: -x[1]):
        if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
            sev_total += weight
            matched.append(kw)
    sev_capped = min(sev_total, 5)

    # Temporal pass
    temp_total = 0
    for kw, weight in sorted(TEMPORAL_KEYWORDS.items(), key=lambda x: -x[1]):
        if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
            temp_total += weight
            matched.append(kw)
    temp_capped = min(temp_total, 3)

    raw = 2 + sev_capped + temp_capped
    final_score = min(10, raw)

    return PriorityResult(
        priority_score=final_score,
        priority_label=_label(final_score),
        severity_score=sev_capped,
        temporal_score=temp_capped,
        matched_signals=matched[:6],  # return top 6 matched signals
    )


if __name__ == "__main__":
    examples = [
        "The signal near Rasulgarh has been blinking yellow for three days, it's chaos at peak hour.",
        "Children playing in the park, the swings are broken.",
        "No water supply since last month. We are sick and suffering.",
        "The streetlight at Unit-4 is not working.",
        "Live wire fallen on the road near Patia school. Children are in danger.",
    ]
    print("Priority scorer demo\n" + "=" * 50)
    for ex in examples:
        r = score_priority(ex)
        print(f"\nText: {ex[:70]}...")
        print(f"  Score: {r.priority_score}/10  |  Label: {r.priority_label}")
        print(f"  Signals: {r.matched_signals}")
