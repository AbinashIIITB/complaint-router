"""
streamlit_app.py
----------------
Civic Complaint Router — Streamlit UI

Entry point for Streamlit Community Cloud.
Paste a complaint to get department, confidence, priority, and matched signals.
"""

import os
import sys
import json
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Complaint Router",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Classical NLP complaint routing. No LLMs. Built with TF-IDF + scikit-learn.",
    },
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] { padding-top: 1.5rem; }

    div[data-testid="metric-container"] {
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        padding: 0.75rem 1rem;
    }

    .signal-pill {
        display: inline-block;
        background: #e9ecef;
        border-radius: 4px;
        padding: 2px 10px;
        margin: 2px 3px;
        font-size: 0.82rem;
        color: #343a40;
    }

    .badge-high   { background:#fde8e8; color:#b91c1c; border:1px solid #fca5a5; }
    .badge-medium { background:#fef9e7; color:#92400e; border:1px solid #fcd34d; }
    .badge-low    { background:#ecfdf5; color:#065f46; border:1px solid #6ee7b7; }
    .priority-badge {
        display: inline-block;
        border-radius: 4px;
        padding: 3px 12px;
        font-weight: 600;
        font-size: 0.9rem;
    }

    .block-container { padding-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Example complaints ─────────────────────────────────────────────────────────
EXAMPLE_COMPLAINTS = [
    ("Traffic",     "The signal near Rasulgarh has been blinking yellow for three days, it's chaos at peak hour."),
    ("Roads",       "There is a huge pothole on the main road near Saheed Nagar. Two wheelers have had accidents. Needs urgent repair."),
    ("Water",       "No water supply in our colony near Chandrasekharpur since last Monday. We are suffering and have to buy water."),
    ("Electricity", "The streetlight near Unit-4 market has been out for a week. The area is completely dark and unsafe at night."),
    ("Sanitation",  "Garbage has not been collected near Bomikhal for two weeks. Rats are appearing and it is a health hazard."),
    ("Parks",       "The park near IRC Village has broken swings and slides. Children are getting injured. Please fix urgently."),
]

# ── Model loader (cached) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    """Load both pipelines once. Cached across Streamlit reruns."""
    import joblib
    models_dir    = os.path.join(ROOT, "models")
    main_path     = os.path.join(models_dir, "tfidf_logreg.pkl")
    baseline_path = os.path.join(models_dir, "baseline_bow_nb.pkl")
    main_model     = joblib.load(main_path)     if os.path.exists(main_path)     else None
    baseline_model = joblib.load(baseline_path) if os.path.exists(baseline_path) else None
    return main_model, baseline_model


@st.cache_data(show_spinner=False)
def load_metrics():
    metrics_path = os.path.join(ROOT, "reports", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            return json.load(f)
    return None


def get_prediction(text: str, main_model):
    from src.priority import score_priority
    proba   = main_model.predict_proba([text])[0]
    classes = main_model.classes_
    top_idx = proba.argsort()[::-1]
    dept       = classes[top_idx[0]]
    confidence = float(proba[top_idx[0]])
    alts = [{"department": classes[i], "probability": float(proba[i])} for i in top_idx[1:3]]
    prio = score_priority(text)
    return dept, confidence, alts, prio


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Complaint Router")
    st.caption("TF-IDF + Logistic Regression. No LLMs.")
    st.divider()

    st.markdown("**Try an example**")
    selected_example = None
    for dept, ex_text in EXAMPLE_COMPLAINTS:
        label = f"{dept} — {ex_text[:45]}..."
        if st.button(label, key=f"ex_{dept}", use_container_width=True):
            selected_example = ex_text

    st.divider()
    metrics = load_metrics()
    if metrics:
        st.markdown("**Model performance**")
        c1, c2 = st.columns(2)
        c1.metric("Main F1",     f"{metrics['main']['macro_f1']:.3f}")
        c2.metric("Baseline F1", f"{metrics['baseline']['macro_f1']:.3f}")
        lift = metrics["main"]["macro_f1"] - metrics["baseline"]["macro_f1"]
        st.caption(f"TF-IDF + LR improves BoW + NB by **+{lift:.3f}** macro F1")

    st.divider()
    st.markdown(
        "**Departments:** Roads, Water, Electricity, Sanitation, Traffic, Parks\n\n"
        "**Corpus:** 800 labelled complaints, 6 classes\n\n"
        "**Stack:** spaCy, scikit-learn, Streamlit"
    )


# ── Main area ─────────────────────────────────────────────────────────────────
st.title("Civic Complaint Router")
st.write(
    "Routes municipal grievances to the correct department using classical NLP. "
    "No large language models — fully interpretable."
)
st.caption("TF-IDF + Logistic Regression")

st.divider()

# Load models
main_model, baseline_model = load_models()

if main_model is None:
    st.error(
        "Trained models not found. Run the training pipeline first:\n"
        "```bash\n"
        "python src/data_gen.py\n"
        "python src/preprocess.py\n"
        "python src/train.py\n"
        "python src/evaluate.py\n"
        "```"
    )
    st.stop()

# ── Input ─────────────────────────────────────────────────────────────────────
default_text = selected_example or ""
complaint_text = st.text_area(
    label="Complaint text",
    value=default_text,
    height=130,
    placeholder="e.g. The signal near Rasulgarh has been blinking yellow for three days...",
    help="Paste or type a free-text civic complaint. The model routes it to the correct department and scores its urgency.",
    key="complaint_input",
)

col_btn, _ = st.columns([1, 5])
with col_btn:
    route_clicked = st.button("Route complaint", type="primary", use_container_width=True)


# ── Prediction output ─────────────────────────────────────────────────────────
if route_clicked and complaint_text.strip():
    dept, confidence, alts, prio = get_prediction(complaint_text.strip(), main_model)

    badge_class = {
        "High":   "badge-high",
        "Medium": "badge-medium",
        "Low":    "badge-low",
    }.get(prio.priority_label, "badge-low")

    st.subheader("Result")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Department",  dept)
    m2.metric("Confidence",  f"{int(confidence * 100)}%")
    m3.metric("Severity",    f"{prio.severity_score} / 5")
    m4.metric("Temporal",    f"{prio.temporal_score} / 3")

    st.markdown(
        f"**Priority:** "
        f"<span class='priority-badge {badge_class}'>"
        f"{prio.priority_label} &nbsp; {prio.priority_score}/10"
        f"</span>",
        unsafe_allow_html=True,
    )

    st.markdown("**Signals detected:**")
    if prio.matched_signals:
        pills = "".join(
            f"<span class='signal-pill'>{s}</span>" for s in prio.matched_signals
        )
        st.markdown(pills, unsafe_allow_html=True)
    else:
        st.caption("No high-priority signals detected.")

    st.markdown("**Alternative departments:**")
    for a in alts:
        st.markdown(f"- {a['department']}: {int(a['probability'] * 100)}%")

    st.divider()

    # ── Model details expander ─────────────────────────────────────────────────
    with st.expander("Model details and evaluation", expanded=False):
        metrics = load_metrics()
        if metrics:
            st.markdown("#### Macro F1 — baseline vs main model")
            e1, e2, e3 = st.columns(3)
            e1.metric("BoW + NB (baseline)",  f"{metrics['baseline']['macro_f1']:.4f}")
            e2.metric(
                "TF-IDF + LR (main)",
                f"{metrics['main']['macro_f1']:.4f}",
                delta=f"+{metrics['main']['macro_f1'] - metrics['baseline']['macro_f1']:.4f}",
            )
            e3.metric(
                "Accuracy (main)",
                f"{metrics['main']['accuracy']:.4f}",
                help="Accuracy is misleading for imbalanced classes — macro F1 is the right metric.",
            )

            st.caption(
                "**Why macro F1?** If the model always predicted 'Roads' (20% of corpus), "
                "accuracy would be 20% but macro F1 collapses to near 0 for minority classes. "
                "class_weight='balanced' penalizes minority-class errors proportionally more."
            )

            st.markdown("#### Per-class metrics — TF-IDF + LR")
            per_class = metrics["main"]["per_class"]
            rows = [
                {
                    "Department": d,
                    "Precision":  m["precision"],
                    "Recall":     m["recall"],
                    "F1-Score":   m["f1"],
                    "Support":    m["support"],
                }
                for d, m in per_class.items()
            ]
            import pandas as pd
            df_table = pd.DataFrame(rows).set_index("Department")
            st.dataframe(
                df_table.style.background_gradient(subset=["F1-Score"], cmap="YlGn"),
                use_container_width=True,
            )

        cm_path  = os.path.join(ROOT, "reports", "confusion_matrix.png")
        pcm_path = os.path.join(ROOT, "reports", "per_class_metrics.png")
        f1c_path = os.path.join(ROOT, "reports", "f1_comparison.png")

        if os.path.exists(cm_path) and os.path.exists(pcm_path):
            tab1, tab2, tab3 = st.tabs(["Confusion matrix", "Per-class metrics", "F1 comparison"])
            with tab1:
                st.image(cm_path, use_container_width=True)
            with tab2:
                st.image(pcm_path, use_container_width=True)
            with tab3:
                if os.path.exists(f1c_path):
                    st.image(f1c_path, use_container_width=True)

    # ── Design decisions expander ──────────────────────────────────────────────
    with st.expander("Why classical ML? Design decisions", expanded=False):
        st.markdown("""
**TF-IDF over BoW**
- Bigrams capture phrases like *"pipe burst"*, *"no water"*, *"power cut"* as single features.
- `sublinear_tf=True` applies log(1+tf) to dampen high-frequency terms without discarding them.

**Logistic Regression over Naive Bayes**
- LR outputs calibrated probabilities — the confidence percentage shown above is meaningful.
- NB assumes feature independence, which is incorrect for civic text where "no water" is a phrase.

**class_weight='balanced'**
- Parks has only ~75 samples (9% of corpus). Without balancing, the model learns to ignore it.
- Balanced weighting means loss is proportional to `n_samples / (n_classes x n_per_class)`.

**Priority scorer — deliberately rule-based**
- An auditor needs to explain *why* a complaint was flagged High priority.
- A keyword list is inspectable and auditable; a neural network is not.

**Hardest boundary: Roads vs Traffic**
- Both departments share words: *signal*, *road*, *junction*, *accident*.
- TF-IDF bigrams help slightly (*"traffic signal"* vs *"road signal"*), but this remains the weakest boundary.
        """)

elif route_clicked and not complaint_text.strip():
    st.warning("Please enter a complaint before routing.")

else:
    st.info(
        "Paste a civic complaint above and click **Route complaint**, "
        "or pick one of the six examples from the sidebar."
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Classical NLP · TF-IDF + scikit-learn · spaCy lemmatization · No LLMs")
