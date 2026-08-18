"""
streamlit_app.py
----------------
Civic Complaint Router — Streamlit UI

Entry point for Streamlit Community Cloud.
Paste a complaint → get department, confidence, priority, and matched signals.
"""

import os
import sys
import json
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
# Allow running from repo root (Streamlit Cloud) or from app/ directory
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Civic Complaint Router",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Classical NLP complaint routing. No LLMs. Built with TF-IDF + scikit-learn.",
    },
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main header */
.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.08);
}
.main-header h1 {
    color: #ffffff;
    font-size: 2.4rem;
    font-weight: 700;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.5px;
}
.main-header p {
    color: rgba(255,255,255,0.65);
    font-size: 1rem;
    margin: 0;
}
.badge {
    display: inline-block;
    background: rgba(42,157,143,0.2);
    color: #2A9D8F;
    padding: 3px 12px;
    border-radius: 100px;
    font-size: 0.75rem;
    font-weight: 600;
    border: 1px solid rgba(42,157,143,0.4);
    margin-top: 0.75rem;
}

/* Result card */
.result-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px;
    padding: 1.8rem;
    margin-bottom: 1.5rem;
}
.dept-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 20px;
    border-radius: 100px;
    font-weight: 700;
    font-size: 1.25rem;
    margin-bottom: 1rem;
}
.priority-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 16px;
    border-radius: 100px;
    font-weight: 600;
    font-size: 0.9rem;
}
.priority-high   { background: rgba(231,111,81,0.18);  color: #E76F51; border: 1px solid rgba(231,111,81,0.4); }
.priority-medium { background: rgba(244,162,97,0.18);  color: #F4A261; border: 1px solid rgba(244,162,97,0.4); }
.priority-low    { background: rgba(87,204,153,0.18);  color: #57CC99; border: 1px solid rgba(87,204,153,0.4); }

.signal-chip {
    display: inline-block;
    background: rgba(255,255,255,0.07);
    color: rgba(255,255,255,0.75);
    padding: 3px 12px;
    border-radius: 100px;
    font-size: 0.8rem;
    margin: 2px;
    border: 1px solid rgba(255,255,255,0.12);
}
.metric-row {
    display: flex;
    gap: 1.5rem;
    margin-top: 1rem;
}
.metric-box {
    flex: 1;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    text-align: center;
}
.metric-label {
    color: rgba(255,255,255,0.45);
    font-size: 0.72rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
}
.metric-value {
    color: #ffffff;
    font-size: 1.4rem;
    font-weight: 700;
}
.progress-bar-container {
    background: rgba(255,255,255,0.08);
    border-radius: 100px;
    height: 8px;
    margin-top: 6px;
    overflow: hidden;
}
.progress-bar-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #2A9D8F, #57CC99);
    transition: width 0.5s ease;
}
.alt-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.4rem;
    font-size: 0.88rem;
    color: rgba(255,255,255,0.7);
}
.alt-bar {
    background: rgba(255,255,255,0.06);
    border-radius: 100px;
    height: 5px;
    flex: 1;
    margin: 0 12px;
    overflow: hidden;
}
.alt-bar-fill {
    height: 100%;
    border-radius: 100px;
    background: rgba(255,255,255,0.25);
}
.example-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
    cursor: pointer;
    font-size: 0.85rem;
    color: rgba(255,255,255,0.75);
    line-height: 1.5;
}
.example-card:hover {
    background: rgba(255,255,255,0.06);
    border-color: rgba(42,157,143,0.4);
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
DEPT_COLORS = {
    "Roads":       "#E76F51",
    "Water":       "#2A9D8F",
    "Electricity": "#F4A261",
    "Sanitation":  "#6A0572",
    "Traffic":     "#264653",
    "Parks":       "#57CC99",
}
DEPT_EMOJI = {
    "Roads": "🛣️", "Water": "💧", "Electricity": "⚡",
    "Sanitation": "🗑️", "Traffic": "🚦", "Parks": "🌳",
}
PRIORITY_CLASS = {"High": "priority-high", "Medium": "priority-medium", "Low": "priority-low"}
PRIORITY_EMOJI = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}

EXAMPLE_COMPLAINTS = [
    ("Traffic",     "The signal near Rasulgarh has been blinking yellow for three days, it's chaos at peak hour."),
    ("Roads",       "There is a huge pothole on the main road near Saheed Nagar. Two wheelers have had accidents. Needs urgent repair."),
    ("Water",       "No water supply in our colony near Chandrasekharpur since last Monday. We are suffering and have to buy water."),
    ("Electricity", "The streetlight near Unit-4 market has been out for a week. The area is completely dark and unsafe at night."),
    ("Sanitation",  "Garbage has not been collected near Bomikhal for two weeks. Rats are appearing and it is a health hazard."),
    ("Parks",       "The park near IRC Village has broken swings and slides. Children are getting injured. Please fix urgently."),
]


# ── Model loader (cached) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading ML models...")
def load_models():
    """Load both pipelines once. Cached across Streamlit reruns."""
    import joblib
    models_dir = os.path.join(ROOT, "models")
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
    st.markdown("### 🏛️ Complaint Router")
    st.markdown("**Classical NLP** — TF-IDF + Logistic Regression. Zero LLMs.")
    st.divider()

    st.markdown("#### 💡 Try an Example")
    selected_example = None
    for dept, ex_text in EXAMPLE_COMPLAINTS:
        color = DEPT_COLORS[dept]
        emoji = DEPT_EMOJI[dept]
        if st.button(f"{emoji} {ex_text[:55]}...", key=f"ex_{dept}", use_container_width=True):
            selected_example = ex_text

    st.divider()
    metrics = load_metrics()
    if metrics:
        st.markdown("#### 📊 Model Performance")
        col1, col2 = st.columns(2)
        col1.metric("Main Macro F1", f"{metrics['main']['macro_f1']:.3f}")
        col2.metric("Baseline F1",   f"{metrics['baseline']['macro_f1']:.3f}")
        lift = metrics['main']['macro_f1'] - metrics['baseline']['macro_f1']
        st.caption(f"TF-IDF+LR lifts BoW+NB by **+{lift:.3f}** macro F1")

    st.divider()
    st.markdown(
        "**Departments**: Roads · Water · Electricity · Sanitation · Traffic · Parks\n\n"
        "**Corpus**: 800 labelled complaints · 6 classes\n\n"
        "**Stack**: spaCy · scikit-learn · Streamlit"
    )


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏛️ Civic Complaint Router</h1>
    <p>Municipal grievance classification for Indian cities — routes complaints to the right department instantly</p>
    <span class="badge">Classical NLP · TF-IDF + Logistic Regression · No LLMs</span>
</div>
""", unsafe_allow_html=True)

# Load models
main_model, baseline_model = load_models()

if main_model is None:
    st.error(
        "⚠️ Trained models not found. Run the pipeline first:\n"
        "```bash\n"
        "python src/data_gen.py\n"
        "python src/preprocess.py\n"
        "python src/train.py\n"
        "python src/evaluate.py\n"
        "```"
    )
    st.stop()

# Text input — use example if sidebar button clicked
default_text = selected_example or ""
complaint_text = st.text_area(
    label="Describe your complaint",
    value=default_text,
    height=130,
    placeholder="e.g. The signal near Rasulgarh has been blinking yellow for three days, it's chaos at peak hour...",
    help="Paste or type a free-text civic complaint. The model routes it to the correct department and scores its urgency.",
    key="complaint_input",
)

col_btn, col_clear = st.columns([1, 5])
with col_btn:
    route_clicked = st.button("🔍 Route Complaint", type="primary", use_container_width=True)

# ── Prediction ────────────────────────────────────────────────────────────────
if route_clicked and complaint_text.strip():
    dept, confidence, alts, prio = get_prediction(complaint_text.strip(), main_model)
    dept_color  = DEPT_COLORS[dept]
    dept_emoji  = DEPT_EMOJI[dept]
    prio_class  = PRIORITY_CLASS[prio.priority_label]
    prio_emoji  = PRIORITY_EMOJI[prio.priority_label]

    # ── Result card ───────────────────────────────────────────────────────────
    signals_html = " ".join(
        f'<span class="signal-chip">⚑ {s}</span>'
        for s in prio.matched_signals
    ) if prio.matched_signals else '<span style="color:rgba(255,255,255,0.4);font-size:0.8rem">No high-priority signals detected</span>'

    conf_pct = int(confidence * 100)
    alt_bars_html = ""
    for a in alts:
        ap = int(a["probability"] * 100)
        alt_bars_html += f"""
        <div class="alt-row">
            <span>{DEPT_EMOJI.get(a['department'],'')} {a['department']}</span>
            <div class="alt-bar"><div class="alt-bar-fill" style="width:{ap}%"></div></div>
            <span style="min-width:36px;text-align:right">{ap}%</span>
        </div>"""

    st.markdown(f"""
    <div class="result-card">
        <div style="display:flex; align-items:center; gap:1rem; flex-wrap:wrap; margin-bottom:1.2rem">
            <div class="dept-badge" style="background:rgba(255,255,255,0.07); color:{dept_color}; border:1.5px solid {dept_color}40;">
                {dept_emoji} {dept}
            </div>
            <div class="priority-badge {prio_class}">
                {prio_emoji} {prio.priority_label} Priority — {prio.priority_score}/10
            </div>
        </div>

        <div class="metric-row">
            <div class="metric-box">
                <div class="metric-label">Confidence</div>
                <div class="metric-value" style="color:{dept_color}">{conf_pct}%</div>
                <div class="progress-bar-container">
                    <div class="progress-bar-fill" style="width:{conf_pct}%; background:linear-gradient(90deg,{dept_color}99,{dept_color});"></div>
                </div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Severity Score</div>
                <div class="metric-value">{prio.severity_score}/5</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Temporal Score</div>
                <div class="metric-value">{prio.temporal_score}/3</div>
            </div>
        </div>

        <div style="margin-top:1.2rem">
            <div style="color:rgba(255,255,255,0.45);font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem">
                Priority Signals Detected
            </div>
            {signals_html}
        </div>

        <div style="margin-top:1.4rem">
            <div style="color:rgba(255,255,255,0.45);font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.6rem">
                Top Alternatives
            </div>
            {alt_bars_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Expandable: Model details ─────────────────────────────────────────────
    with st.expander("📈 Model Details & Evaluation", expanded=False):
        metrics = load_metrics()
        if metrics:
            st.markdown("#### Macro F1 — Baseline vs Main Model")
            col1, col2, col3 = st.columns(3)
            col1.metric("BoW + NB (baseline)", f"{metrics['baseline']['macro_f1']:.4f}")
            col2.metric("TF-IDF + LR (main)",  f"{metrics['main']['macro_f1']:.4f}",
                        delta=f"+{metrics['main']['macro_f1'] - metrics['baseline']['macro_f1']:.4f}")
            col3.metric("Accuracy (main)",      f"{metrics['main']['accuracy']:.4f}",
                        help="Accuracy is misleading here — macro F1 is the right metric for imbalanced classes.")

            st.caption(
                "ℹ️ **Why macro F1?** If I always predicted 'Roads' (20% of corpus), "
                "accuracy would be 20% but macro F1 collapses to near 0 for minority classes. "
                "class_weight='balanced' penalizes minority class errors proportionally more."
            )

            st.markdown("#### Per-Class Metrics (TF-IDF + LR)")
            per_class = metrics["main"]["per_class"]
            rows = []
            for d, m in per_class.items():
                rows.append({
                    "Department": f"{DEPT_EMOJI.get(d,'')} {d}",
                    "Precision":  m["precision"],
                    "Recall":     m["recall"],
                    "F1-Score":   m["f1"],
                    "Support":    m["support"],
                })
            import pandas as pd
            df_table = pd.DataFrame(rows).set_index("Department")
            st.dataframe(
                df_table.style.background_gradient(subset=["F1-Score"], cmap="YlGn"),
                use_container_width=True,
            )

        cm_path = os.path.join(ROOT, "reports", "confusion_matrix.png")
        pcm_path = os.path.join(ROOT, "reports", "per_class_metrics.png")
        f1c_path = os.path.join(ROOT, "reports", "f1_comparison.png")

        if os.path.exists(cm_path) and os.path.exists(pcm_path):
            tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Per-Class Metrics", "F1 Comparison"])
            with tab1:
                st.image(cm_path, use_container_width=True)
            with tab2:
                st.image(pcm_path, use_container_width=True)
            with tab3:
                if os.path.exists(f1c_path):
                    st.image(f1c_path, use_container_width=True)

    # ── Design decision callout ────────────────────────────────────────────────
    with st.expander("🔍 Why Classical ML? Design Decisions", expanded=False):
        st.markdown("""
        **TF-IDF over BoW**
        - Bigrams capture phrases like *"pipe burst"*, *"no water"*, *"power cut"* as single features.
        - `sublinear_tf=True` applies log(1+tf) to dampen high-frequency terms without discarding them.

        **Logistic Regression over Naive Bayes**
        - LR outputs calibrated probabilities → confidence % you see above is meaningful.
        - NB assumes feature independence — wrong for civic text where "no water" is a phrase.

        **class_weight='balanced'**
        - Parks has only ~75 samples (9% of corpus). Without balancing, the model learns to ignore it.
        - Balanced weighting = loss proportional to `n_samples / (n_classes × n_per_class)`.

        **Priority Scorer: deliberately rule-based**
        - An auditor needs to explain *why* a complaint was flagged High priority.
        - A keyword list is inspectable and auditable — a neural network is not.

        **Hardest boundary: Roads vs Traffic**
        - Both departments share words: *signal*, *road*, *junction*, *accident*.
        - TF-IDF bigrams help slightly (*"traffic signal"* vs *"road signal"*), but this remains the worst-performing boundary.
        """)

elif route_clicked and not complaint_text.strip():
    st.warning("Please enter a complaint text before routing.")

else:
    # Default state — instruction card
    st.markdown("""
    <div style="
        background: rgba(255,255,255,0.03);
        border: 1px dashed rgba(255,255,255,0.12);
        border-radius: 12px;
        padding: 2.5rem;
        text-align: center;
        color: rgba(255,255,255,0.5);
        margin-top: 1rem;
    ">
        <div style="font-size:2.5rem;margin-bottom:1rem">✍️</div>
        <div style="font-size:1rem;font-weight:500;color:rgba(255,255,255,0.7);margin-bottom:0.5rem">
            Paste a civic complaint above and click Route Complaint
        </div>
        <div style="font-size:0.85rem">
            Or pick one of the 6 example complaints from the sidebar →
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:rgba(255,255,255,0.3);font-size:0.8rem'>"
    "Classical NLP · TF-IDF + scikit-learn · spaCy lemmatization · No LLMs"
    "</div>",
    unsafe_allow_html=True,
)
