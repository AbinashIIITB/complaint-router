# Civic Complaint Router

> **Classical NLP + ML — Deliberately no LLMs.**

A text classification system that routes free-text Indian municipal complaints to the correct department (Roads, Water, Electricity, Sanitation, Traffic, Parks) and assigns an explainable, rule-based priority score.

**Live Demo →** [complaint-router.streamlit.app](https://complaint-router.streamlit.app)

---

## The Problem

Municipal grievance portals in India suffer from systematic misrouting — a broken streetlight ends up in the Sanitation queue and waits weeks. This project solves that with a production-ready, interpretable classical ML pipeline.

**Input**: "The signal near Rasulgarh has been blinking yellow for three days, it's chaos at peak hour."
**Output**: 🚦 **Traffic** | Confidence 94% | 🔴 **High Priority (8/10)** | Signals: `chaos`, `days`

---

## Architecture

```
Raw Complaint Text
       │
       ▼
┌─────────────────────────────────┐
│   Preprocessing (spaCy)         │
│   lowercase → lemmatize →       │
│   stopword removal              │
└────────────────┬────────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
┌──────────────┐   ┌──────────────────────┐
│  Classifier  │   │   Priority Scorer     │
│  TF-IDF      │   │   Rule-based keyword  │
│  (bigrams)   │   │   severity + temporal │
│  + LogReg    │   │   weighting           │
└──────┬───────┘   └──────────┬───────────┘
       │                       │
       └───────────┬───────────┘
                   ▼
        Department + Confidence
        + Priority Score (1–10)
        + Matched Signals
```

---

## Numbers

| Metric | Value |
|---|---|
| Dataset size | 800 labelled complaints |
| Classes | 6 departments |
| Class balance | Roads 20.6% → Parks 9.4% |
| Baseline macro F1 | 0.9494 (BoW + Naive Bayes) |
| Main model macro F1 | 0.9684 (TF-IDF + LR, 5-fold GridSearchCV) |
| F1 lift over baseline | +0.0190 |
| Main model accuracy | 0.9750 (on 160-sample held-out test set) |
| Worst class | Parks (F1 0.9032) / Traffic–Roads boundary |

---

## Tech Stack

| Layer | Tool |
|---|---|
| Text preprocessing | spaCy (`en_core_web_sm`) |
| Feature extraction | `TfidfVectorizer` (bigrams, sublinear_tf) |
| Classifier | `LogisticRegression` (class_weight=balanced) |
| Baseline | `CountVectorizer` + `MultinomialNB` |
| Tuning | `GridSearchCV` (5-fold stratified CV) |
| Priority scoring | Rule-based keyword weighting |
| UI | Streamlit |
| Persistence | joblib |

---

## Design Decisions

### TF-IDF + bigrams over Bag of Words
Bigrams capture multi-word phrases as features: *"pipe burst"*, *"no water"*, *"power cut"*, *"traffic signal"*. `sublinear_tf=True` applies log(1+tf) to dampen high-frequency terms.

### Logistic Regression over Naive Bayes
LR outputs calibrated probabilities (the confidence % you see in the UI is meaningful), and doesn't assume feature independence — which is wrong for civic text.

### `class_weight='balanced'`
Parks has only ~75 samples. Without balancing the loss, the model learns to ignore it. Balanced weighting = `n_samples / (n_classes × n_per_class)` per class.

### Why macro F1, not accuracy?
If I always predicted "Roads" (20% of corpus), accuracy = 20% but macro F1 collapses to near 0 for minority classes. Macro F1 weights every class equally.

### Priority scorer: deliberately rule-based
An auditor in a civic context needs to explain *why* a complaint was flagged High priority. A keyword list is inspectable. A neural net is not.

### Hardest boundary: Roads vs Traffic
Both share vocabulary: *signal*, *road*, *junction*, *accident*. TF-IDF bigrams help slightly (*"traffic signal"* vs generic *"road"*), but this remains the worst-performing class boundary.

---

## Project Structure

```
complaint-router/
├── data/
│   ├── raw/complaints_raw.csv          # 800 labelled complaints
│   └── processed/complaints_clean.csv  # after spaCy preprocessing
├── models/
│   ├── baseline_bow_nb.pkl             # BoW + NB pipeline
│   └── tfidf_logreg.pkl                # TF-IDF + LR pipeline
├── reports/
│   ├── confusion_matrix.png
│   ├── per_class_metrics.png
│   ├── f1_comparison.png
│   └── metrics_summary.md
├── src/
│   ├── data_gen.py      # Synthetic corpus generator
│   ├── preprocess.py    # spaCy lemmatization pipeline
│   ├── train.py         # Trains + saves both models
│   ├── evaluate.py      # Plots + metrics
│   ├── priority.py      # Rule-based priority scorer
│   └── predict.py       # Inference wrapper + smoke tests
├── app/
│   └── streamlit_app.py # Streamlit UI
└── requirements.txt
```

---

## How to Run Locally

```bash
# 1. Clone and install
git clone https://github.com/<your-username>/complaint-router.git
cd complaint-router
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Build the pipeline
python src/data_gen.py        # generate corpus
python src/preprocess.py      # lemmatize + clean
python src/train.py           # train both models
python src/evaluate.py        # generate plots

# 3. Run Streamlit
streamlit run app/streamlit_app.py
```

### Smoke tests

```bash
python src/predict.py --test   # 6 routing smoke tests, one per department
python src/evaluate.py --assert-f1 0.82   # fails if main model < 0.82 macro F1
```

---

## Deployment (Streamlit Community Cloud)

1. Push this repo to GitHub (public)
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Set **Main file path**: `app/streamlit_app.py`
4. Click Deploy

> **Note**: The trained `.pkl` model files must be committed to the repo for Streamlit Cloud to load them. They are ~1–2 MB each.

---

## Dataset Construction

The 800-complaint corpus was generated with a template-based Python script (`src/data_gen.py`) using:
- ~28 templates per department (6 departments)
- Odisha place names: Rasulgarh, Unit-4, Chandrasekharpur, Patia, Cuttack, Puri, etc.
- Temporal markers: *"for three days"*, *"since last week"*, *"for over a week"*
- Urgency phrases: *"children are at risk"*, *"public safety hazard"*, *"please take immediate action"*

A stratified sample of 80 complaints (10 per class × Parks 8) was manually reviewed. ~3% mislabelling was found in the Roads/Traffic boundary cases, which were corrected.
