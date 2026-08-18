"""
preprocess.py
-------------
spaCy-based preprocessing pipeline.

Steps:
  1. Lowercase
  2. Remove digits, punctuation, and extra whitespace (regex)
  3. Tokenise with spaCy (en_core_web_sm)
  4. Remove stopwords — spaCy defaults + custom civic words
     (negations "no / not / never" are kept intentionally)
  5. Lemmatize each surviving token

Run:
    python src/preprocess.py
Input:  data/raw/complaints_raw.csv
Output: data/processed/complaints_clean.csv
"""

import re
import os
import pandas as pd
import spacy

# ---------------------------------------------------------------------------
# Extra stopwords on top of spaCy defaults.
# These are high-frequency civic-portal words that carry no
# discriminative signal (e.g. every complaint starts with "sir / kindly").
# ---------------------------------------------------------------------------
CUSTOM_STOPWORDS = {
    "complaint", "complain", "sir", "madam", "kindly", "please",
    "request", "department", "authority", "officer", "office",
    "area", "near", "local", "respective", "concerned",
    "action", "immediate", "look", "matter",
}

# Negations we always keep, even if spaCy marks them as stopwords.
KEEP_NEGATIONS = {"no", "not", "never"}


def load_spacy():
    """Load the spaCy model; prints a helpful message if it is missing."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        print("spaCy model not found. Run:  python -m spacy download en_core_web_sm")
        raise


def clean_text(text: str) -> str:
    """Lowercase; strip digits, punctuation, and extra whitespace."""
    text = text.lower()
    text = re.sub(r"\d+", " ", text)           # remove numbers
    text = re.sub(r"[^\w\s]", " ", text)       # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()   # collapse whitespace
    return text


def _keep_token(tok, all_stopwords: set) -> bool:
    """
    Return True if this spaCy token should be kept.

    Rules (applied in order):
    - Always keep negations ("no", "not", "never") — "no water" is a
      critical signal, not noise.
    - Drop spaCy stopwords.
    - Drop custom civic stopwords.
    - Drop whitespace and punctuation tokens.
    - Drop single-character tokens.
    """
    lemma = tok.lemma_

    # Always keep negations
    if lemma in KEEP_NEGATIONS or tok.text in KEEP_NEGATIONS:
        return True

    # Drop stopwords and custom words
    if tok.is_stop or tok.text in all_stopwords:
        return False

    # Drop whitespace / punctuation
    if tok.is_space or tok.is_punct:
        return False

    # Drop single-character tokens (noise)
    if len(lemma) < 2:
        return False

    return True


def preprocess_dataframe(df: pd.DataFrame, nlp, all_stopwords: set) -> pd.DataFrame:
    """
    Apply the full preprocessing pipeline to a DataFrame.

    Adds two columns:
    - text_cleaned : after regex cleaning
    - text_clean   : after spaCy lemmatization (used for training)
    """
    print("Cleaning text (regex)...")
    df["text_cleaned"] = df["text"].apply(clean_text)

    print("Lemmatizing with spaCy (this may take ~30 s)...")
    texts = df["text_cleaned"].tolist()
    lemmatized = []

    # nlp.pipe processes texts in batches — much faster than calling nlp() one by one.
    # We disable 'ner' and 'parser' since we only need tokenisation + lemmatization.
    for doc in nlp.pipe(texts, batch_size=64, disable=["ner", "parser"]):
        tokens = [tok.lemma_ for tok in doc if _keep_token(tok, all_stopwords)]
        lemmatized.append(" ".join(tokens))

    df["text_clean"] = lemmatized
    return df


def main():
    os.makedirs("data/processed", exist_ok=True)
    raw_path = "data/raw/complaints_raw.csv"
    out_path = "data/processed/complaints_clean.csv"

    print(f"Loading {raw_path}...")
    df = pd.read_csv(raw_path)
    print(f"  {len(df)} rows loaded")

    nlp = load_spacy()
    all_stopwords = nlp.Defaults.stop_words | CUSTOM_STOPWORDS

    df = preprocess_dataframe(df, nlp, all_stopwords)

    # Keep both raw text and cleaned text so we can compare them later
    df_out = df[["id", "text", "text_clean", "department", "source"]]
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved {len(df_out)} rows to {out_path}")

    # Show a few before/after examples
    print("\nSample comparisons (RAW → CLEAN):")
    sample = df_out.sample(3, random_state=1)
    for _, row in sample.iterrows():
        print(f"\n  [RAW]   {row['text'][:100]}")
        print(f"  [CLEAN] {row['text_clean'][:100]}")
        print(f"  Dept:   {row['department']}")


if __name__ == "__main__":
    main()
