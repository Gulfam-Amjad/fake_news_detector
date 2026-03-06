# ============================================================
# src/preprocess.py
#
# PURPOSE:
#   This module handles ALL data loading and text preprocessing.
#   Raw text from news articles is messy — full of HTML tags,
#   punctuation, stopwords ("the", "is", "a"), and different
#   capitalizations. We clean all of that here so the model
#   only learns from meaningful words.
#
# PIPELINE:
#   Load CSVs → Merge → Label → Clean Text → Save Processed Data
# ============================================================

import os
import re                          # Regular expressions for pattern matching
import pandas as pd                # DataFrame manipulation
import nltk                        # NLP toolkit
from nltk.corpus import stopwords  # Common words to remove (the, is, a...)
from nltk.stem import PorterStemmer, WordNetLemmatizer  # Word normalization

# ── Download required NLTK data (runs only first time) ─────────────────────
# These are like "vocabulary packs" that NLTK needs.
# 'stopwords' → list of common words to ignore
# 'wordnet'   → dictionary for lemmatization
# 'punkt'     → sentence/word tokenizer
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('punkt',     quiet=True)


# ── Constants ────────────────────────────────────────────────────────────────
RAW_DIR       = os.path.join("data", "raw")       # Where True.csv / Fake.csv live
PROCESSED_DIR = os.path.join("data", "processed") # Where we save cleaned data
OUTPUT_FILE   = os.path.join(PROCESSED_DIR, "news_cleaned.csv")

# Build the English stopword set (we'll remove these from articles)
# Example stopwords: "the", "and", "is", "in", "it", "of", "to"
STOP_WORDS = set(stopwords.words("english"))

# Initialize stemmer and lemmatizer:
#   Stemmer   → "running" becomes "run" (crude, fast — chops suffix)
#   Lemmatizer→ "running" becomes "run" (smarter, uses dictionary)
stemmer    = PorterStemmer()
lemmatizer = WordNetLemmatizer()


# ── Step 1: Load Raw Data ────────────────────────────────────────────────────

def load_data(raw_dir: str = RAW_DIR) -> pd.DataFrame:
    """
    Load and combine True.csv and Fake.csv into one DataFrame.

    Each file gets a label column:
        1 = REAL news  (from True.csv)
        0 = FAKE news  (from Fake.csv)

    Returns:
        pd.DataFrame with columns: title, text, subject, date, label
    """
    true_path = os.path.join(raw_dir, "True.csv")
    fake_path = os.path.join(raw_dir, "Fake.csv")

    # Load CSVs into DataFrames
    print(f"[INFO] Loading real news from: {true_path}")
    df_real = pd.read_csv(true_path)
    df_real["label"] = 1             # 1 = REAL

    print(f"[INFO] Loading fake news from: {fake_path}")
    df_fake = pd.read_csv(fake_path)
    df_fake["label"] = 0             # 0 = FAKE

    # Combine both DataFrames into one
    df = pd.concat([df_real, df_fake], axis=0, ignore_index=True)

    print(f"[INFO] Dataset loaded: {len(df)} total articles")
    print(f"         Real: {df['label'].sum()} | Fake: {(df['label'] == 0).sum()}")

    return df


# ── Step 2: Combine Title + Text ─────────────────────────────────────────────

def combine_title_and_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the 'title' and 'text' columns into a single 'content' column.

    WHY? The title alone can carry strong signals (fake news often has
    sensational headlines). Combining both gives the model more context.

    Example:
        title = "BREAKING: Trump signs deal"
        text  = "President Trump today announced..."
        content = "BREAKING: Trump signs deal President Trump today announced..."
    """
    # Fill NaN values so string concatenation doesn't fail
    df["title"] = df["title"].fillna("")
    df["text"]  = df["text"].fillna("")

    # Combine with a space between title and body
    df["content"] = df["title"] + " " + df["text"]

    print("[INFO] Combined 'title' and 'text' into 'content' column.")
    return df


# ── Step 3: Clean Text ───────────────────────────────────────────────────────

def clean_text(text: str, use_lemmatization: bool = True) -> str:
    """
    Clean and normalize a single piece of text.

    Steps:
        1. Lowercase everything         → "Trump" == "trump"
        2. Remove URLs                  → http://... → ""
        3. Remove HTML tags             → <b>text</b> → "text"
        4. Remove non-alphabet chars    → "hello! 123" → "hello"
        5. Tokenize (split into words)  → "hello world" → ["hello", "world"]
        6. Remove stopwords             → ["the", "cat"] → ["cat"]
        7. Lemmatize or Stem words      → ["running"] → ["run"]
        8. Join back into a string      → ["run", "fast"] → "run fast"

    Args:
        text (str): Raw article text
        use_lemmatization (bool): If True, use lemmatizer (slower but smarter).
                                  If False, use stemmer (faster but cruder).
    Returns:
        str: Cleaned text
    """
    # Guard against non-string inputs (NaN, None, numbers)
    if not isinstance(text, str):
        return ""

    # Step 1: Lowercase
    text = text.lower()

    # Step 2: Remove URLs (http://..., https://..., www....)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Step 3: Remove HTML tags (e.g., <p>, <br/>, <strong>)
    text = re.sub(r"<.*?>", "", text)

    # Step 4: Keep only letters and spaces (remove numbers, punctuation, symbols)
    # [^a-z\s] means: "anything that is NOT a letter or whitespace" → remove it
    text = re.sub(r"[^a-z\s]", "", text)

    # Step 5: Split text into individual words (tokenize)
    tokens = text.split()

    # Step 6 & 7: Remove stopwords AND normalize each word
    cleaned_tokens = []
    for word in tokens:
        if word not in STOP_WORDS and len(word) > 2:  # skip tiny words too
            if use_lemmatization:
                # Lemmatize: uses dictionary — "better" → "good", "running" → "run"
                word = lemmatizer.lemmatize(word)
            else:
                # Stem: crude chop — "running" → "run", "fairly" → "fairli"
                word = stemmer.stem(word)
            cleaned_tokens.append(word)

    # Step 8: Join tokens back into a single string
    return " ".join(cleaned_tokens)


def apply_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply clean_text() to every row's 'content' column.
    Shows a progress counter since this can take 30–60 seconds.
    """
    print("[INFO] Cleaning text (this may take ~30 seconds)...")

    # pandas .apply() runs clean_text() on every row
    df["content_clean"] = df["content"].apply(clean_text)

    # Drop rows that ended up empty after cleaning
    before = len(df)
    df = df[df["content_clean"].str.strip() != ""]
    after  = len(df)
    print(f"[INFO] Removed {before - after} empty rows after cleaning.")
    print(f"[INFO] Final dataset size: {after} articles")

    return df


# ── Step 4: Save Processed Data ──────────────────────────────────────────────

def save_processed(df: pd.DataFrame, output_path: str = OUTPUT_FILE) -> None:
    """
    Save the cleaned DataFrame to a CSV so we don't have to re-clean
    every time we run experiments.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Saved processed data to: {output_path}")


def load_processed(output_path: str = OUTPUT_FILE) -> pd.DataFrame:
    """
    Load pre-processed data (skip cleaning step if already done).
    """
    if not os.path.exists(output_path):
        raise FileNotFoundError(
            f"Processed file not found at {output_path}.\n"
            "Run preprocess.py first to generate it."
        )
    df = pd.read_csv(output_path)
    print(f"[INFO] Loaded processed data: {len(df)} articles from {output_path}")
    return df


# ── Step 5: Shuffle Data ─────────────────────────────────────────────────────

def shuffle_data(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Randomly shuffle the DataFrame rows.

    WHY? Our data is currently sorted — all real news first, then all fake.
    If we split 80/20 without shuffling, our training set might be mostly
    real news and test set mostly fake. Shuffling ensures both sets have
    a balanced mix.

    seed=42 makes the shuffle reproducible (same result every run).
    """
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"[INFO] Data shuffled (seed={seed}).")
    return df


# ── Main Execution ────────────────────────────────────────────────────────────

def run_preprocessing() -> pd.DataFrame:
    """
    Orchestrates the full preprocessing pipeline:
    Load → Combine → Clean → Shuffle → Save
    """
    df = load_data()
    df = combine_title_and_text(df)
    df = apply_cleaning(df)
    df = shuffle_data(df)
    save_processed(df)
    return df


# ── Script Entry Point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    # When you run: python src/preprocess.py
    # This will process all raw data and save it to data/processed/
    df = run_preprocessing()

    # Quick preview of the cleaned data
    print("\n[PREVIEW] First 3 rows of cleaned content:")
    print(df[["label", "content_clean"]].head(3).to_string())
