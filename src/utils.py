# ============================================================
# src/utils.py
#
# PURPOSE:
#   Reusable helper functions for:
#     - Exploratory Data Analysis (EDA)
#     - Visualizations (word clouds, distributions)
#     - Feature importance (which words matter most?)
#     - Text statistics (article length analysis)
#
# These functions are called from train.py and notebooks.
# Keeping them here avoids code duplication.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud


# ── Dataset Statistics ────────────────────────────────────────────────────────

def print_dataset_stats(df: pd.DataFrame) -> None:
    """
    Print a summary of the dataset: counts, class balance, text lengths.

    Understanding your data BEFORE training is crucial.
    Class imbalance (e.g., 90% REAL vs 10% FAKE) would bias the model.
    """
    total = len(df)
    real  = (df["label"] == 1).sum()
    fake  = (df["label"] == 0).sum()

    print("=" * 50)
    print("📊 Dataset Statistics")
    print("=" * 50)
    print(f"  Total articles : {total:,}")
    print(f"  Real (label=1) : {real:,}  ({real/total*100:.1f}%)")
    print(f"  Fake (label=0) : {fake:,}  ({fake/total*100:.1f}%)")

    # Text length statistics
    df["text_len"] = df["content_clean"].apply(lambda x: len(str(x).split()))

    print(f"\n  Article length (in words after cleaning):")
    print(f"    Mean   : {df['text_len'].mean():.0f} words")
    print(f"    Median : {df['text_len'].median():.0f} words")
    print(f"    Min    : {df['text_len'].min()} words")
    print(f"    Max    : {df['text_len'].max()} words")
    print("=" * 50)


# ── Word Cloud ────────────────────────────────────────────────────────────────

def plot_wordcloud(df: pd.DataFrame, label: int, title: str, save_dir: str = "models") -> None:
    """
    Generate and save a word cloud for REAL or FAKE articles.

    Word cloud = visualization where word SIZE = frequency.
    Common words appear BIG. Rare words appear small.

    Great for quick intuition — fake news might show words like
    "deep state", "breaking", "exposed". Real news might show
    "said", "according", "government", "percent".

    Args:
        df    : Full DataFrame with 'content_clean' and 'label'
        label : 1 for REAL, 0 for FAKE
        title : Chart title
        save_dir : Where to save the image
    """
    # Get all text for this label (join all articles into one big string)
    subset_text = " ".join(
        df[df["label"] == label]["content_clean"].dropna().tolist()
    )

    if not subset_text.strip():
        print(f"[WARNING] No text found for label={label}. Skipping word cloud.")
        return

    # Generate word cloud
    # max_words limits to top N words (less visual clutter)
    # background_color = white is most readable
    wc = WordCloud(
        max_words        = 200,
        background_color = "white",
        colormap         = "Blues" if label == 1 else "Reds",
        width            = 800,
        height           = 400,
    ).generate(subset_text)

    # Plot
    plt.figure(figsize=(12, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")  # No axis needed for word clouds
    plt.title(title, fontsize=16, fontweight="bold")
    plt.tight_layout()

    # Save
    os.makedirs(save_dir, exist_ok=True)
    filename = title.replace(" ", "_").lower() + ".png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Word cloud saved: {save_path}")


# ── Top Words Per Class ───────────────────────────────────────────────────────

def get_top_words(df: pd.DataFrame, label: int, n: int = 20) -> list:
    """
    Find the most common words in REAL or FAKE articles.

    Uses Python's Counter (a dict-like class that counts occurrences).

    Args:
        df    : DataFrame with 'content_clean' and 'label'
        label : 1=REAL, 0=FAKE
        n     : How many top words to return

    Returns:
        List of (word, count) tuples, sorted by count descending
    """
    # Get all words from articles of this class
    all_words = " ".join(
        df[df["label"] == label]["content_clean"].dropna()
    ).split()

    # Count word frequencies
    word_counts = Counter(all_words)

    # Return top N
    return word_counts.most_common(n)


def plot_top_words(df: pd.DataFrame, n: int = 15, save_dir: str = "models") -> None:
    """
    Plot side-by-side bar charts comparing top words in REAL vs FAKE news.

    This is useful to understand:
    - What vocabulary patterns separate the two classes
    - If there are any suspicious patterns (e.g., "BREAKING" in fake)
    """
    top_real = get_top_words(df, label=1, n=n)
    top_fake = get_top_words(df, label=0, n=n)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot REAL news top words
    words_r, counts_r = zip(*top_real)
    axes[0].barh(list(words_r)[::-1], list(counts_r)[::-1], color="#3b82f6")
    axes[0].set_title(f"Top {n} Words — REAL News", fontsize=13)
    axes[0].set_xlabel("Frequency")

    # Plot FAKE news top words
    words_f, counts_f = zip(*top_fake)
    axes[1].barh(list(words_f)[::-1], list(counts_f)[::-1], color="#ef4444")
    axes[1].set_title(f"Top {n} Words — FAKE News", fontsize=13)
    axes[1].set_xlabel("Frequency")

    plt.suptitle("Most Common Words by Category", fontsize=15, fontweight="bold")
    plt.tight_layout()

    save_path = os.path.join(save_dir, "top_words_comparison.png")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Top words chart saved: {save_path}")


# ── Article Length Distribution ───────────────────────────────────────────────

def plot_length_distribution(df: pd.DataFrame, save_dir: str = "models") -> None:
    """
    Plot the distribution of article lengths (in words) for REAL vs FAKE.

    Hypothesis: Fake news might be shorter (emotional clickbait) or
    longer (verbose conspiracy theories). Visualizing this helps us
    understand if length is a useful feature.
    """
    df = df.copy()
    df["word_count"] = df["content_clean"].apply(lambda x: len(str(x).split()))

    plt.figure(figsize=(10, 5))

    # Plot histograms for each class
    # alpha makes them semi-transparent so they overlap nicely
    plt.hist(
        df[df["label"] == 1]["word_count"],
        bins=50, alpha=0.6, color="#3b82f6", label="REAL"
    )
    plt.hist(
        df[df["label"] == 0]["word_count"],
        bins=50, alpha=0.6, color="#ef4444", label="FAKE"
    )

    plt.title("Article Length Distribution — REAL vs FAKE", fontsize=14)
    plt.xlabel("Word Count (after cleaning)")
    plt.ylabel("Number of Articles")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(save_dir, "length_distribution.png")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[INFO] Length distribution chart saved: {save_path}")


# ── Feature Importance (Logistic Regression) ──────────────────────────────────

def plot_feature_importance(pipeline, n: int = 20, save_dir: str = "models") -> None:
    """
    Show which words the model considers most important for each class.

    This only works for Logistic Regression (has .coef_ attribute).

    HOW IT WORKS:
        Logistic Regression assigns a WEIGHT (coefficient) to each word.
        - High positive weight → word strongly predicts REAL
        - High negative weight → word strongly predicts FAKE

    This is called "model interpretability" — you can explain WHY
    the model made a certain decision.

    Args:
        pipeline : Trained sklearn Pipeline (TF-IDF + LogisticRegression)
        n        : How many top features to show per class
    """
    # Check if the classifier supports feature importance
    clf = pipeline.named_steps.get("clf")
    if not hasattr(clf, "coef_"):
        print("[INFO] Feature importance only supported for Logistic Regression. Skipping.")
        return

    # Get feature names from the TF-IDF vectorizer
    tfidf      = pipeline.named_steps["tfidf"]
    feature_names = np.array(tfidf.get_feature_names_out())

    # Get model coefficients (one per feature)
    # clf.coef_ is shape (1, n_features) for binary classification
    coef = clf.coef_[0]

    # Top N words that push towards REAL (highest positive coefficients)
    top_real_idx  = np.argsort(coef)[-n:][::-1]
    top_real_words = feature_names[top_real_idx]
    top_real_coefs = coef[top_real_idx]

    # Top N words that push towards FAKE (highest negative coefficients)
    top_fake_idx  = np.argsort(coef)[:n]
    top_fake_words = feature_names[top_fake_idx]
    top_fake_coefs = coef[top_fake_idx]

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].barh(range(n), top_real_coefs[::-1], color="#3b82f6")
    axes[0].set_yticks(range(n))
    axes[0].set_yticklabels(top_real_words[::-1])
    axes[0].set_title(f"Top {n} Words → REAL (positive weights)")
    axes[0].set_xlabel("Coefficient Weight")

    axes[1].barh(range(n), np.abs(top_fake_coefs[::-1]), color="#ef4444")
    axes[1].set_yticks(range(n))
    axes[1].set_yticklabels(top_fake_words[::-1])
    axes[1].set_title(f"Top {n} Words → FAKE (negative weights)")
    axes[1].set_xlabel("Absolute Coefficient Weight")

    plt.suptitle("Logistic Regression — Feature Importance", fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_path = os.path.join(save_dir, "feature_importance.png")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Feature importance chart saved: {save_path}")
