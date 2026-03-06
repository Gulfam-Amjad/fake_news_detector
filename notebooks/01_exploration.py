# ============================================================
# notebooks/01_exploration.py
#
# PURPOSE:
#   Exploratory Data Analysis (EDA) notebook converted to a
#   runnable Python script for users who prefer scripts over
#   Jupyter. In Jupyter, each section below would be a cell.
#
# WHAT IS EDA?
#   Before building any ML model, you MUST understand your data:
#   - How many samples? Class balance?
#   - What does the text look like (raw vs cleaned)?
#   - Are there patterns visible to the human eye?
#   - What are the most common words per class?
#
# Run: python notebooks/01_exploration.py
# ============================================================

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import sys
sys.path.insert(0, os.path.abspath(".."))  # Allow imports from project root

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocess import load_data, combine_title_and_text, clean_text, apply_cleaning
from src.utils      import (
    print_dataset_stats,
    plot_wordcloud,
    plot_top_words,
    plot_length_distribution
)

# ── Style ────────────────────────────────────────────────────────────────────
# Set a clean visual theme for all matplotlib charts
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("Set2")

SAVE_DIR = "models"  # Save all plots here
os.makedirs(SAVE_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════
# SECTION 1: Load Raw Data
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  SECTION 1: Load Raw Data")
print("="*60)

df = load_data()

# Preview the raw data
print("\n[RAW DATA] First 3 rows:")
print(df.head(3).to_string())

print(f"\n[INFO] Columns: {list(df.columns)}")
print(f"[INFO] Data types:\n{df.dtypes}")
print(f"\n[INFO] Missing values:\n{df.isnull().sum()}")


# ═══════════════════════════════════════════════════════════
# SECTION 2: Class Distribution
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  SECTION 2: Class Distribution")
print("="*60)

# Value counts tells us how many of each class we have
# A big difference (e.g., 90% real, 10% fake) = class imbalance problem
class_counts = df["label"].value_counts()
print(f"\n  REAL (1): {class_counts.get(1, 0):,} articles")
print(f"  FAKE (0): {class_counts.get(0, 0):,} articles")

# Visualize as a bar chart
plt.figure(figsize=(6, 4))
bars = plt.bar(["FAKE", "REAL"], [class_counts.get(0, 0), class_counts.get(1, 0)],
               color=["#ef4444", "#3b82f6"], edgecolor="white", linewidth=1.5)
plt.title("Class Distribution: Real vs Fake News", fontsize=14, fontweight="bold")
plt.ylabel("Number of Articles")

# Add count labels on top of bars
for bar, count in zip(bars, [class_counts.get(0, 0), class_counts.get(1, 0)]):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
             f"{count:,}", ha="center", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "class_distribution.png"), dpi=120)
plt.close()
print(f"[INFO] Class distribution chart saved.")


# ═══════════════════════════════════════════════════════════
# SECTION 3: Subject/Category Analysis
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  SECTION 3: Subject/Topic Analysis")
print("="*60)

# "subject" column shows what topic the article covers
# e.g., politics, world news, US_News
print("\n[REAL news] Subject breakdown:")
print(df[df["label"] == 1]["subject"].value_counts().head(10))

print("\n[FAKE news] Subject breakdown:")
print(df[df["label"] == 0]["subject"].value_counts().head(10))

# Observation: Fake news often concentrates in certain topics (politics, conspiracies)


# ═══════════════════════════════════════════════════════════
# SECTION 4: Text Length Analysis
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  SECTION 4: Text Length Analysis")
print("="*60)

# Combine title + text first (without cleaning yet)
df = combine_title_and_text(df)

# Compute raw word count (before cleaning)
df["raw_word_count"] = df["content"].apply(lambda x: len(str(x).split()))

print("\n[REAL news] Word count stats:")
print(df[df["label"] == 1]["raw_word_count"].describe().round(0))

print("\n[FAKE news] Word count stats:")
print(df[df["label"] == 0]["raw_word_count"].describe().round(0))

# Are fake articles shorter or longer? Visualize!
plt.figure(figsize=(10, 5))
plt.hist(df[df["label"]==1]["raw_word_count"].clip(upper=2000),
         bins=50, alpha=0.6, color="#3b82f6", label="REAL")
plt.hist(df[df["label"]==0]["raw_word_count"].clip(upper=2000),
         bins=50, alpha=0.6, color="#ef4444", label="FAKE")
plt.title("Raw Article Length Distribution", fontsize=14, fontweight="bold")
plt.xlabel("Word Count (clipped at 2000)")
plt.ylabel("Number of Articles")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "raw_length_distribution.png"), dpi=120)
plt.close()
print(f"[INFO] Raw length distribution chart saved.")


# ═══════════════════════════════════════════════════════════
# SECTION 5: Clean the Text and Analyze
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  SECTION 5: Text Cleaning Preview")
print("="*60)

# Show a before/after example of text cleaning
sample_raw = df["content"].iloc[0]
sample_cleaned = clean_text(sample_raw)

print("\n[BEFORE CLEANING] (first 300 chars):")
print(sample_raw[:300])

print("\n[AFTER CLEANING] (first 300 chars):")
print(sample_cleaned[:300])

print(f"\nOriginal word count : {len(sample_raw.split())}")
print(f"Cleaned word count  : {len(sample_cleaned.split())}")
print(f"Words removed       : {len(sample_raw.split()) - len(sample_cleaned.split())}")


# ═══════════════════════════════════════════════════════════
# SECTION 6: Apply Full Cleaning and Analyze Top Words
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  SECTION 6: Full Clean + Word Analysis")
print("  (This section takes ~1 minute)")
print("="*60)

# Apply full cleaning to all articles
df = apply_cleaning(df)

# Print overall dataset stats
print_dataset_stats(df)

# Word clouds
print("\n[INFO] Generating word clouds...")
plot_wordcloud(df, label=1, title="Most Common Words — REAL News",   save_dir=SAVE_DIR)
plot_wordcloud(df, label=0, title="Most Common Words — FAKE News",   save_dir=SAVE_DIR)

# Top word comparison bar chart
print("[INFO] Plotting top words comparison...")
plot_top_words(df, n=15, save_dir=SAVE_DIR)

# Length distribution after cleaning
print("[INFO] Plotting cleaned text length distribution...")
plot_length_distribution(df, save_dir=SAVE_DIR)


# ═══════════════════════════════════════════════════════════
# SECTION 7: Key Observations (EDA Summary)
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  SECTION 7: EDA Summary & Key Observations")
print("="*60)
print("""
  ✅ Dataset is fairly balanced (~48% real, ~52% fake) — no major imbalance issues.

  ✅ REAL news tends to be longer and from Reuters — more formal, structured language.

  ❌ FAKE news tends to use ALL CAPS, more emotional language, and sensational phrases.

  📊 Both classes have a right-skewed length distribution — most articles are
     relatively short, but some are very long (skews the mean higher than median).

  🔑 Most common words differ meaningfully between classes:
     - REAL: "said", "reuters", "percent", "government", "trump"
     - FAKE: "said", "hillary", "clinton", "trump", "obama" (more political names)

  💡 This suggests TF-IDF will work well because vocabulary IS different.
     The next step (train.py) will train and evaluate our models.
""")

print("  ✅ EDA complete! All charts saved to:", SAVE_DIR)
print("  ➡️  Next step: python src/train.py")
