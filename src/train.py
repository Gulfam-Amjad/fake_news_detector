# ============================================================
# src/train.py
#
# PURPOSE:
#   Train multiple ML models to classify news as REAL or FAKE.
#   Uses sklearn Pipelines so that vectorization + model are
#   always bundled together — no risk of forgetting to transform
#   new data at prediction time.
#
# FLOW:
#   Load Processed Data
#       → Train/Test Split
#       → Build Pipelines (TF-IDF + Model)
#       → Train All Models
#       → Evaluate (Accuracy, F1, Confusion Matrix)
#       → Save Best Model
#
# TF-IDF EXPLAINED:
#   Instead of counting raw word occurrences, TF-IDF gives higher
#   weight to words that are frequent in ONE document but rare
#   across ALL documents. This helps identify "signature" words.
#   e.g., "election" in a fake article matters more than "the".
# ============================================================

import os
import joblib                          # Save/load sklearn models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection   import train_test_split
from sklearn.pipeline          import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model      import LogisticRegression, PassiveAggressiveClassifier
from sklearn.ensemble          import RandomForestClassifier
from sklearn.metrics           import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score
)

# Import our preprocessing module
from src.preprocess import load_processed

# ── Constants ────────────────────────────────────────────────────────────────
PROCESSED_FILE = os.path.join("data", "processed", "news_cleaned.csv")
MODELS_DIR     = "models"
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
TEST_SIZE      = 0.20    # 80% train, 20% test
RANDOM_STATE   = 42      # For reproducibility


# ── Step 1: Prepare Features and Labels ──────────────────────────────────────

def prepare_data(df: pd.DataFrame):
    """
    Split DataFrame into features (X) and labels (y), then into
    train and test sets.

    X = the cleaned article text (input to model)
    y = label: 1=REAL, 0=FAKE (what model must predict)

    train_test_split:
        - test_size=0.2 means 20% of data goes to testing
        - stratify=y ensures BOTH splits have same ratio of real/fake
          (very important if dataset is slightly imbalanced)
    """
    X = df["content_clean"]   # Feature: cleaned text
    y = df["label"]           # Target: 0 or 1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y              # Keep class balance in both splits
    )

    print(f"[INFO] Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"[INFO] Train class balance → Real: {y_train.sum()} | Fake: {(y_train==0).sum()}")

    return X_train, X_test, y_train, y_test


# ── Step 2: Define TF-IDF Vectorizer ─────────────────────────────────────────

def build_tfidf():
    """
    Create a TF-IDF Vectorizer with carefully chosen settings.

    KEY PARAMETERS:
        max_features  — Only use the top N most important words.
                        Using all words is slow and includes noise.
        ngram_range   — (1,2) means use single words AND 2-word phrases.
                        "not good" as a bigram is different from "not" + "good".
        min_df        — Ignore words that appear in fewer than 2 docs.
                        Removes typos and super-rare words.
        max_df        — Ignore words that appear in >95% of docs.
                        Words too common add no signal ("president" in news).
        sublinear_tf  — Apply log(tf) instead of raw tf.
                        Stops very frequent words from dominating.
    """
    return TfidfVectorizer(
        max_features = 50_000,       # Top 50k words/phrases
        ngram_range  = (1, 2),       # Unigrams + bigrams
        min_df       = 2,            # Must appear in ≥2 articles
        max_df       = 0.95,         # Must not appear in >95% of articles
        sublinear_tf = True,         # Apply log scaling to term freq
        strip_accents = "unicode",   # Normalize accented characters
    )


# ── Step 3: Build Model Pipelines ────────────────────────────────────────────

def build_pipelines() -> dict:
    """
    Create sklearn Pipelines — each Pipeline bundles:
        1. TF-IDF vectorizer (converts text → numbers)
        2. A classifier      (learns patterns → predictions)

    WHY Pipelines?
        If you fit TF-IDF separately, you risk accidentally
        transforming test data with info from training data
        (called "data leakage"). A Pipeline keeps them together
        and handles this correctly automatically.

    Returns:
        dict of { "model_name": Pipeline }
    """
    tfidf = build_tfidf()

    pipelines = {

        # ── Model 1: Logistic Regression ─────────────────────────────────
        # Simple but powerful linear classifier. Learns which words
        # correlate with REAL vs FAKE and assigns weights to each.
        # C=1.0 controls regularization (higher C = less regularization).
        "Logistic Regression": Pipeline([
            ("tfidf", build_tfidf()),
            ("clf",   LogisticRegression(
                          C=1.0,
                          max_iter=1000,      # Enough iterations to converge
                          solver="lbfgs",     # Efficient solver for large datasets
                          random_state=RANDOM_STATE
                      ))
        ]),

        # ── Model 2: Passive Aggressive Classifier ────────────────────────
        # An "online learning" algorithm. Updates itself whenever it
        # makes a mistake — "aggressively" correcting wrong predictions.
        # Very fast and often great for text classification.
        # C=0.1 is the aggressiveness parameter.
        "Passive Aggressive": Pipeline([
            ("tfidf", build_tfidf()),
            ("clf",   PassiveAggressiveClassifier(
                          C=0.1,
                          max_iter=50,
                          random_state=RANDOM_STATE
                      ))
        ]),

        # ── Model 3: Random Forest ────────────────────────────────────────
        # Ensemble of many decision trees. More powerful but slower.
        # n_estimators = how many trees (more = better but slower).
        # Handles non-linear patterns Logistic Regression can't learn.
        # NOTE: Needs max_features due to RAM — sparse TF-IDF + RF is heavy.
        "Random Forest": Pipeline([
            ("tfidf", TfidfVectorizer(
                          max_features=10_000,  # Smaller for RF (RAM reasons)
                          ngram_range=(1, 1),   # Unigrams only for RF
                          sublinear_tf=True,
                      )),
            ("clf",   RandomForestClassifier(
                          n_estimators=100,     # 100 trees
                          n_jobs=-1,            # Use all CPU cores
                          random_state=RANDOM_STATE
                      ))
        ]),
    }

    return pipelines


# ── Step 4: Train and Evaluate ────────────────────────────────────────────────

def train_and_evaluate(pipelines: dict, X_train, X_test, y_train, y_test) -> dict:
    """
    Train every pipeline and evaluate it on the test set.

    METRICS EXPLAINED:
        Accuracy  — % of all predictions that are correct
        Precision — Of all "FAKE" predictions, how many were truly FAKE?
        Recall    — Of all actual FAKE articles, how many did we catch?
        F1 Score  — Harmonic mean of Precision & Recall (best single metric)

    For fake news detection, we care about:
        - High Recall for FAKE: We don't want to miss real fake news
        - High Precision for REAL: We don't want to label real news as fake

    Returns:
        dict of { "model_name": {"accuracy": ..., "f1": ..., "pipeline": ...} }
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    results = {}

    for name, pipeline in pipelines.items():
        print(f"\n{'='*50}")
        print(f"[TRAINING] {name}")
        print(f"{'='*50}")

        # ── Train (fit) the pipeline on training data ─────────────────────
        # The pipeline first fits TF-IDF (learns vocab), then trains classifier
        pipeline.fit(X_train, y_train)

        # ── Predict on unseen test data ───────────────────────────────────
        y_pred = pipeline.predict(X_test)

        # ── Compute Metrics ───────────────────────────────────────────────
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="weighted")

        print(f"  Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
        print(f"  F1 Score : {f1:.4f}")
        print(f"\n  Classification Report:")
        print(classification_report(y_test, y_pred,
                                    target_names=["FAKE (0)", "REAL (1)"]))

        # ── Confusion Matrix ──────────────────────────────────────────────
        # Shows: True Positives, False Positives, True Negatives, False Negatives
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, name)

        results[name] = {
            "accuracy":  acc,
            "f1":        f1,
            "pipeline":  pipeline,
        }

    return results


# ── Step 5: Plot Confusion Matrix ─────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, model_name: str) -> None:
    """
    Visualize the confusion matrix as a heatmap.

    CONFUSION MATRIX LAYOUT:
                    Predicted FAKE   Predicted REAL
        Actual FAKE     TN               FP
        Actual REAL     FN               TP

        TN = Correctly labeled fake as fake (True Negative)
        TP = Correctly labeled real as real (True Positive)
        FP = Wrongly labeled real news as fake (False Positive) ← bad!
        FN = Missed fake news, called it real (False Negative) ← worse!
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,           # Show numbers inside cells
        fmt="d",              # Display as integers
        cmap="Blues",         # Blue color gradient
        xticklabels=["FAKE", "REAL"],
        yticklabels=["FAKE", "REAL"]
    )
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    # Save the plot to models folder
    safe_name = model_name.replace(" ", "_").lower()
    save_path = os.path.join(MODELS_DIR, f"confusion_{safe_name}.png")
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  [INFO] Confusion matrix saved: {save_path}")


# ── Step 6: Compare and Save Best Model ──────────────────────────────────────

def save_best_model(results: dict) -> None:
    """
    Find the model with the highest F1 score and save it to disk.

    We use joblib.dump() which:
        - Serializes the entire Pipeline (TF-IDF + classifier) to a file
        - Preserves ALL learned parameters (vocabulary, weights, etc.)
        - Can be loaded later with joblib.load() — no retraining needed!

    WHY F1 and not accuracy?
        F1 accounts for both precision and recall. In imbalanced datasets,
        a model can get high accuracy by predicting the majority class.
        F1 gives a fairer view of performance.
    """
    # Find the best model by F1 score
    best_name     = max(results, key=lambda k: results[k]["f1"])
    best_pipeline = results[best_name]["pipeline"]
    best_f1       = results[best_name]["f1"]
    best_acc      = results[best_name]["accuracy"]

    print(f"\n{'='*50}")
    print(f"[BEST MODEL] {best_name}")
    print(f"  Accuracy : {best_acc:.4f}")
    print(f"  F1 Score : {best_f1:.4f}")
    print(f"{'='*50}")

    # Save the winning pipeline to disk
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_pipeline, BEST_MODEL_PATH)
    print(f"[INFO] Best model saved to: {BEST_MODEL_PATH}")

    # Also save a comparison summary CSV
    summary_rows = []
    for name, res in results.items():
        summary_rows.append({
            "model":    name,
            "accuracy": round(res["accuracy"], 4),
            "f1":       round(res["f1"], 4),
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("f1", ascending=False)
    summary_path = os.path.join(MODELS_DIR, "model_comparison.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"[INFO] Model comparison saved to: {summary_path}")
    print(summary_df.to_string(index=False))


# ── Main Training Orchestrator ────────────────────────────────────────────────

def run_training():
    """
    Full training pipeline:
        1. Load processed data
        2. Split into train/test
        3. Build model pipelines
        4. Train + evaluate all models
        5. Save the best model
    """
    # Load the preprocessed CSV from data/processed/
    df = load_processed(PROCESSED_FILE)

    # Prepare X (text) and y (labels) and split
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Build all our model pipelines
    pipelines = build_pipelines()

    # Train each one and collect metrics
    results = train_and_evaluate(pipelines, X_train, X_test, y_train, y_test)

    # Save the best performing model
    save_best_model(results)

    return results


# ── Script Entry Point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Run: python src/train.py
    results = run_training()

# ── Streamlit-compatible training wrapper ─────────────────────────────────────

def train_best_model(test_size: float = 0.2, max_features: int = 5000) -> dict:
    """
    Train a single best model with configurable parameters.
    
    Used by Streamlit app for interactive training with custom settings.
    
    Args:
        test_size (float): Fraction of data to use for testing (0.0-1.0)
        max_features (int): Max vocabulary size for TF-IDF
    
    Returns:
        dict with keys: accuracy, f1, train_size, test_size
    """
    global TEST_SIZE, RANDOM_STATE
    
    # Update test size for this run
    original_test_size = TEST_SIZE
    TEST_SIZE = test_size
    
    try:
        # Load data
        df = load_processed(PROCESSED_FILE)
        
        # Prepare train/test split
        X = df["content_clean"]
        y = df["label"]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )
        
        # Build optimized pipeline with custom max_features
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
            )),
            ("clf", LogisticRegression(
                C=2.0,
                max_iter=1000,
                solver="lbfgs",
                random_state=RANDOM_STATE,
            )),
        ])
        
        # Train
        print(f"[TRAINING] Logistic Regression with max_features={max_features}")
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        
        print(f"[RESULT] Accuracy: {acc:.4f} | F1: {f1:.4f}")
        
        # Save
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(pipeline, BEST_MODEL_PATH)
        print(f"[SAVED] Model to: {BEST_MODEL_PATH}")
        
        return {
            "accuracy": acc,
            "f1": f1,
            "train_size": len(X_train),
            "test_size": len(X_test),
        }
        
    finally:
        # Restore original test size
        TEST_SIZE = original_test_size