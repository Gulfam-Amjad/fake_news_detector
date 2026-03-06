# ============================================================
# src/predict.py
#
# PURPOSE:
#   Load a saved model and make predictions on NEW, unseen
#   news articles. This is the "inference" module.
#
# USAGE:
#   From command line:   python src/predict.py
#   From another file:   from src.predict import predict_news
#
# HOW IT WORKS:
#   1. Load the saved Pipeline (TF-IDF + Classifier) from disk
#   2. Run the same text cleaning as during training
#   3. Feed cleaned text into the pipeline → get prediction
#   4. Return: label (REAL/FAKE) + confidence score
# ============================================================

import os
import joblib          # To load the saved .pkl model file
from src.preprocess import clean_text   # Same cleaning used during training

# ── Path to the saved model ───────────────────────────────────────────────────
MODEL_PATH = os.path.join("models", "best_model.pkl")


# ── Load Model ───────────────────────────────────────────────────────────────

def load_model(model_path: str = MODEL_PATH):
    """
    Load the saved sklearn Pipeline from disk.

    The Pipeline contains BOTH the TF-IDF vectorizer and the classifier,
    so we get a single object that handles everything.

    joblib.load() reads the binary .pkl file and reconstructs
    the exact same Pipeline that was saved during training.

    Raises:
        FileNotFoundError if the model file doesn't exist yet.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'.\n"
            "Please run: python src/train.py   to train and save the model first."
        )

    print(f"[INFO] Loading model from: {model_path}")
    pipeline = joblib.load(model_path)
    print(f"[INFO] Model loaded successfully. Type: {type(pipeline).__name__}")
    return pipeline


# ── Single Prediction ─────────────────────────────────────────────────────────

def predict_news(text: str, pipeline=None) -> dict:
    """
    Predict whether a single news article is REAL or FAKE.

    Steps:
        1. Clean the input text (same preprocessing as training)
        2. Pipeline predicts the label (0=FAKE, 1=REAL)
        3. Pipeline predicts probabilities (confidence %)

    Args:
        text     (str) : Raw news article text (title + body)
        pipeline       : Loaded sklearn pipeline. If None, loads from disk.

    Returns:
        dict: {
            "label":      "REAL" or "FAKE",
            "confidence": float between 0 and 1,
            "prob_real":  probability of being REAL,
            "prob_fake":  probability of being FAKE,
        }

    Example:
        >>> result = predict_news("Scientists discover water on Mars...")
        >>> print(result["label"])     # "REAL"
        >>> print(result["confidence"]) # 0.97
    """
    # Load model from disk if not provided
    if pipeline is None:
        pipeline = load_model()

    # ── Step 1: Clean the raw text ────────────────────────────────────────
    # CRITICAL: We must use the EXACT same cleaning as during training.
    # If training cleaned text one way but we predict on messy text,
    # the model sees patterns it was never trained on → bad predictions.
    cleaned = clean_text(text)

    # Guard: if cleaning removes everything, return uncertain result
    if not cleaned.strip():
        return {
            "label":      "UNCERTAIN",
            "confidence": 0.0,
            "prob_real":  0.5,
            "prob_fake":  0.5,
            "cleaned_text": ""
        }

    # ── Step 2: Get class prediction ─────────────────────────────────────
    # pipeline.predict() runs: TF-IDF transform → classifier predict
    # Returns array like [1] or [0]
    label_code = pipeline.predict([cleaned])[0]   # 0=FAKE, 1=REAL

    # ── Step 3: Get probability scores ───────────────────────────────────
    # predict_proba() returns [[prob_class_0, prob_class_1]]
    # e.g., [[0.03, 0.97]] means 3% FAKE, 97% REAL
    probabilities = pipeline.predict_proba([cleaned])[0]
    prob_fake = probabilities[0]   # Index 0 = class 0 = FAKE
    prob_real = probabilities[1]   # Index 1 = class 1 = REAL

    # ── Step 4: Format result ─────────────────────────────────────────────
    label      = "REAL" if label_code == 1 else "FAKE"
    confidence = prob_real if label_code == 1 else prob_fake

    return {
        "label":        label,
        "confidence":   round(float(confidence), 4),
        "prob_real":    round(float(prob_real),  4),
        "prob_fake":    round(float(prob_fake),  4),
        "cleaned_text": cleaned,
    }


# ── Batch Prediction ──────────────────────────────────────────────────────────

def predict_batch(texts: list, pipeline=None) -> list:
    """
    Run predictions on a list of articles efficiently.

    More efficient than calling predict_news() in a loop because
    sklearn pipelines are optimized for batch operations.

    Args:
        texts    (list of str): List of raw news article texts
        pipeline              : Loaded sklearn pipeline (or None to load from disk)

    Returns:
        list of dicts, one per article (same format as predict_news)
    """
    if pipeline is None:
        pipeline = load_model()

    print(f"[INFO] Running batch prediction on {len(texts)} articles...")

    # Clean all texts first
    cleaned_texts = [clean_text(t) for t in texts]

    # Replace empty strings with a placeholder so pipeline doesn't crash
    cleaned_texts = [t if t.strip() else "empty" for t in cleaned_texts]

    # Batch prediction — much faster than a loop
    label_codes   = pipeline.predict(cleaned_texts)
    probabilities = pipeline.predict_proba(cleaned_texts)

    # Build result list
    results = []
    for i, (code, probs) in enumerate(zip(label_codes, probabilities)):
        label      = "REAL" if code == 1 else "FAKE"
        prob_fake  = probs[0]
        prob_real  = probs[1]
        confidence = prob_real if code == 1 else prob_fake

        results.append({
            "index":      i,
            "label":      label,
            "confidence": round(float(confidence), 4),
            "prob_real":  round(float(prob_real),  4),
            "prob_fake":  round(float(prob_fake),  4),
        })

    return results


# ── Interactive Demo ──────────────────────────────────────────────────────────

def interactive_demo():
    """
    Simple command-line loop to test the model interactively.
    Type or paste a news article, press Enter, and see the prediction.
    Type 'quit' to exit.
    """
    print("\n" + "="*60)
    print("  🔍 Fake News Detector — Interactive Demo")
    print("="*60)
    print("Paste a news headline or article. Type 'quit' to exit.\n")

    # Load once before the loop (saves time on each iteration)
    pipeline = load_model()

    while True:
        print("-"*60)
        user_input = input("📰 Enter news text: ").strip()

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye! 👋")
            break

        if not user_input:
            print("[WARNING] Empty input. Please enter some text.")
            continue

        # Make prediction
        result = predict_news(user_input, pipeline=pipeline)

        # Display result with simple formatting
        icon  = "✅" if result["label"] == "REAL" else "❌"
        label = result["label"]
        conf  = result["confidence"] * 100

        print(f"\n{icon}  Verdict: {label}")
        print(f"   Confidence : {conf:.1f}%")
        print(f"   P(REAL)    : {result['prob_real']*100:.1f}%")
        print(f"   P(FAKE)    : {result['prob_fake']*100:.1f}%")


# ── Script Entry Point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Run: python src/predict.py
    # Launches the interactive command-line demo

    # You can also test with hardcoded examples first:
    EXAMPLE_REAL = """
    NASA's Perseverance rover has successfully collected its first rock core sample
    from Mars, a milestone for the mission that aims to return samples to Earth.
    Scientists at JPL confirmed the sample was sealed in a titanium tube.
    """

    EXAMPLE_FAKE = """
    BREAKING: Hillary Clinton arrested at airport with 40 pounds of cocaine!
    Deep state exposed — mainstream media refuses to cover the TRUTH.
    Share before they DELETE this!!! Trump has the evidence they don't want you to see.
    """

    pipeline = load_model()

    print("\n[TEST 1 - Should be REAL]")
    r1 = predict_news(EXAMPLE_REAL, pipeline=pipeline)
    print(f"  Verdict: {r1['label']} ({r1['confidence']*100:.1f}% confident)")

    print("\n[TEST 2 - Should be FAKE]")
    r2 = predict_news(EXAMPLE_FAKE, pipeline=pipeline)
    print(f"  Verdict: {r2['label']} ({r2['confidence']*100:.1f}% confident)")

    # Launch interactive demo
    interactive_demo()
