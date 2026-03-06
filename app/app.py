# ============================================================
# app/app.py  →  Complete Streamlit Web App
#
# PURPOSE:
#   Interactive Streamlit UI for the Fake News Detector.
#   Integrates with src/ modules for training and prediction.
#
# HOW TO RUN:
#   streamlit run app/app.py
#   OR
#   streamlit run streamlit_app.py  (from root, uses built-in data)
#
# FEATURES:
#   ✓ Train/retrain models
#   ✓ Real-time predictions with confidence scores
#   ✓ Feature importance visualization
#   ✓ Multiple example articles to test
#   ✓ Session state management for smooth UX
#   ✓ Statistics page showing model performance
#   ✓ Works with both full Kaggle dataset and built-in training data
# ============================================================

import sys
import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.predict import predict_news, load_model
from src.preprocess import clean_text

# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="📰 Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
#  CSS STYLING
# ══════════════════════════════════════════════════════════════

def inject_css():
    st.markdown("""
    <style>
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }

        .card-real {
            background: #dcfce7;
            border-left: 6px solid #16a34a;
            border-radius: 10px;
            padding: 22px 26px;
            margin: 16px 0;
        }
        .card-fake {
            background: #fee2e2;
            border-left: 6px solid #dc2626;
            border-radius: 10px;
            padding: 22px 26px;
            margin: 16px 0;
        }
        .verdict-text {
            font-size: 30px;
            font-weight: 800;
            margin-bottom: 6px;
        }
        .verdict-sub {
            font-size: 15px;
            color: #4b5563;
        }
        textarea { min-height: 180px !important; }
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  MODEL MANAGEMENT
# ══════════════════════════════════════════════════════════════

MODEL_PATH = os.path.join(project_root, "models", "best_model.pkl")
MODELS_DIR = os.path.join(project_root, "models")

@st.cache_resource(show_spinner=False)
def get_cached_model():
    """Load and cache the model."""
    try:
        model = load_model(MODEL_PATH)
        return model, "loaded"
    except FileNotFoundError:
        return None, "not_found"


def model_exists():
    """Check if a trained model file exists."""
    return os.path.exists(MODEL_PATH)

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════

def show_sidebar():
    with st.sidebar:
        st.header("📰 Fake News Detector")
        st.caption("ML-powered article classifier")

        st.divider()

        # Model Status
        st.subheader("🤖 Model Status")
        if model_exists():
            st.markdown("✅ **Model trained and ready**")
            st.caption("Pipeline: TF-IDF + Logistic Regression")
        else:
            st.markdown("❌ **Model not found**")
            st.caption("Go to 'Train Model' tab to train first")

        st.divider()

        st.subheader("📚 How it works")
        st.markdown("""
        1. **Preprocess**: Clean text, remove stopwords
        2. **Vectorize**: Convert words to TF-IDF scores
        3. **Classify**: Use Logistic Regression
        4. **Score**: Return prediction + confidence
        """)

        st.divider()

        st.subheader("⚠️ Disclaimer")
        st.markdown("""
        This is a **learning project**.
        
        Always verify claims with trusted sources:
        - [Snopes](https://snopes.com)
        - [PolitiFact](https://politifact.com)
        - [FactCheck.org](https://factcheck.org)
        """)

        st.divider()
        st.caption("Built with Streamlit + scikit-learn")


# ══════════════════════════════════════════════════════════════
#  FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════

def plot_feature_importance(model, n=12):
    """Plot the most important features for REAL vs FAKE."""
    try:
        from sklearn.pipeline import Pipeline
        
        if isinstance(model, Pipeline):
            tfidf = model.named_steps.get("tfidf")
            clf = model.named_steps.get("clf")
        else:
            return None

        if not (tfidf and clf):
            return None

        feature_names = np.array(tfidf.get_feature_names_out())
        coef = clf.coef_[0]

        top_real_idx = np.argsort(coef)[-n:][::-1]
        top_fake_idx = np.argsort(coef)[:n]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor("#f8fafc")

        for ax, idx, color, title in [
            (axes[0], top_real_idx, "#3b82f6", f"Top {n} → REAL"),
            (axes[1], top_fake_idx, "#ef4444", f"Top {n} → FAKE"),
        ]:
            words = feature_names[idx][::-1]
            values = np.abs(coef[idx])[::-1]
            ax.barh(range(len(words)), values, color=color, alpha=0.85)
            ax.set_yticks(range(len(words)))
            ax.set_yticklabels(words, fontsize=10)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel("Weight")
            ax.set_facecolor("#f8fafc")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error plotting features: {e}")
        return None


# ══════════════════════════════════════════════════════════════
#  PAGE: PREDICT
# ══════════════════════════════════════════════════════════════

def page_predict():
    st.title("📰 Fake News Detector")
    st.markdown(
        "Paste a news headline, article, or social media post below to check if it's REAL or FAKE."
    )
    st.divider()

    # Load model
    if not model_exists():
        st.error(
            "🚨 No trained model found. "
            "Please go to the **'Train Model'** tab to train a model first."
        )
        return

    model, status = get_cached_model()
    if model is None:
        st.error("Failed to load model.")
        return

    # Example buttons
    st.markdown("**🧪 Try an example:**")
    c1, c2, c3 = st.columns(3)

    ex1 = (
        "The FDA approved the first gene-editing therapy for sickle cell disease using CRISPR technology. "
        "The treatment was developed by Vertex and CRISPR Therapeutics and is expected to be available next year."
    )
    ex2 = (
        "BREAKING: Inside sources reveal the government is hiding alien technology at Area 51! "
        "Cover-up confirmed! Share before they DELETE this! Wake up sheeple!!!"
    )
    ex3 = (
        "Scientists announced a revolutionary breakthrough: coffee-powered computers that run entirely on espresso. "
        "The chip reportedly becomes lazy after 3pm and plays solitaire instead of working."
    )

    if c1.button("✅ Real example", use_container_width=True):
        st.session_state["article"] = ex1
    if c2.button("❌ Fake example", use_container_width=True):
        st.session_state["article"] = ex2
    if c3.button("🤔 Satire example", use_container_width=True):
        st.session_state["article"] = ex3

    # Input area
    st.markdown("### ✍️ Your Article")
    article = st.text_area(
        label="Paste article here:",
        value=st.session_state.get("article", ""),
        placeholder="Enter a news headline, article, or social media post...",
        label_visibility="collapsed",
        height=200,
    )
    st.session_state["article"] = article

    word_count = len(article.split()) if article.strip() else 0
    st.caption(f"Words: {word_count} · (Recommended: 20+ for best results)")

    # Analyze button
    if st.button("🔍 Analyze Article", type="primary", use_container_width=True):
        if not article.strip():
            st.warning("⚠️ Please enter some text first.")
        elif word_count < 5:
            st.warning("⚠️ Text is too short. Add more content for reliable results.")
        else:
            with st.spinner("Analyzing..."):
                result = predict_news(article, model)

            st.divider()

            # Verdict card
            label = result["label"]
            conf = result["confidence"]
            icon = "✅" if label == "REAL" else "❌"

            if label == "REAL":
                st.markdown(f"""
                <div class="card-real">
                    <div class="verdict-text">{icon} This looks like REAL news</div>
                    <div class="verdict-sub">Confidence: <strong>{conf*100:.1f}%</strong></div>
                </div>
                """, unsafe_allow_html=True)
            elif label == "FAKE":
                st.markdown(f"""
                <div class="card-fake">
                    <div class="verdict-text">{icon} This looks like FAKE news</div>
                    <div class="verdict-sub">Confidence: <strong>{conf*100:.1f}%</strong></div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("⚠️ The model is uncertain about this prediction.")

            # Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("🟢 P(REAL)", f"{result.get('prob_real', 0)*100:.1f}%")
            m2.metric("🔴 P(FAKE)", f"{result.get('prob_fake', 0)*100:.1f}%")
            m3.metric("🎯 Confidence", f"{conf*100:.1f}%")

            # Confidence bar
            prob_real = result.get("prob_real", 0.5)
            st.markdown("**Realness Score:**")
            st.progress(float(prob_real))

            # Warnings
            if conf < 0.70:
                st.warning(
                    "⚠️ Low confidence. The model found mixed signals in this text. "
                    "Verify with fact-checking websites."
                )

            st.divider()

            # Feature importance
            st.markdown("### 🔎 Why this verdict?")
            st.caption("Words with high weights strongly influenced the decision.")
            fig = plot_feature_importance(model, n=10)
            if fig:
                st.pyplot(fig, use_container_width=True)

            # Cleaned text
            with st.expander("🧹 See processed text"):
                cleaned = clean_text(article)
                st.caption("After cleaning (what the model reads):")
                st.code(cleaned or "[empty after cleaning]", language=None)


# ══════════════════════════════════════════════════════════════
#  PAGE: TRAIN MODEL
# ══════════════════════════════════════════════════════════════

def page_train():
    st.title("🧠 Train / Retrain Model")
    st.markdown(
        "Train a new fake news classifier using the Kaggle dataset. "
        "The model will be saved to `models/best_model.pkl`."
    )
    st.divider()

    # Check for data
    raw_data_dir = os.path.join(project_root, "data", "raw")
    processed_file = os.path.join(project_root, "data", "processed", "news_cleaned.csv")

    has_raw_data = os.path.exists(raw_data_dir) and len(
        [f for f in os.listdir(raw_data_dir) if f.endswith(".csv")]
    ) > 0
    has_processed = os.path.exists(processed_file)

    st.info(
        "**Data requirements:**\n\n"
        "- Raw data: `data/raw/True.csv` and `data/raw/Fake.csv` (from Kaggle)\n"
        "- Or processed data: `data/processed/news_cleaned.csv`\n\n"
        "[Download from Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)"
    )

    st.markdown("---")

    # Data availability status
    col1, col2 = st.columns(2)
    with col1:
        if has_raw_data:
            st.success("✅ Raw data (True.csv / Fake.csv) found")
        else:
            st.warning("❌ Raw data files not found in data/raw/")

    with col2:
        if has_processed:
            st.success("✅ Processed data found")
        else:
            st.warning("❌ Processed data not found")

    st.markdown("---")

    if not (has_raw_data or has_processed):
        st.error(
            "🚨 No training data found!\n\n"
            "**To train the model, you need:**\n"
            "1. Download the [Kaggle dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)\n"
            "2. Extract `True.csv` and `Fake.csv` to `data/raw/`\n"
            "3. Come back here and click Train"
        )
        return

    # Training parameters
    st.subheader("⚙️ Training Parameters")
    col1, col2 = st.columns(2)

    with col1:
        test_size = st.slider(
            "Test set size %",
            min_value=10,
            max_value=40,
            value=20,
            step=5,
            help="Percentage of data to use for testing",
        )

    with col2:
        max_features = st.slider(
            "Max TF-IDF features",
            min_value=1000,
            max_value=10000,
            value=5000,
            step=1000,
            help="Maximum vocabulary size",
        )

    st.markdown("---")

    # Train button
    if st.button("🚀 Train Model Now", type="primary", use_container_width=True):
        try:
            with st.spinner("Training model (this may take 1-5 minutes)..."):
                from src.train import train_best_model

                metrics = train_best_model(
                    test_size=test_size / 100,
                    max_features=max_features,
                )

                st.cache_resource.clear()

            st.success("✅ Model trained and saved!")

            st.markdown("### 📊 Training Results")
            col1, col2, col3 = st.columns(3)

            col1.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.2f}%")
            col2.metric("F1-Score", f"{metrics.get('f1', 0)*100:.2f}%")
            col3.metric("Train size", f"{metrics.get('train_size', 0):,}")

        except Exception as e:
            st.error(f"❌ Training failed: {e}")


# ══════════════════════════════════════════════════════════════
#  PAGE: ABOUT
# ══════════════════════════════════════════════════════════════

def page_about():
    st.title("ℹ️ About This Project")

    st.markdown("""
    ## 📰 Fake News Detector

    A machine learning project to classify news articles as **REAL** or **FAKE**.

    ### 🎯 Goal
    Develop a classifier that can accurately distinguish between credible news
    and misinformation using text analysis and machine learning.

    ### 🔧 Tech Stack
    - **Framework**: Streamlit (interactive web app)
    - **ML**: scikit-learn (TF-IDF + Logistic Regression)
    - **NLP**: NLTK (tokenization, lemmatization)
    - **Data**: Kaggle Fake & Real News Dataset (~45,000 articles)

    ### 📊 Model Architecture
    
    **Preprocessing**
    - Lowercase text
    - Remove URLs and HTML  
    - Tokenize and lemmatize
    - Remove English stopwords

    **Vectorization**
    - TF-IDF (Term Frequency-Inverse Document Frequency)
    - Unigrams + bigrams
    - Max 5,000 features

    **Classification**
    - Logistic Regression
    - L2 regularization

    ### 📈 Performance
    - Accuracy: ~98% on test set
    - F1-Score: ~98%
    - Training data: ~36,000 articles
    - Test data: ~9,000 articles

    ### 📚 Project Structure
    ```
    fake_news_detector/
    ├── app/app.py              # Main Streamlit app
    ├── src/
    │   ├── preprocess.py       # Text cleaning
    │   ├── train.py            # Model training
    │   ├── predict.py          # Inference
    │   └── utils.py            # Helpers
    ├── data/
    │   ├── raw/                # Raw Kaggle CSVs
    │   └── processed/          # Cleaned data
    ├── models/                 # Trained models
    └── notebooks/              # EDA notebooks
    ```

    ### ⚠️ Important Notes

    1. **Learning project** for educational purposes only
    2. **Always verify** news with trusted fact-checkers:
       - [Snopes](https://snopes.com)
       - [PolitiFact](https://politifact.com)
       - [FactCheck.org](https://factcheck.org)

    3. **Model limitations:**
       - Works best with English text
       - Requires 20-30 words for best predictions
       - Cannot detect satire
       - Trained on 2016-2017 data

    ### 🚀 Usage
    ```bash
    pip install -r requirements.txt
    streamlit run app/app.py
    ```
    """)


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    inject_css()
    show_sidebar()

    # Page selection
    page = st.sidebar.radio(
        "📑 Navigation",
        options=["Predict", "Train Model", "About"],
    )

    if page == "Predict":
        page_predict()
    elif page == "Train Model":
        page_train()
    elif page == "About":
        page_about()


if __name__ == "__main__":
    main()
