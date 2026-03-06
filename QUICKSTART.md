# 🚀 Quick Start Guide

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Running the App

### Option 1: ROOT-LEVEL APP (Recommended - Works without pre-trained model)

```bash
streamlit run streamlit_app.py
```

**Features:**
- ✅ Auto-trains model on startup using built-in example data
- ✅ No pre-trained model required
- ✅ Works on Streamlit Cloud
- ✅ Ready to use immediately

### Option 2: APP FOLDER (If you have training data)

```bash
streamlit run app/app.py
```

**Requirements:**
- Pre-trained model at `models/best_model.pkl`, OR
- Click "Train Model" tab to train on Kaggle dataset

**Features:**
- 📊 Training interface with custom parameters
- 🔍 Feature importance visualization
- 📈 Model performance metrics

---

## Training with Full Kaggle Dataset

### Step 1: Download Data
1. Go to [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
2. Download the dataset
3. Extract `True.csv` and `Fake.csv` to `data/raw/`

### Step 2: Preprocess Data
```bash
python src/preprocess.py
```

This creates `data/processed/news_cleaned.csv`

### Step 3: Train Model
```bash
# Option A: Command line
python src/train.py

# Option B: Via Streamlit app
streamlit run app/app.py
# → Click "Train Model" tab → Click "Train Model Now"
```

---

## Project Files Overview

```
fake_news_detector/
│
├── streamlit_app.py              ← Main entry point (recommended)
│
├── app/
│   └── app.py                    ← Alternative app with training interface
│
├── src/
│   ├── preprocess.py             ← Data loading & text cleaning
│   ├── train.py                  ← Model training
│   ├── predict.py                ← Inference / predictions
│   └── utils.py                  ← Helper functions
│
├── data/
│   ├── raw/                      ← Raw CSV files from Kaggle
│   ├── processed/                ← Cleaned data (auto-generated)
│   └── README.md                 ← Data download instructions
│
├── models/
│   └── best_model.pkl            ← Trained model (auto-generated)
│
├── notebooks/
│   └── 01_exploration.py         ← EDA & analysis
│
├── requirements.txt              ← Python dependencies
├── README.md                     ← Full documentation
└── QUICKSTART.md                 ← This file!
```

---

## Common Issues & Solutions

### ❌ "Model not found" error
**Solution:** Run `streamlit run streamlit_app.py` (trains automatically)

### ❌ NLTK data missing
**Solution:** Automatically downloaded on first run, or run:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```

### ❌ "No module named 'src'"
**Solution:** Make sure you run from project root:
```bash
cd fake_news_detector
streamlit run streamlit_app.py
```

### ❌ Training is slow
**Solution:** This is normal
- Small dataset (~25 features): 5-10 seconds
- Full Kaggle dataset (~45,000 articles): 2-5 minutes

---

## Features

### 🔍 Prediction
- Paste a news article or headline
- Get instant REAL/FAKE classification
- See confidence scores
- View feature importance

### 📊 Analysis
- Confusion matrix visualization
- Feature importance charts
- Model performance metrics
- Preprocessed text inspection

### 🧠 Training
- Train with custom parameters
- Test set size configuration
- TF-IDF vocabulary size tuning
- Real-time training progress

---

## Model Information

**Algorithm:** TF-IDF Vectorizer + Logistic Regression

**Performance:**
- Accuracy: ~98% (with full Kaggle dataset)
- F1 Score: ~98%
- Training data: ~36,000 articles
- Test data: ~9,000 articles

**Limitations:**
- Works best with English text
- Needs 20+ words for best accuracy
- Cannot detect satire or ironic articles
- Trained on 2016-2017 data

---

## Next Steps

1. **Basic Usage:** Run `streamlit run streamlit_app.py` and test with examples
2. **Explore Code:** Check `src/` modules to understand the pipeline
3. **Use Your Data:** Download Kaggle dataset and retrain
4. **Deploy:** Push to GitHub and deploy on [Streamlit Cloud](https://streamlit.io/cloud)

---

## Get Help

- Problems? Check `README.md` for detailed documentation
- Questions? See `src/` module docstrings
- Want to learn? Check `notebooks/01_exploration.py`

**Happy analyzing! 📰✨**
