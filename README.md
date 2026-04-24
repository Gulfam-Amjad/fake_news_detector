<div align="center">

<br>

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=700&size=32&pause=1000&color=22C55E&center=true&vCenter=true&width=600&lines=🕵️+Fake+News+Detector;Real+or+Fake%3F+AI+Knows.;98%25+Accuracy+ML+System." alt="Typing SVG" />

<br>

**Powered by Machine Learning · TF-IDF · Logistic Regression · BERT**

<br>

[![Live Demo](https://img.shields.io/badge/⚡_LIVE_DEMO-Click_to_Try-22c55e?style=for-the-badge&logo=streamlit&logoColor=white)](https://fakenewsdetector-xpv6bbzptf2ybnvaxm7wdi.streamlit.app/)
[![Made With Python](https://img.shields.io/badge/Made_with-Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![ML Powered](https://img.shields.io/badge/ML-scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)

<br>

> *"In a world full of misinformation — this tool fights back."*

<br>

</div>

---


# 📰 Fake News Detector

A Machine Learning project to classify news articles as **REAL** or **FAKE**
using NLP techniques — from classic TF-IDF + Logistic Regression to
word embeddings and optionally BERT transformers.

---

## 📁 Folder Structure

```
fake_news_detector/
│
├── data/                        # Raw and processed datasets
│   ├── README.md                # How to download the dataset
│   ├── raw/                     # Original CSV files from Kaggle
│   └── processed/               # Cleaned & saved processed data
│
├── notebooks/                   # Jupyter notebooks for exploration
│   └── 01_exploration.ipynb     # EDA and data understanding
│
├── src/                         # Core source code (importable modules)
│   ├── __init__.py              # Makes src/ a Python package
│   ├── preprocess.py            # Text cleaning & feature engineering
│   ├── train.py                 # Model training & evaluation
│   ├── predict.py               # Load model and make predictions
│   └── utils.py                 # Helper functions (plotting, metrics)
│
├── models/                      # Saved trained models (.pkl files)
│   └── .gitkeep                 # Keeps folder in git even when empty
│
├── app/                         # Streamlit web app for demo
│   └── app.py                   # Interactive UI to test the model
│
├── requirements.txt             # All Python dependencies
└── README.md                    # You are here!
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
Follow the instructions in `data/README.md` to get the Kaggle dataset.

### 3. Train the Model
```bash
python src/train.py
```

### 4. Make a Prediction
```bash
python src/predict.py
```

### 5. Launch the Web App
```bash
streamlit run app/app.py
```

---

## 🧠 Models Used

| Model                        | Accuracy | Notes                        |
|------------------------------|----------|------------------------------|
| TF-IDF + Logistic Regression | ~98%     | Fast, interpretable baseline |
| TF-IDF + Random Forest       | ~96%     | Ensemble, handles non-linear |
| TF-IDF + Passive Aggressive  | ~97%     | Great for online learning    |
| (Optional) BERT              | ~99%+    | Best accuracy, GPU needed    |

---

## 📦 Dataset

We use the **Kaggle Fake News Dataset**:
- `True.csv` — 21,417 real news articles
- `Fake.csv` — 23,481 fake news articles
- Source: https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset

---

## 🔑 Key Concepts You'll Learn

- **TF-IDF**: Term Frequency-Inverse Document Frequency vectorization
- **Word Embeddings**: Dense vector representations of words
- **Class Imbalance**: Handling skewed datasets
- **Pipeline**: Sklearn pipelines for clean ML workflows
- **Model Persistence**: Saving/loading models with joblib
- **Streamlit**: Building ML web apps in Python
