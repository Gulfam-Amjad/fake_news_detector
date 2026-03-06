#!/usr/bin/env bash
# ============================================================
# 🚀 LAUNCH SCRIPT - Fake News Detector
# Your 50k-trained model is ready!
# ============================================================

echo "════════════════════════════════════════════════════"
echo "📰 FAKE NEWS DETECTOR"
echo "════════════════════════════════════════════════════"
echo ""
echo "✅ Using your trained model: models/best_model.pkl"
echo "✅ Trained on 50,000 articles"
echo "✅ ~98% accuracy expected"
echo ""
echo "════════════════════════════════════════════════════"
echo "🚀 STARTING STREAMLIT APP..."
echo "════════════════════════════════════════════════════"
echo ""
echo "The app will open in your browser at:"
echo "   http://localhost:8501 or http://localhost:8503"
echo ""
echo "To stop: Press Ctrl+C"
echo ""
echo "════════════════════════════════════════════════════"
echo ""

streamlit run streamlit_app.py
