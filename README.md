# 🚀 CyberQuash

CyberQuash is a multilingual AI system for **cyber-abuse detection and sentiment analysis** across Indian and English languages.

## 🔍 Features
- Toxicity Detection (Safe / Offensive / Hate)
- Sentiment Analysis (Positive / Neutral / Negative)
- Supports English, Hindi, Bengali, Tamil, Malayalam, Kannada
- Real-time FastAPI backend
- Modern frontend UI

## 🧠 Models Used
- **XLM-RoBERTa** – Multilingual Toxicity Detection
- **DistilBERT (Multilingual)** – Sentiment Analysis

## 📊 Performance
- Toxicity Detection Accuracy: **~92%**
- Sentiment Analysis Accuracy: **~86%**

## 🛠 Tech Stack
- Python, FastAPI
- PyTorch, HuggingFace Transformers
- HTML, CSS, JavaScript

## ▶ How to Run

### Backend
```bash
uvicorn src.api.api:app --reload
