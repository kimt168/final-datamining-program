import joblib
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from underthesea import word_tokenize
import pandas as pd
STOPWORDS = set([
    "là", "có", "và", "vào", "những", "được", "của", "với", "cho", "về", 
    "tại", "trong", "ở", "đang", "đến", "qua", "vượt", "từ", "bị", "cả", 
    "thì", "rất", "rồi", "vì", "khi", "không", "việt", "nam", "việt_nam", 
    "sau", "hơn", "cách", "để", "bộ"
])

def clean_text(text: str) -> str:
    if pd.isnull(text):  # Kiểm tra nếu văn bản rỗng
        return ""

    # Chuyển thành chữ thường
    text = text.lower()

    # Loại bỏ dấu câu và ký tự đặc biệt
    text = re.sub(f"[{string.punctuation}]", " ", text)

    # Loại bỏ số
    text = re.sub(r"\d+", "", text)

    # Tokenization (Tách từ)
    text_tokenized = word_tokenize(text, format="text")

    # Xóa stopwords
    text_cleaned = " ".join([word for word in text_tokenized.split() if word not in STOPWORDS])

    return text_cleaned

def load_models():
    tfidf = joblib.load("models/tfidf_vectorizer.pkl")
    svm_model = joblib.load("models/SVM.pkl")
    mnb_model = joblib.load("models/MNB.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    return tfidf, svm_model, mnb_model, label_encoder

def predict_label_and_probs(text: str, model, tfidf: TfidfVectorizer, encoder) -> tuple:
    X = tfidf.transform([text])
    pred_label = model.predict(X)[0]
    pred_probs = model.predict_proba(X)[0]
    label = encoder.inverse_transform([pred_label])[0]
    prob_dict = {encoder.classes_[i]: float(pred_probs[i]) for i in range(len(encoder.classes_))}
    sorted_probs = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
    return label, sorted_probs
