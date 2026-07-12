# Backend configuration
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(BASE_DIR, "models"))

MODEL_FILES = {
    "model": os.environ.get("MODEL_FILE", "fraud_model.pkl"),
    "tfidf": os.environ.get("TFIDF_FILE", "tfidf_vectorizer.pkl"),
    "scaler": os.environ.get("SCALER_FILE", "feature_scaler.pkl"),
    "encoder": os.environ.get("ENCODER_FILE", "category_encoder.pkl"),
}

# Prediction threshold for labeling suspicious items
PREDICTION_THRESHOLD = float(os.environ.get("PREDICTION_THRESHOLD", 0.5))

# Risk level thresholds
RISK_LEVELS = {
    "low": 0.33,
    "medium": 0.66,
}
