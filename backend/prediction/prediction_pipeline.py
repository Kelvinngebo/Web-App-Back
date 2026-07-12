import os
import numpy as np
from config import PREDICTION_THRESHOLD, RISK_LEVELS

from preprocessing.cleaning import clean_text
from preprocessing.parser import parse_basic
from prediction.predictor import load_artifacts, artifacts_available
from utils.explanation import generate_explanations

# NOTE: This pipeline is a scaffold. Replace feature engineering and vector logic with
# the exact steps used in your research scripts so predictions match training.


def _risk_level_from_prob(p):
    if p < RISK_LEVELS["low"]:
        return "low"
    if p < RISK_LEVELS["medium"]:
        return "medium"
    return "high"


def run_pipeline(ad_text: str, title: str = "") -> dict:
    # Basic cleaning
    cleaned = clean_text(ad_text)

    # Basic parsing (price, phone detection, etc.)
    parsed = parse_basic(cleaned)

    # Load artifacts
    artifacts = load_artifacts()

    # If model artifacts missing, return a helpful error-like response (for dev)
    if not artifacts.get("model") or not artifacts.get("tfidf"):
        return {
            "label": "unknown",
            "probability": 0.0,
            "risk_level": "low",
            "explanations": ["Model artifacts not found. Please place model files under backend/models/ and restart."],
            "features": parsed,
            "_artifact_errors": artifacts.get("_errors", []),
        }

    # Example: vectorize text
    tfidf = artifacts["tfidf"]
    text_vec = tfidf.transform([cleaned]) if hasattr(tfidf, "transform") else None

    # TODO: Replace the following with full feature engineering + combining vectors
    # For now we attempt to call model.predict_proba with text_vec if possible
    model = artifacts["model"]

    try:
        if text_vec is not None:
            # This assumes the model accepts the tfidf vector directly
            probs = model.predict_proba(text_vec)
            suspicious_prob = float(probs[0][1])
        else:
            suspicious_prob = 0.0
    except Exception as e:
        suspicious_prob = 0.0

    label = "suspicious" if suspicious_prob >= PREDICTION_THRESHOLD else "legitimate"
    risk_level = _risk_level_from_prob(suspicious_prob)

    explanations = generate_explanations(parsed, suspicious_prob)

    return {
        "label": label,
        "probability": round(suspicious_prob, 4),
        "risk_level": risk_level,
        "explanations": explanations,
        "features": parsed,
    }
