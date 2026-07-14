import os
import numpy as np
import pandas as pd
from scipy import sparse

from config import PREDICTION_THRESHOLD, RISK_LEVELS
from preprocessing.feature_engineering import build_features
from prediction.predictor import load_artifacts
from utils.explanation import generate_explanations
from preprocessing.cleaning import clean_text


def _risk_level_from_prob(p):
    if p < RISK_LEVELS["low"]:
        return "low"
    if p < RISK_LEVELS["medium"]:
        return "medium"
    return "high"


def _make_numeric_array(features: dict, numeric_order: list):
    vals = []
    for k in numeric_order:
        v = features.get(k)
        if v is None:
            v = 0
        vals.append(float(v))
    return np.array(vals, dtype=float).reshape(1, -1)


# Define a reasonable numeric feature ordering to match training if possible.
NUMERIC_FEATURE_ORDER = [
    "price_final",
    "image_count",
    "title_length",
    "description_length",
    "title_word_count",
    "description_word_count",
    "missing_description_flag",
    "condition_available",
    "delivery_flag",
    "negotiable_flag",
    "has_price_in_desc",
    "price_missing_flag",
    "log_price",
    "price_zscore_category",
    "category_frequency",
    "category_reliability",
    "suspicious_word_count",
    "fraud_indicator_count",
]


def run_pipeline(ad_text: str, title: str = "", category: str = None, image_count: int = None, contact_type: str = None, preview: bool = False) -> dict:
    # Clean input
    cleaned = clean_text(ad_text)

    # Build features
    features = build_features(title=title, description=cleaned, category=category, image_count=image_count, contact_type=contact_type)

    # Load artifacts
    artifacts = load_artifacts()

    if preview:
        # Return features and whether artifacts are present
        return {
            "features": features,
            "artifacts_present": {
                "model": artifacts.get("model") is not None,
                "tfidf": artifacts.get("tfidf") is not None,
                "scaler": artifacts.get("scaler") is not None,
            },
            "_errors": artifacts.get("_errors", []),
            "_warnings": artifacts.get("_warnings", []),
        }

    # If artifacts missing, return helpful message (but artifacts loader may have created dummy artifacts)
    if not artifacts.get("model") or not artifacts.get("tfidf"):
        return {
            "label": "unknown",
            "probability": 0.0,
            "risk_level": "low",
            "explanations": ["Model artifacts not found. Please place model files under backend/models/ and restart."],
            "features": features,
            "_artifact_errors": artifacts.get("_errors", []),
            "_warnings": artifacts.get("_warnings", []),
        }

    tfidf = artifacts.get("tfidf")
    model = artifacts.get("model")
    scaler = artifacts.get("scaler")

    # Vectorize text
    text = features.get("text_combined", "")
    try:
        text_vec = tfidf.transform([text]) if hasattr(tfidf, "transform") else None
    except Exception as e:
        text_vec = None

    # numeric array
    numeric_arr = _make_numeric_array(features, NUMERIC_FEATURE_ORDER)

    # apply scaler if available
    if scaler is not None:
        try:
            numeric_arr = scaler.transform(numeric_arr)
        except Exception:
            pass

    # Combine text vector and numeric features if possible
    X = None
    if text_vec is not None:
        try:
            X = sparse.hstack([text_vec, sparse.csr_matrix(numeric_arr)], format="csr")
        except Exception:
            # fallback to text vector only
            X = text_vec
    else:
        X = sparse.csr_matrix(numeric_arr)

    # Attempt prediction
    suspicious_prob = 0.0
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            # handle binary/multi-class
            if probs.shape[1] >= 2:
                suspicious_prob = float(probs[0][1])
            else:
                suspicious_prob = float(probs[0][0])
        else:
            # fallback to predict (0/1)
            pred = model.predict(X)
            suspicious_prob = float(pred[0])
    except Exception as e:
        # last-resort: try model on raw text if it's a text pipeline
        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba([text])
                suspicious_prob = float(probs[0][1]) if probs.shape[1] >= 2 else float(probs[0][0])
        except Exception:
            suspicious_prob = 0.0

    label = "suspicious" if suspicious_prob >= PREDICTION_THRESHOLD else "legitimate"
    risk_level = _risk_level_from_prob(suspicious_prob)

    explanations = generate_explanations(features, suspicious_prob)

    resp = {
        "label": label,
        "probability": round(suspicious_prob, 4),
        "risk_level": risk_level,
        "explanations": explanations,
        "features": features,
    }

    # Include any artifact warnings or errors for debugging
    if artifacts.get("_errors"):
        resp["_artifact_errors"] = artifacts.get("_errors")
    if artifacts.get("_warnings"):
        resp["_warnings"] = artifacts.get("_warnings")

    return resp
