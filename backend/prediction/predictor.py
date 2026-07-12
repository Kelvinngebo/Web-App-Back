import os
import joblib
from threading import Lock

from config import MODEL_DIR, MODEL_FILES

_loaded = None
_lock = Lock()


def load_artifacts():
    """Load model artifacts (singleton). Expects files under backend/models/ by default.
    Returns a dict: {"model": ..., "tfidf": ..., "scaler": ..., "encoder": ...}
    """
    global _loaded
    if _loaded is not None:
        return _loaded

    with _lock:
        if _loaded is not None:
            return _loaded

        base = MODEL_DIR
        # ensure model directory exists
        model_path = os.path.join(base, MODEL_FILES.get("model"))
        tfidf_path = os.path.join(base, MODEL_FILES.get("tfidf"))
        scaler_path = os.path.join(base, MODEL_FILES.get("scaler"))
        encoder_path = os.path.join(base, MODEL_FILES.get("encoder"))

        artifacts = {}
        try:
            artifacts["model"] = joblib.load(model_path)
        except Exception as e:
            artifacts["model"] = None
            artifacts.setdefault("_errors", []).append(f"model_load_error: {e}")

        try:
            artifacts["tfidf"] = joblib.load(tfidf_path)
        except Exception as e:
            artifacts["tfidf"] = None
            artifacts.setdefault("_errors", []).append(f"tfidf_load_error: {e}")

        try:
            artifacts["scaler"] = joblib.load(scaler_path)
        except Exception as e:
            artifacts["scaler"] = None
            artifacts.setdefault("_errors", []).append(f"scaler_load_error: {e}")

        try:
            if encoder_path and os.path.exists(encoder_path):
                artifacts["encoder"] = joblib.load(encoder_path)
            else:
                artifacts["encoder"] = None
        except Exception as e:
            artifacts["encoder"] = None
            artifacts.setdefault("_errors", []).append(f"encoder_load_error: {e}")

        _loaded = artifacts
        return _loaded


def artifacts_available():
    arts = load_artifacts()
    return arts.get("model") is not None and arts.get("tfidf") is not None
