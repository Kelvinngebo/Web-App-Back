import os
import joblib
from threading import Lock

from config import MODEL_DIR, MODEL_FILES

# For dummy model creation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

_loaded = None
_lock = Lock()


def _create_dummy_artifacts(base, model_path, tfidf_path, scaler_path=None, encoder_path=None):
    """Create and persist a tiny TF-IDF + LogisticRegression model for development/testing.
    This runs when real artifacts are missing so the API remains testable.
    """
    print("Creating dummy artifacts for development/testing...")
    # Tiny synthetic dataset
    texts = [
        "Cheap products buy now",
        "Limited offer, cheap price",
        "Trusted seller business contact",
        "Legitimate business with warranty",
        "Contact me for more details",
        "Exclusive deal, buy now"
    ]
    # labels: 1 -> suspicious, 0 -> legitimate
    labels = [1, 1, 0, 0, 1, 1]

    tfidf = TfidfVectorizer(max_features=500)
    X = tfidf.fit_transform(texts)

    model = LogisticRegression()
    model.fit(X, labels)

    # Persist to disk so future loads use these files
    try:
        os.makedirs(base, exist_ok=True)
        joblib.dump(model, model_path)
        joblib.dump(tfidf, tfidf_path)
        print(f"Dummy model and tfidf saved to {model_path} and {tfidf_path}")
    except Exception as e:
        print("Failed to save dummy artifacts:", e)

    return {"model": model, "tfidf": tfidf, "scaler": None, "encoder": None}


def load_artifacts():
    """Load model artifacts (singleton). Expects files under backend/models/ by default.
    If artifacts are missing, create a tiny dummy model for development.
    Returns a dict: {"model": ..., "tfidf": ..., "scaler": ..., "encoder": ...}
    """
    global _loaded
    if _loaded is not None:
        return _loaded

    with _lock:
        if _loaded is not None:
            return _loaded

        base = MODEL_DIR
        model_path = os.path.join(base, MODEL_FILES.get("model"))
        tfidf_path = os.path.join(base, MODEL_FILES.get("tfidf"))
        scaler_path = os.path.join(base, MODEL_FILES.get("scaler"))
        encoder_path = os.path.join(base, MODEL_FILES.get("encoder"))

        artifacts = {}

        # Attempt to load model
        try:
            if os.path.exists(model_path):
                artifacts["model"] = joblib.load(model_path)
            else:
                artifacts["model"] = None
                artifacts.setdefault("_errors", []).append(f"model_not_found: {model_path}")
        except Exception as e:
            artifacts["model"] = None
            artifacts.setdefault("_errors", []).append(f"model_load_error: {e}")

        # Attempt to load tfidf
        try:
            if os.path.exists(tfidf_path):
                artifacts["tfidf"] = joblib.load(tfidf_path)
            else:
                artifacts["tfidf"] = None
                artifacts.setdefault("_errors", []).append(f"tfidf_not_found: {tfidf_path}")
        except Exception as e:
            artifacts["tfidf"] = None
            artifacts.setdefault("_errors", []).append(f"tfidf_load_error: {e}")

        # Attempt to load scaler
        try:
            if os.path.exists(scaler_path):
                artifacts["scaler"] = joblib.load(scaler_path)
            else:
                artifacts["scaler"] = None
        except Exception as e:
            artifacts["scaler"] = None
            artifacts.setdefault("_errors", []).append(f"scaler_load_error: {e}")

        # Attempt to load encoder
        try:
            if encoder_path and os.path.exists(encoder_path):
                artifacts["encoder"] = joblib.load(encoder_path)
            else:
                artifacts["encoder"] = None
        except Exception as e:
            artifacts["encoder"] = None
            artifacts.setdefault("_errors", []).append(f"encoder_load_error: {e}")

        # If critical artifacts missing, create dummy artifacts for development
        if artifacts.get("model") is None or artifacts.get("tfidf") is None:
            dummy = _create_dummy_artifacts(base, model_path, tfidf_path, scaler_path, encoder_path)
            # Merge dummy artifacts into artifacts
            artifacts["model"] = dummy.get("model")
            artifacts["tfidf"] = dummy.get("tfidf")
            artifacts["scaler"] = dummy.get("scaler")
            artifacts["encoder"] = dummy.get("encoder")
            artifacts.setdefault("_warnings", []).append("dummy_artifacts_created_for_dev")

        _loaded = artifacts
        return _loaded


def artifacts_available():
    arts = load_artifacts()
    return arts.get("model") is not None and arts.get("tfidf") is not None
