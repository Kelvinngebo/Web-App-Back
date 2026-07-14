import os
import joblib
import traceback

MODEL_DIR = os.environ.get("MODEL_DIR", "backend/models")
EXPECTED = [
    "fraud_model.pkl",
    "tfidf_vectorizer.pkl",
    "feature_scaler.pkl",
    "category_encoder.pkl",
]

output_lines = []

def log(s):
    print(s)
    output_lines.append(str(s))

def inspect_file(path):
    log(f"--- Inspecting: {path}")
    if not os.path.exists(path):
        log("MISSING")
        return
    size = os.path.getsize(path)
    log(f"Size: {size} bytes")
    try:
        obj = joblib.load(path)
        log(f"Loaded with joblib. Type: {type(obj)}")
        # heuristic inspections
        if hasattr(obj, 'vocabulary_'):
            try:
                vocab = getattr(obj, 'vocabulary_')
                log(f"  vocabulary_ size: {len(vocab)}")
            except Exception as e:
                log(f"  failed getting vocabulary_: {e}")
        if hasattr(obj, 'get_params'):
            try:
                params = obj.get_params()
                sample = dict(list(params.items())[:5])
                log(f"  sample params: {sample}")
            except Exception as e:
                log(f"  failed getting params: {e}")
        if hasattr(obj, 'coef_'):
            try:
                import numpy as _np
                coef = getattr(obj, 'coef_')
                log(f"  coef shape: {_np.shape(coef)}")
            except Exception as e:
                log(f"  failed reading coef_: {e}")
        if hasattr(obj, 'estimators_'):
            try:
                n = len(getattr(obj, 'estimators_'))
                log(f"  n_estimators: {n}")
            except Exception as e:
                log(f"  failed reading estimators_: {e}")
    except Exception as e:
        log(f"joblib.load FAILED: {e}")
        log(traceback.format_exc())

if __name__ == '__main__':
    log(f"Model dir: {MODEL_DIR}")
    os.makedirs('backend/scripts/results', exist_ok=True)
    for fname in EXPECTED:
        path = os.path.join(MODEL_DIR, fname)
        inspect_file(path)

    out_path = 'backend/scripts/results/validate_models_output.txt'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    print('\nValidation complete. Output saved to', out_path)
