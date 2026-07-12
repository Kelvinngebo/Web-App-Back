# Backend model placement and testing

Place your trained model artifacts under backend/models/.
Recommended files (match these names or update backend/config.py):

- fraud_model.pkl
- tfidf_vectorizer.pkl
- feature_scaler.pkl  # optional
- category_encoder.pkl  # optional

If any .pkl files are large (>5-10 MB), the scaffold already adds .gitattributes to track *.pkl with Git LFS. If you haven't set up Git LFS locally, run:

    git lfs install
    git lfs track "*.pkl"
    git add .gitattributes

Then add your model files, commit and push to the branch you prefer (for example, main).

Quick local test (after placing models):

1. Create and activate a virtualenv
2. pip install -r backend/requirements.txt
3. From repo root run:

    export MODEL_DIR=backend/models
    python backend/app.py

4. POST a JSON payload to http://127.0.0.1:5000/api/predict

    { "ad_text": "Selling iPhone X, contact +123456789, price $50" }

The scaffold will attempt to use the tfidf vectorizer directly with the model. Replace prediction/prediction_pipeline.py with the exact feature-engineering and combination logic from your research to ensure parity with training.
