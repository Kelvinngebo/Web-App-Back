from flask import Blueprint, request, jsonify

from prediction.prediction_pipeline import run_pipeline

prediction_bp = Blueprint("prediction", __name__)


@prediction_bp.route("/predict", methods=["POST"])
def predict():
    data = request.get_json() or {}
    ad_text = data.get("ad_text", "")
    title = data.get("title", "")
    category = data.get("category")
    image_count = data.get("image_count")
    contact_type = data.get("contact_type")

    result = run_pipeline(ad_text=ad_text, title=title, category=category, image_count=image_count, contact_type=contact_type)
    return jsonify(result), 200


@prediction_bp.route("/preview_features", methods=["POST"])
def preview_features():
    """Return the computed features for an input ad without calling the model. Useful for debugging."""
    data = request.get_json() or {}
    ad_text = data.get("ad_text", "")
    title = data.get("title", "")
    category = data.get("category")
    image_count = data.get("image_count")
    contact_type = data.get("contact_type")

    result = run_pipeline(ad_text=ad_text, title=title, category=category, image_count=image_count, contact_type=contact_type, preview=True)
    return jsonify(result), 200
