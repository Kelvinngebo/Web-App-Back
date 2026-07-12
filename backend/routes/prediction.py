from flask import Blueprint, request, jsonify

from prediction.prediction_pipeline import run_pipeline

prediction_bp = Blueprint("prediction", __name__)


@prediction_bp.route("/predict", methods=["POST"])
def predict():
    data = request.get_json() or {}
    ad_text = data.get("ad_text", "")
    title = data.get("title", "")

    # run_pipeline returns a dict ready to jsonify
    result = run_pipeline(ad_text=ad_text, title=title)
    return jsonify(result), 200
