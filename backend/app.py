from flask import Flask
from flask_cors import CORS

from routes.prediction import prediction_bp


def create_app():
    app = Flask(__name__)
    CORS(app)
    app.register_blueprint(prediction_bp, url_prefix="/api")

    @app.route("/healthz")
    def health():
        return {"status": "ok"}

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
