from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "wine_quality_model.joblib"


def clamp_score_by_label(score: float, label: str, calibration_rules: dict[str, dict[str, float]]) -> float:
    rules = calibration_rules.get(label, {})
    if "min" in rules:
        score = max(score, float(rules["min"]))
    if "max" in rules:
        score = min(score, float(rules["max"]))
    return score


def denormalize_value(normalized_value: float, actual_min: float, actual_max: float, ui_max: float) -> float:
    if actual_max == actual_min:
        return actual_min
    return actual_min + ((normalized_value / ui_max) * (actual_max - actual_min))


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. Run 'python build_model.py' once before starting the app."
    )


model_package = joblib.load(MODEL_PATH)
scaler_model = model_package["scaler"]
classifier_model = model_package["classifier"]
regressor_model = model_package["regressor"]
feature_columns = model_package["feature_columns"]
feature_labels = model_package["feature_labels"]
feature_metadata = model_package["feature_metadata"]
feature_importance = model_package["feature_importance"]
feature_ranges = model_package["feature_ranges"]
calibration_rules = model_package["calibration_rules"]

app = Flask(__name__)


@app.get("/")
def index():
    return render_template(
        "index.html",
        feature_metadata=feature_metadata,
        feature_importance=feature_importance,
    )


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}

    try:
        actual_values = []
        for column in feature_columns:
            api_name = feature_labels[column]
            normalized_value = float(payload[api_name])
            ui_min = feature_ranges[api_name]["ui_min"]
            ui_max = feature_ranges[api_name]["ui_max"]
            if normalized_value < ui_min or normalized_value > ui_max:
                return jsonify({"error": f"{api_name} must be between {ui_min:g} and {ui_max:g}."}), 400
            actual_values.append(
                denormalize_value(
                    normalized_value,
                    feature_ranges[api_name]["actual_min"],
                    feature_ranges[api_name]["actual_max"],
                    ui_max,
                )
            )
    except KeyError as exc:
        return jsonify({"error": f"Missing field: {exc.args[0]}"}), 400
    except (TypeError, ValueError):
        return jsonify({"error": "All inputs must be numeric and within their allowed ranges."}), 400

    features_df = pd.DataFrame([actual_values], columns=feature_columns)
    features_scaled = scaler_model.transform(features_df)
    predicted_label = classifier_model.predict(features_scaled)[0]
    raw_score = float(regressor_model.predict(features_scaled)[0])
    numeric_score = clamp_score_by_label(raw_score, predicted_label, calibration_rules)
    numeric_score = min(max(numeric_score, 0.0), 10.0)

    return jsonify(
        {
            "quality_label": predicted_label,
            "quality_score": f"{numeric_score:.1f}/10",
            "quality_value": round(numeric_score, 1),
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
