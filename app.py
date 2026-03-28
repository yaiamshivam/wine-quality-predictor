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


def nearest_target_key(target_quality: float, available_keys: list[str]) -> str:
    return min(available_keys, key=lambda key: abs(float(key) - target_quality))


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
quality_range = model_package["quality_range"]
quality_presets = model_package["quality_presets"]
target_feature_profiles = model_package["target_feature_profiles"]

app = Flask(__name__)


@app.get("/")
def index():
    return render_template(
        "index.html",
        feature_metadata=feature_metadata,
        feature_importance=feature_importance,
        quality_range=quality_range,
        quality_presets=quality_presets,
    )


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}

    try:
        actual_values = []
        for column in feature_columns:
            api_name = feature_labels[column]
            value = float(payload[api_name])
            min_value = feature_ranges[api_name]["actual_min"]
            max_value = feature_ranges[api_name]["actual_max"]
            if value < min_value or value > max_value:
                return jsonify({"error": f"{api_name} must be between {min_value:.2f} and {max_value:.2f}."}), 400
            actual_values.append(value)
    except KeyError as exc:
        return jsonify({"error": f"Missing field: {exc.args[0]}"}), 400
    except (TypeError, ValueError):
        return jsonify({"error": "All inputs must be numeric and within the shown dataset ranges."}), 400

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


@app.post("/recommend-features")
def recommend_features():
    payload = request.get_json(silent=True) or {}
    try:
        target_quality = float(payload["target_quality"])
    except KeyError:
        return jsonify({"error": "Missing field: target_quality"}), 400
    except (TypeError, ValueError):
        return jsonify({"error": "target_quality must be numeric."}), 400

    min_quality = float(quality_range["min"])
    max_quality = float(quality_range["max"])
    if target_quality < min_quality or target_quality > max_quality:
        return jsonify({"error": f"target_quality must be between {min_quality:.1f} and {max_quality:.1f}."}), 400

    matched_key = nearest_target_key(target_quality, list(target_feature_profiles.keys()))
    suggested_features = target_feature_profiles[matched_key]
    return jsonify(
        {
            "target_quality": round(float(matched_key), 1),
            "quality_label": "Low" if float(matched_key) <= 4 else ("Medium" if float(matched_key) <= 6 else "High"),
            "features": suggested_features,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
