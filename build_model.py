from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "WineQT - WineQT.csv.csv"
MODEL_PATH = BASE_DIR / "wine_quality_model.joblib"

FEATURE_COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

FEATURE_LABELS = {
    "fixed acidity": "fixed_acidity",
    "volatile acidity": "volatile_acidity",
    "citric acid": "citric_acid",
    "residual sugar": "residual_sugar",
    "chlorides": "chlorides",
    "free sulfur dioxide": "free_sulfur_dioxide",
    "total sulfur dioxide": "total_sulfur_dioxide",
    "density": "density",
    "pH": "pH",
    "sulphates": "sulphates",
    "alcohol": "alcohol",
}

FEATURE_DESCRIPTIONS = {
    "fixed_acidity": "Tartaric acid level that shapes structure and sharpness.",
    "volatile_acidity": "Acetic acid level; higher values can make wine smell vinegary.",
    "citric_acid": "Citric acid content that can add freshness and balance.",
    "residual_sugar": "Natural sugar left after fermentation.",
    "chlorides": "Salt concentration in the wine.",
    "free_sulfur_dioxide": "Free SO2 that protects wine from oxidation and microbes.",
    "total_sulfur_dioxide": "Overall SO2 content, including bound and free forms.",
    "density": "Liquid density, often linked to sugar and alcohol content.",
    "pH": "Acidity balance of the wine.",
    "sulphates": "Additive that can improve stability and perceived quality.",
    "alcohol": "Alcohol percentage by volume.",
}

FEATURE_UI_MAX = {
    "pH": 14.0,
}

DEFAULT_UI_MAX = 10.0
CALIBRATION_RULES = {
    "Low": {"max": 4.4},
    "Medium": {"min": 4.6, "max": 6.4},
    "High": {"min": 6.6},
}


def get_ui_max(api_name: str) -> float:
    return FEATURE_UI_MAX.get(api_name, DEFAULT_UI_MAX)


def normalize_value(actual_value: float, actual_min: float, actual_max: float, ui_max: float) -> float:
    if actual_max == actual_min:
        return 0.0
    return ((actual_value - actual_min) / (actual_max - actual_min)) * ui_max


def quality_to_label(score: int | float) -> str:
    if score <= 4:
        return "Low"
    if score <= 6:
        return "Medium"
    return "High"


def build_model_package() -> dict[str, object]:
    data = pd.read_csv(DATASET_PATH).drop_duplicates().copy()
    data["quality_label"] = data["quality"].apply(quality_to_label)

    x = data[FEATURE_COLUMNS]
    y_score = data["quality"]
    y_label = data["quality_label"]

    feature_ranges: dict[str, dict[str, float]] = {}
    feature_metadata: list[dict[str, object]] = []

    for column in FEATURE_COLUMNS:
        api_name = FEATURE_LABELS[column]
        actual_min = float(x[column].min())
        actual_max = float(x[column].max())
        ui_max = get_ui_max(api_name)
        default_normalized = normalize_value(float(x[column].median()), actual_min, actual_max, ui_max)

        feature_ranges[api_name] = {
            "actual_min": actual_min,
            "actual_max": actual_max,
            "ui_min": 0.0,
            "ui_max": ui_max,
        }
        feature_metadata.append(
            {
                "column": column,
                "name": api_name,
                "label": column.title(),
                "description": FEATURE_DESCRIPTIONS[api_name],
                "min": 0.0,
                "max": ui_max,
                "step": 0.1,
                "default": round(default_normalized, 1),
            }
        )

    x_train, _x_test, y_train_score, _y_test_score, y_train_label, _y_test_label = train_test_split(
        x,
        y_score,
        y_label,
        test_size=0.2,
        random_state=42,
        stratify=y_label,
    )

    train_df = x_train.copy()
    train_df["quality"] = y_train_score.values
    train_df["quality_label"] = y_train_label.values

    max_class_size = train_df["quality_label"].value_counts().max()
    balanced_parts = []
    for label, group in train_df.groupby("quality_label"):
        balanced_parts.append(
            resample(group, replace=True, n_samples=max_class_size, random_state=42)
        )
    balanced_train_df = pd.concat(balanced_parts).sample(frac=1, random_state=42).reset_index(drop=True)

    scaler = StandardScaler()
    x_train_classifier = balanced_train_df[FEATURE_COLUMNS]
    y_train_classifier = balanced_train_df["quality_label"]
    x_train_classifier_scaled = scaler.fit_transform(x_train_classifier)
    x_train_regressor_scaled = scaler.transform(x_train)

    classifier = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced_subsample",
        random_state=42,
    )
    classifier.fit(x_train_classifier_scaled, y_train_classifier)

    quality_counts = y_train_score.value_counts().to_dict()
    regression_weights = y_train_score.map(lambda value: 1.0 / quality_counts[value]).to_numpy()
    regression_weights = regression_weights / regression_weights.mean()

    regressor = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        random_state=42,
    )
    regressor.fit(x_train_regressor_scaled, y_train_score, sample_weight=regression_weights)

    feature_importance = [
        {"feature": FEATURE_LABELS[column], "importance": round(float(score), 4)}
        for column, score in sorted(
            zip(FEATURE_COLUMNS, regressor.feature_importances_),
            key=lambda item: item[1],
            reverse=True,
        )
    ]

    return {
        "scaler": scaler,
        "classifier": classifier,
        "regressor": regressor,
        "feature_columns": FEATURE_COLUMNS,
        "feature_labels": FEATURE_LABELS,
        "feature_metadata": feature_metadata,
        "feature_importance": feature_importance,
        "feature_ranges": feature_ranges,
        "calibration_rules": CALIBRATION_RULES,
    }


if __name__ == "__main__":
    model_package = build_model_package()
    joblib.dump(model_package, MODEL_PATH, compress=3)
    print(f"Saved trained model package to {MODEL_PATH}")
    print(f"Compressed file size: {MODEL_PATH.stat().st_size} bytes")
