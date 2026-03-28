from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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


def get_ui_max(api_name: str) -> float:
    return FEATURE_UI_MAX.get(api_name, DEFAULT_UI_MAX)


def normalize_value(actual_value: float, actual_min: float, actual_max: float, ui_max: float) -> float:
    if actual_max == actual_min:
        return 0.0
    return ((actual_value - actual_min) / (actual_max - actual_min)) * ui_max


def build_model_package() -> dict[str, object]:
    data = pd.read_csv(DATASET_PATH).drop_duplicates()
    x = data[FEATURE_COLUMNS]
    y = data["quality"]

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

    x_train, _x_test, y_train, _y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    regressor = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        random_state=42,
    )
    regressor.fit(x_train_scaled, y_train)

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
        "regressor": regressor,
        "feature_columns": FEATURE_COLUMNS,
        "feature_labels": FEATURE_LABELS,
        "feature_metadata": feature_metadata,
        "feature_importance": feature_importance,
        "feature_ranges": feature_ranges,
    }


if __name__ == "__main__":
    model_package = build_model_package()
    joblib.dump(model_package, MODEL_PATH)
    print(f"Saved trained model package to {MODEL_PATH}")
