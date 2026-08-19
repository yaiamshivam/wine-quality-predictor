# Wine Quality Prediction Website

A full-stack machine learning web application that predicts wine quality from chemical properties and presents the result through a polished, presentation-ready interface. The app supports both direct prediction from feature inputs and guided feature generation from a desired quality score.

## Overview

This project was built as a complete AI-powered website for wine quality analysis. Users can:

- enter wine chemical properties manually
- use sliders based on the actual dataset ranges
- generate feature values automatically from a target quality score
- view both a quality label and a numeric prediction
- explore feature importance from the trained model

The application is designed to run locally and also deploy publicly so that anyone can use it without needing the original dataset on their own machine at runtime.

## Features

- Balanced machine learning backend using scikit-learn
- Flask API and server-side rendering
- Dark, modern, glass-style UI
- Real dataset-based feature ranges
- Quality category output: `Low`, `Medium`, `High`
- Numeric score output in `/10` format
- Preset quality buttons for `Poor`, `Medium`, and `High`
- Desired quality generator that auto-fills feature values
- Feature importance chart for presentation/demo purposes
- Public deployment support with Render

## Tech Stack

- Frontend: HTML, CSS, JavaScript
- Backend: Python, Flask
- Machine Learning: scikit-learn
- Data Handling: pandas
- Model Packaging: joblib
- Deployment: Render

## Machine Learning Approach

The current version uses a stronger balanced-model pipeline instead of a plain regressor-only flow.

### Model design

- `RandomForestClassifier` predicts the quality band: `Low`, `Medium`, or `High`
- `RandomForestRegressor` predicts the numeric wine quality score
- `StandardScaler` is used before both models
- calibration rules help keep score ranges aligned with the predicted quality band

### Quality bands

- `Low`: quality `<= 4`
- `Medium`: quality `5 to 6`
- `High`: quality `>= 7`

### Desired quality range

The target quality generator uses the actual range present in the dataset:

- minimum: `3.0`
- maximum: `9.0`

This keeps generated values realistic and consistent with the trained model.

## Project Structure

```text
Wine Quality Predictor Model/
├── app.py
├── build_model.py
├── requirements.txt
├── render.yaml
├── README.md
├── wine_quality_model.joblib
├── WineQT - WineQT.csv.csv
├── static/
│   ├── script.js
│   └── style.css
└── templates/
    └── index.html
```

## Local Setup

Open PowerShell in the project folder:

```powershell
cd "D:\College\Projects\Wine Quality Predictor Model"
pip install -r requirements.txt
python build_model.py
python app.py
```

Then open:

[http://127.0.0.1:5000](http://127.0.0.1:5000)

## How It Works

### Manual prediction

1. Enter or adjust the wine chemistry values.
2. Click `Predict Quality`.
3. The app returns:
   - a quality band
   - a numeric score out of 10

### Auto-generate inputs from a desired quality

1. Choose a target quality score using the desired quality slider.
2. Or click one of the preset buttons: `Poor`, `Medium`, or `High`.
3. The app auto-fills a full feature profile.
4. Run prediction to verify the generated profile.

## Deployment on Render

This project is configured for public deployment using Render.

### 1. Push the project to GitHub

Use the repository connected to Render:

```powershell
git add .
git commit -m "Update project"
git push
```

### 2. Deploy on Render

1. Create or open a Render web service
2. Connect the GitHub repository
3. Render will use `render.yaml`
4. The expected commands are:
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app`

### 3. Open the public site

After deployment, Render provides a public `.onrender.com` URL that can be shared with anyone.

## Important Runtime Note

The public website does not require users to keep the dataset on their own systems.

At runtime, the site uses the packaged model file:

- `wine_quality_model.joblib`

The dataset is only needed when rebuilding the model locally with:

```powershell
python build_model.py
```

## Rebuilding the Model

If you update the notebook, dataset logic, or training approach, regenerate the packaged model before pushing:

```powershell
cd "D:\College\Projects\Wine Quality Predictor Model"
python build_model.py
git add .
git commit -m "Update trained model"
git push
```

## Notes

- The project currently includes the dataset for rebuilding and experimentation.
- The deployed app uses the packaged model artifact for prediction.
- The feature generator is aligned with the trained model so target scores produce more meaningful input combinations.

## Author

Shivam
