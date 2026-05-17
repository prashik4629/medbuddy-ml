import logging

import joblib
import pandas as pd

from backend.core.config import settings


logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]


def load_model():
    model = joblib.load(settings.MODEL_PATH)
    logger.info("Model loaded successfully")
    return model


model = load_model()


def predict(input_data: dict) -> dict:
    try:
        df = pd.DataFrame([input_data])[FEATURE_COLUMNS]

        prediction = int(model.predict(df)[0])
        probability = float(model.predict_proba(df)[0][1])

        logger.info(
            f"Prediction: {prediction} | "
            f"Probability: {probability:.4f}"
        )

        return {
            "prediction": prediction,
            "probability": round(probability, 4),
        }

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise