
from pathlib import Path
from fastapi.responses import FileResponse
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Resolve file paths relative to this file, not the terminal's location.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "purchase_prediction_model.joblib"

app = FastAPI(
    title="Online Shopper Purchase Prediction API",
    description="Predict purchase intention from online shopping session data.",
    version="1.0.0",
)

# Load the trained model once when the API starts.
try:
    saved_data = joblib.load(MODEL_PATH)
    model = saved_data["model"]
    threshold = float(saved_data["threshold"])
    feature_columns = saved_data["feature_columns"]
except Exception as exc:
    raise RuntimeError(
        f"Could not load the saved model from {MODEL_PATH}. "
        "Check that the model file exists and is valid."
    ) from exc


# These are the features expected by your saved model.
class VisitorSession(BaseModel):
    Administrative: int = Field(ge=0)
    Administrative_Duration: float = Field(ge=0)
    Informational: int = Field(ge=0)
    Informational_Duration: float = Field(ge=0)
    ProductRelated: int = Field(ge=0)
    ProductRelated_Duration: float = Field(ge=0)

    BounceRates: float = Field(ge=0, le=1)
    ExitRates: float = Field(ge=0, le=1)
    SpecialDay: float = Field(ge=0, le=1)

    Month: str
    OperatingSystems: int
    Browser: int
    Region: int
    TrafficType: int
    VisitorType: str
    Weekend: bool


@app.get("/")
def home():
    return {
        "message": "Online Shopper Purchase Prediction API is running.",
        "docs": "/docs",
        "prediction_endpoint": "POST /predict",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "threshold": threshold,
    }


@app.post("/predict")
def predict_purchase(visitor: VisitorSession):
    try:
        # Convert incoming visitor data into a one-row DataFrame.
        visitor_data = visitor.model_dump()
        input_df = pd.DataFrame([visitor_data])

        # Match the exact feature order used during training.
        input_df = input_df[feature_columns]

        # Get the model's estimated purchase probability.
        probability = float(model.predict_proba(input_df)[0, 1])

        # Apply the threshold selected during model evaluation.
        predicted_purchase = probability >= threshold

        return {
            "purchase_probability": round(probability, 4),
            "purchase_probability_percent": round(probability * 100, 2),
            "threshold": threshold,
            "prediction": (
                "Purchase likely"
                if predicted_purchase
                else "Purchase unlikely"
            ),
            "predicted_purchase": predicted_purchase,
            "note": (
                "This is a model estimate, not a guarantee of a purchase."
            ),
        }

    except (KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Input data does not match the model's expected features: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the server terminal for details.",
        ) from exc
@app.get("/store", include_in_schema=False)
def store_demo():
    store_file = BASE_DIR / "store_demo.html"

    if not store_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Demo store HTML file was not found."
        )

    return FileResponse(store_file)