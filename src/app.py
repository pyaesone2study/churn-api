"""FastAPI serving layer for the churn prediction pipeline.

Design notes (for interview defense):
- The pipeline saved by src/train.py already contains preprocessing +
  model as one fitted object. This endpoint calls .predict_proba() on
  that exact object with no separate preprocessing step of its own —
  train/serve parity by construction, not by convention. If the
  preprocessing logic ever changes, it changes in src/pipeline.py once,
  and both training and serving pick it up automatically.
- Pydantic's CustomerFeatures model is the API's actual validation layer:
  categorical fields use Literal[...] so a malformed request (typo'd
  contract type, unexpected string) is rejected with a 422 before it
  ever reaches the model, rather than silently degrading through the
  pipeline's OneHotEncoder(handle_unknown="ignore") into an all-zero
  encoding. That "ignore" behavior exists for genuinely novel categories
  that show up in the wild after training — it's a fallback, not a
  substitute for input validation at the API boundary.
- The model is loaded once at process startup (lifespan), not per
  request. Reloading a joblib pipeline from disk on every request would
  work but adds needless I/O latency to every prediction.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("churn_api")

MODEL_PATH = Path("models/model.joblib")

# Holds the loaded pipeline. Populated at startup via the lifespan
# handler below, not at import time, so the app object itself never
# fails to import even if models/model.joblib is missing (e.g. app.py
# is imported by a test suite that mocks the model separately).
ml_model: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MODEL_PATH.exists():
        ml_model["pipeline"] = joblib.load(MODEL_PATH)
        logger.info(f"Loaded model from {MODEL_PATH}")
    else:
        ml_model["pipeline"] = None
        logger.warning(
            f"{MODEL_PATH} not found — /predict will return 503 until "
            f"src/train.py has been run to produce it."
        )
    yield
    ml_model.clear()


app = FastAPI(title="Churn Prediction API", version="0.1.0", lifespan=lifespan)


class CustomerFeatures(BaseModel):
    """Request schema for /predict.

    Field constraints mirror the actual category values and numeric
    ranges observed in the training data (see notebooks/01_eda.ipynb).
    Numeric lower bounds are enforced (tenure/charges can't be negative);
    deliberately no upper bound is enforced on tenure or charges, since
    a real customer could exceed the training data's historical max —
    capping at the training ceiling would reject legitimate future data.
    """

    tenure: int = Field(ge=0, description="Months the customer has been with the company")
    MonthlyCharges: float = Field(gt=0, description="Current monthly charge amount")
    TotalCharges: float = Field(ge=0, description="Total amount charged to date")

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ]

    model_config = {
        "json_schema_extra": {
            "example": {
                "tenure": 12, "MonthlyCharges": 70.5, "TotalCharges": 845.2,
                "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
                "Dependents": "No", "PhoneService": "Yes", "MultipleLines": "No",
                "InternetService": "Fiber optic", "OnlineSecurity": "No",
                "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
                "StreamingTV": "Yes", "StreamingMovies": "Yes",
                "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
            }
        }
    }


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_predicted: bool


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

    model_config = {"protected_namespaces": ()}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=ml_model.get("pipeline") is not None)


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures) -> PredictionResponse:
    pipeline = ml_model.get("pipeline")
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run 'python -m src.train' to produce models/model.joblib.",
        )

    # SeniorCitizen is int 0/1 at the API boundary (more natural for
    # callers) but the pipeline was trained on it as a string — see
    # src/data.py's astype(str) cast. Convert here so the dataframe
    # matches training exactly.
    payload = features.model_dump()
    payload["SeniorCitizen"] = str(payload["SeniorCitizen"])

    X = pd.DataFrame([payload])

    try:
        proba = float(pipeline.predict_proba(X)[0, 1])
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed.")

    logger.info(f"Prediction served: churn_probability={proba:.4f}")

    return PredictionResponse(churn_probability=proba, churn_predicted=proba >= 0.5)