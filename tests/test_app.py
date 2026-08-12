"""Tests for src/app.py.

Uses FastAPI's TestClient, which runs the app in-process (no real network
call, no need for `uvicorn` to be running) and triggers the same
lifespan startup logic as a real server — so these tests exercise the
actual model-loading path in src/app.py, not a mocked stand-in.

Requires models/model.joblib to exist (run `python -m src.train` first).
"""

import pytest
from fastapi.testclient import TestClient

from src.app import app


@pytest.fixture(scope="module")
def client():
    """Yield a TestClient used AS A CONTEXT MANAGER.

    This matters: FastAPI/Starlette only run the app's lifespan startup
    logic (the block in src/app.py that loads models/model.joblib) when
    TestClient is entered via `with`. A bare `TestClient(app)` with no
    `with` skips startup entirely — the model never loads, /health
    reports model_loaded=False, and /predict returns 503. Using a
    fixture guarantees every test gets a client that actually went
    through startup, and shuts down cleanly after the module's tests finish.
    """
    with TestClient(app) as c:
        yield c

# A known-valid payload, reused across tests and mutated per-test where
# a specific field needs to be invalid. Keeping one base payload means
# each test only has to show what's DIFFERENT about it, which is easier
# to read than eighteen fields repeated in every test function.
VALID_PAYLOAD = {
    "tenure": 12,
    "MonthlyCharges": 70.5,
    "TotalCharges": 845.2,
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
}


def test_health_returns_ok_and_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_valid_payload_returns_200_with_expected_shape(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert "churn_probability" in body
    assert "churn_predicted" in body
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert isinstance(body["churn_predicted"], bool)
    # churn_predicted should agree with the 0.5 threshold applied to
    # churn_probability, since that's the exact rule src/app.py uses.
    assert body["churn_predicted"] == (body["churn_probability"] >= 0.5)


def test_predict_high_risk_profile_scores_above_low_risk_profile(client):
    """Sanity check the model's directionality, not an exact number.

    Doesn't assert a specific probability (that would break the moment
    the model is retrained), but DOES assert the ordering, since a
    correctly-behaving churn model must rank a stereotypically
    high-risk customer above a stereotypically low-risk one.
    """
    high_risk = dict(VALID_PAYLOAD)  # already month-to-month/fiber/e-check

    low_risk = dict(VALID_PAYLOAD)
    low_risk.update({
        "tenure": 60,
        "Contract": "Two year",
        "InternetService": "DSL",
        "PaymentMethod": "Credit card (automatic)",
        "OnlineSecurity": "Yes",
        "TechSupport": "Yes",
        "MonthlyCharges": 25.0,
        "TotalCharges": 1500.0,
    })

    high_risk_proba = client.post("/predict", json=high_risk).json()["churn_probability"]
    low_risk_proba = client.post("/predict", json=low_risk).json()["churn_probability"]

    assert high_risk_proba > low_risk_proba


def test_predict_rejects_invalid_categorical_value(client):
    """An unrecognized category should fail validation (422), not reach
    the model and silently degrade through OneHotEncoder's
    handle_unknown="ignore" fallback.
    """
    bad_payload = dict(VALID_PAYLOAD)
    bad_payload["Contract"] = "Lifetime"  # not a real category

    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_rejects_negative_tenure(client):
    bad_payload = dict(VALID_PAYLOAD)
    bad_payload["tenure"] = -5

    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_rejects_missing_required_field(client):
    incomplete_payload = dict(VALID_PAYLOAD)
    del incomplete_payload["Contract"]

    response = client.post("/predict", json=incomplete_payload)
    assert response.status_code == 422