"""Data loading and splitting for the Telco Churn project.

Why this lives in src/ and not a notebook:
- The API (Phase 2) and training script both need identical loading logic.
  Duplicating it in a notebook guarantees drift between what you explored
  and what you shipped.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Columns as they appear in the Kaggle CSV
TARGET = "Churn"
ID_COL = "customerID"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_raw(csv_path: str | Path) -> pd.DataFrame:
    """Load the raw Telco CSV and apply minimal, deterministic cleaning.

    Cleaning done here (and only here):
    1. TotalCharges arrives as an *object* dtype because 11 rows contain
       a single space " " — these are customers with tenure == 0 (brand
       new, never billed). We coerce to numeric; the blanks become NaN
       and the pipeline's imputer will handle them. We do NOT drop rows:
       a real scoring API will also see brand-new customers.
    2. SeniorCitizen arrives as 0/1 int but is semantically categorical.
       We cast to string so it flows through the categorical encoder
       like its siblings — one code path, fewer special cases.
    3. Target mapped to 0/1 here so every downstream consumer agrees
       on the encoding.
    """
    df = pd.read_csv(csv_path)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)
    df[TARGET] = df[TARGET].map({"No": 0, "Yes": 1})

    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Drop the ID column and separate X from y.

    customerID is a pure identifier — keeping it risks the model
    memorizing it (a leakage-adjacent bug) and breaks the API schema,
    since callers shouldn't need to invent an ID to get a prediction.
    """
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    return X, y


def make_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split.

    Why stratify: churn is ~26.5% positive. A random split can shift
    that ratio between train and test, which distorts threshold-based
    metrics (F1, confusion matrix) and makes runs harder to compare.
    Stratifying pins the class ratio in both halves.

    Why a fixed random_state: reproducibility. Interview answer:
    "so my metrics are comparable across experiments, and a reviewer
    can rerun my code and get my numbers."
    """
    X, y = split_features_target(df)
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
