"""Preprocessing and model pipelines for the Telco Churn project.

Why everything lives inside sklearn Pipeline/ColumnTransformer objects,
rather than being done ahead of time in pandas:
- cross_validate() clones and refits the whole pipeline on each fold, so
  the imputer's median and the scaler's mean/std are computed from
  training-fold data only. Preprocessing outside the pipeline (e.g.
  scaling the full dataset before splitting) leaks test-set statistics
  into training, which quietly inflates validation scores.
- It also guarantees train/serve parity in Phase 2: the FastAPI app will
  call .predict() on this exact fitted pipeline object, so there is no
  second, hand-written preprocessing path in app.py that could drift out
  of sync with what the model was trained on.
"""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE

# scale_pos_weight for XGBoost: n_negative / n_positive on the training
# set. Churn is ~26.5% positive, so ~73.5 / 26.5 ≈ 2.77. Hard-coded here
# (rather than computed from y_train at call time) because it's a fixed
# property of this dataset's class balance, consistent with class_weight
# ="balanced" below for LogReg/RandomForest — same idea, different API.
XGB_SCALE_POS_WEIGHT = 2.77


def build_preprocessor() -> ColumnTransformer:
    """Build the shared numeric + categorical preprocessing pipeline.

    Numeric branch: median impute -> StandardScaler.
    - Median impute is robust to the right-skew in MonthlyCharges /
      TotalCharges (mean would be pulled by the high-spend tail).
    - StandardScaler matters for LogReg, whose regularized, gradient-based
      optimizer is sensitive to feature scale. It's a harmless no-op for
      RandomForest/XGBoost (tree splits are scale-invariant), which is
      exactly why one shared preprocessor works for all three models
      instead of needing a model-specific branch.

    Categorical branch: most-frequent impute -> OneHotEncoder.
    - handle_unknown="ignore" is a production-robustness choice, not just
      a modeling one: a category never seen in training (a new payment
      method, a data-entry typo) degrades to an all-zero encoding at
      inference time instead of raising and crashing the API request.
    """
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def build_model_pipelines() -> dict[str, Pipeline]:
    """Return {name: full Pipeline} for all three candidate models.

    Each pipeline is preprocessor + estimator, so callers can .fit() /
    .predict() / .score() directly on raw X (untouched dataframes
    straight out of src.data), never on manually-transformed arrays.

    Class imbalance is handled via class weighting, not resampling
    (e.g. SMOTE): simpler, no synthetic-sample artifacts, and performs
    comparably at this moderate ~26.5% imbalance level. Revisit only if
    the PR curve in train.py shows weighting is insufficient.

    Note: each pipeline below calls build_preprocessor() separately
    rather than sharing one ColumnTransformer instance. Sharing a single
    instance across pipelines is a subtle bug — fitting one pipeline
    mutates that shared object's internal state, so a later pipeline
    fit on the same instance would silently overwrite it, corrupting
    any pipeline you'd already fit and were holding onto.
    """
    logreg = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        )),
    ])

    random_forest = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("model", RandomForestClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])

    xgboost = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("model", XGBClassifier(
            scale_pos_weight=XGB_SCALE_POS_WEIGHT,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
        )),
    ])

    return {
        "logreg": logreg,
        "random_forest": random_forest,
        "xgboost": xgboost,
    }