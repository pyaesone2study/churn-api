"""Cross-validate the three candidate pipelines, select a winner, fit it
on the full training set, evaluate once on the held-out test set, and
persist the fitted pipeline with joblib.

Design notes (for interview defense):
- Cross-validation and model selection happen entirely on X_train/y_train.
  The test set is touched exactly once, at the very end, for the final
  evaluation numbers reported here. This mirrors the same leakage
  discipline used in the EDA notebook and in src/data.py's split.
- Model selection is automatic (highest mean CV ROC-AUC), not manual,
  so this script can run unattended in a CI job or container entrypoint
  later (Phase 2) without a human reading output and choosing by hand.
  ROC-AUC is threshold-independent, which matters here because the
  actual decision threshold hasn't been chosen yet (that's a business
  decision for Phase 2, not a modeling one). Mean CV F1 is computed and
  printed alongside it purely for interview narrative / sanity-checking
  the ROC-AUC-based choice, not as a tie-breaker.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.data import load_raw, make_split
from src.pipeline import build_model_pipelines

DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODELS_DIR = Path("models")
N_SPLITS = 5
CV_SCORING = ["roc_auc", "f1"]


def cross_validate_pipelines(pipelines: dict, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Run stratified k-fold CV for every candidate pipeline.

    StratifiedKFold (not plain KFold) keeps each fold's ~26.5% churn
    ratio consistent with the overall training set, same reasoning as
    the stratified train/test split in src.data.make_split.

    Returns {name: {"roc_auc_mean", "roc_auc_std", "f1_mean", "f1_std"}}.
    """
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    results = {}

    for name, pipeline in pipelines.items():
        scores = cross_validate(
            pipeline, X_train, y_train,
            cv=cv, scoring=CV_SCORING, n_jobs=-1,
        )
        results[name] = {
            "roc_auc_mean": scores["test_roc_auc"].mean(),
            "roc_auc_std": scores["test_roc_auc"].std(),
            "f1_mean": scores["test_f1"].mean(),
            "f1_std": scores["test_f1"].std(),
        }

    return results


def print_cv_results(results: dict) -> None:
    print(f"\n{'Cross-validation results (' + str(N_SPLITS) + '-fold)':^60}")
    print("-" * 60)
    print(f"{'model':<15}{'ROC-AUC':>20}{'F1':>20}")
    for name, r in results.items():
        roc_str = f"{r['roc_auc_mean']:.4f} +/- {r['roc_auc_std']:.4f}"
        f1_str = f"{r['f1_mean']:.4f} +/- {r['f1_std']:.4f}"
        print(f"{name:<15}{roc_str:>20}{f1_str:>20}")
    print("-" * 60)


def select_best(results: dict) -> str:
    """Pick the pipeline name with the highest mean CV ROC-AUC."""
    best_name = max(results, key=lambda name: results[name]["roc_auc_mean"])
    print(f"\nSelected '{best_name}' (highest mean CV ROC-AUC)")
    return best_name


def evaluate_on_test(pipeline, X_test: pd.DataFrame, y_test: pd.Series, name: str) -> None:
    """One-time held-out evaluation: ROC-AUC, F1, PR curve, confusion matrix.

    This is the ONLY place the test set is used in this script. It's
    used exactly once, after the winning pipeline has already been
    chosen from CV results, so this number can't leak into model
    selection.
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    test_roc_auc = roc_auc_score(y_test, y_proba)
    test_f1 = f1_score(y_test, y_pred)
    avg_precision = average_precision_score(y_test, y_proba)

    print(f"\n{'Held-out test evaluation: ' + name:^60}")
    print("-" * 60)
    print(f"ROC-AUC:            {test_roc_auc:.4f}")
    print(f"F1:                 {test_f1:.4f}")
    print(f"Average precision:  {avg_precision:.4f}  (PR-curve baseline = 0.265)")

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion matrix (rows=actual, cols=predicted, [No, Yes]):")
    print(cm)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm, display_labels=["No churn", "Churn"]).plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix — {name}")
    fig.tight_layout()
    fig.savefig(MODELS_DIR / f"confusion_matrix_{name}.png", dpi=150)
    plt.close(fig)

    from sklearn.metrics import PrecisionRecallDisplay
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=ax, name=name)
    ax.axhline(0.265, color="gray", linestyle="--", label="baseline (0.265)")
    ax.set_title(f"Precision-Recall curve — {name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(MODELS_DIR / f"pr_curve_{name}.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved confusion_matrix_{name}.png and pr_curve_{name}.png to {MODELS_DIR}/")


def save_model(pipeline, name: str) -> Path:
    """Save the fitted pipeline under two names:
    - "{name}_pipeline.joblib": which model actually won, for your own
      records/README, e.g. logreg_pipeline.joblib.
    - "model.joblib": a fixed, algorithm-agnostic filename that src/app.py
      always loads. This decouples the serving layer from which model
      happens to win a given training run — if you rerun train.py later
      and XGBoost wins instead of LogReg, app.py doesn't need to change
      or know the difference.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    named_path = MODELS_DIR / f"{name}_pipeline.joblib"
    joblib.dump(pipeline, named_path)
    print(f"Saved fitted pipeline to {named_path}")

    canonical_path = MODELS_DIR / "model.joblib"
    joblib.dump(pipeline, canonical_path)
    print(f"Saved fitted pipeline to {canonical_path} (canonical path used by src/app.py)")

    return named_path


def main() -> None:
    df = load_raw(DATA_PATH)
    X_train, X_test, y_train, y_test = make_split(df)

    pipelines = build_model_pipelines()

    cv_results = cross_validate_pipelines(pipelines, X_train, y_train)
    print_cv_results(cv_results)

    best_name = select_best(cv_results)
    best_pipeline = pipelines[best_name]

    # Refit the winner on the FULL training set (CV folds only ever saw
    # ~80% of it at a time). This is the pipeline that gets evaluated on
    # test and ultimately saved.
    best_pipeline.fit(X_train, y_train)

    evaluate_on_test(best_pipeline, X_test, y_test, best_name)
    save_model(best_pipeline, best_name)


if __name__ == "__main__":
    main()