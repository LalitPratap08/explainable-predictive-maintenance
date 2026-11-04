"""
src/train.py
============
Model training and evaluation for the AI4I 2020 Predictive Maintenance dataset.

What this module does (step by step):
  1. build_preprocessor()   – Build a ColumnTransformer that scales numerics
                              and one-hot-encodes the 'Type' categorical.
  2. build_models()         – Return the three classifier configurations.
  3. select_model_by_cv()   – Use 5-fold stratified cross-validation on the
                              TRAINING set to pick the best model.
                              The test set is NOT touched during selection.
  4. evaluate_on_test()     – Report all metrics for the chosen model on the
                              held-out test set.
  5. plot_confusion_matrix() – Save a confusion-matrix PNG to results/.
  6. run_training()         – Convenience function that chains all steps,
                              saves outputs, and returns a results dict.

Usage (from project root, with virtual environment active):
    python -m src.train

Outputs created:
    results/model_comparison_cv.csv   – cross-validation scores for all models
    results/confusion_matrix_<name>.png – one per model (test-set)
    models/best_pipeline.joblib        – the best fitted sklearn Pipeline

Important:
    • Preprocessing (StandardScaler + OneHotEncoder) is fitted on training
      data only to prevent data leakage.
    • class_weight='balanced' handles the ~3.4 % class imbalance without
      over-sampling or other complex techniques.
    • Model selection criterion: mean PR-AUC on validation folds
      (robust to imbalanced classes). The test set is used only for the
      final reported metrics.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Re-use column constants defined in preprocessing
from src.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    run_preprocessing,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42          # reproducibility seed used everywhere
CV_FOLDS: int = 5               # number of stratified cross-validation folds
MODEL_SELECTION_METRIC: str = "test_average_precision"  # PR-AUC on val folds
RESULTS_DIR: Path = Path("results")
MODELS_DIR: Path = Path("models")
RAW_DATA_PATH: Path = Path("data/raw/ai4i2020.csv")


# ---------------------------------------------------------------------------
# Step 1: Build the preprocessing transformer
# ---------------------------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    """
    Build a sklearn ColumnTransformer that:
      - Applies StandardScaler to the 5 numeric sensor columns.
      - Applies OneHotEncoder to the 'Type' categorical column.

    Fitted only on training data (inside the Pipeline) to prevent leakage.

    Returns
    -------
    ColumnTransformer
    """
    numeric_transformer = StandardScaler()

    # handle_unknown='ignore' prevents errors if an unseen category appears
    # at inference time (e.g. a new machine type).
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,   # return dense array (compatible with all estimators)
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",      # discard any column not in either list
    )
    return preprocessor


# ---------------------------------------------------------------------------
# Step 2: Define the three model configurations
# ---------------------------------------------------------------------------

def build_models() -> dict[str, Any]:
    """
    Return a dictionary of {model_name: sklearn_estimator}.

    All classifiers use:
      - class_weight='balanced' — automatically up-weights the minority
        (failure) class. This is appropriate for the ~3.4 % imbalance and
        avoids the complexity of SMOTE or other resampling methods.
      - random_state=RANDOM_STATE — reproducible results.

    Returns
    -------
    dict[str, estimator]
    """
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,          # increase from default 100 to ensure convergence
            random_state=RANDOM_STATE,
            solver="lbfgs",
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_depth=10,           # modest depth limit reduces overfitting
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced",
            n_estimators=200,       # enough trees for stable estimates
            random_state=RANDOM_STATE,
            n_jobs=-1,              # use all available CPU cores
        ),
    }
    return models


# ---------------------------------------------------------------------------
# Step 3: Cross-validate all models on training data
# ---------------------------------------------------------------------------

def select_model_by_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    models: dict[str, Any],
    cv_folds: int = CV_FOLDS,
) -> tuple[str, pd.DataFrame]:
    """
    Evaluate each model using stratified k-fold cross-validation on the
    TRAINING set only. The test set is never used here.

    Scoring metrics collected per fold:
      - accuracy
      - f1 (macro)
      - precision (macro)
      - recall (macro)
      - roc_auc
      - average_precision (= PR-AUC, the primary selection criterion)

    Parameters
    ----------
    X_train, y_train : training data
    preprocessor : ColumnTransformer (unfitted)
    models : dict of {name: estimator}
    cv_folds : number of folds (default 5)

    Returns
    -------
    best_name : str
        Name of the model with the highest mean PR-AUC on validation folds.
    cv_results_df : pd.DataFrame
        Mean ± std of each metric for all models, sorted by PR-AUC descending.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    scoring = {
        "accuracy":          "accuracy",
        "f1":                "f1",
        "precision":         "precision",
        "recall":            "recall",
        "roc_auc":           "roc_auc",
        "average_precision": "average_precision",   # PR-AUC
    }

    all_rows: list[dict] = []

    logger.info("=" * 60)
    logger.info("CROSS-VALIDATION (%d-fold, stratified, training set only)", cv_folds)
    logger.info("=" * 60)

    for name, estimator in models.items():
        logger.info("Evaluating: %s …", name)

        # Wrap preprocessor + estimator in a single Pipeline.
        # The preprocessor is fitted on each fold's training portion only.
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ])

        cv_output = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=1,
        )

        row: dict[str, float] = {"Model": name}
        for metric in scoring:
            values = cv_output[f"test_{metric}"]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"]  = float(np.std(values))

        all_rows.append(row)
        logger.info(
            "  PR-AUC = %.4f ± %.4f  |  Recall = %.4f ± %.4f  |  F1 = %.4f ± %.4f",
            row["average_precision_mean"], row["average_precision_std"],
            row["recall_mean"],           row["recall_std"],
            row["f1_mean"],               row["f1_std"],
        )

    cv_df = pd.DataFrame(all_rows).sort_values("average_precision_mean", ascending=False)
    cv_df = cv_df.reset_index(drop=True)

    best_name = str(cv_df.loc[0, "Model"])
    logger.info("-" * 60)
    logger.info(
        "Best model by CV PR-AUC: %s  (mean PR-AUC = %.4f)",
        best_name,
        cv_df.loc[0, "average_precision_mean"],
    )
    logger.info("-" * 60)
    return best_name, cv_df


# ---------------------------------------------------------------------------
# Step 4: Evaluate a fitted pipeline on the held-out test set
# ---------------------------------------------------------------------------

def evaluate_on_test(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> dict[str, float]:
    """
    Compute all evaluation metrics for a fitted pipeline on the test set.

    Parameters
    ----------
    pipeline : fitted sklearn Pipeline
    X_test, y_test : held-out test data
    model_name : str (used only for logging)

    Returns
    -------
    dict with keys:
        accuracy, precision, recall, f1, roc_auc, pr_auc
    """
    y_pred = pipeline.predict(X_test)

    # predict_proba gives probability scores needed for ROC-AUC and PR-AUC
    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        roc_auc = float(roc_auc_score(y_test, y_prob))
        pr_auc  = float(average_precision_score(y_test, y_prob))
    else:
        logger.warning(
            "%s does not support predict_proba; ROC-AUC and PR-AUC not available.",
            model_name,
        )
        y_prob  = None
        roc_auc = float("nan")
        pr_auc  = float("nan")

    metrics = {
        "accuracy":  float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc":   roc_auc,
        "pr_auc":    pr_auc,
    }

    logger.info("=" * 60)
    logger.info("TEST-SET METRICS: %s", model_name)
    logger.info("=" * 60)
    logger.info("  Accuracy  : %.4f", metrics["accuracy"])
    logger.info("  Precision : %.4f", metrics["precision"])
    logger.info("  Recall    : %.4f", metrics["recall"])
    logger.info("  F1        : %.4f", metrics["f1"])
    logger.info("  ROC-AUC   : %.4f", metrics["roc_auc"])
    logger.info("  PR-AUC    : %.4f", metrics["pr_auc"])
    logger.info("")
    logger.info("  Classification report:")
    report_str = classification_report(y_test, y_pred, target_names=["No failure", "Failure"])
    for line in report_str.splitlines():
        logger.info("    %s", line)

    return metrics


# ---------------------------------------------------------------------------
# Step 5: Plot and save confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    output_dir: Path = RESULTS_DIR,
) -> Path:
    """
    Generate and save a confusion-matrix figure for a fitted pipeline.

    Parameters
    ----------
    pipeline : fitted sklearn Pipeline
    X_test, y_test : test data
    model_name : str
    output_dir : directory where the PNG is saved

    Returns
    -------
    Path to the saved PNG file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No failure", "Failure"],
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=11, pad=10)
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")
    filepath = output_dir / f"confusion_matrix_{safe_name}.png"
    fig.savefig(filepath, dpi=120, bbox_inches="tight")
    plt.close(fig)

    logger.info("Confusion matrix saved: %s", filepath)
    return filepath


# ---------------------------------------------------------------------------
# Step 6: Main pipeline — train, compare, evaluate, save
# ---------------------------------------------------------------------------

def run_training(
    raw_path: str | Path = RAW_DATA_PATH,
    results_dir: str | Path = RESULTS_DIR,
    models_dir: str | Path = MODELS_DIR,
    random_state: int = RANDOM_STATE,
    cv_folds: int = CV_FOLDS,
) -> dict:
    """
    Run the complete training pipeline end-to-end.

    Steps:
      1. Preprocess data (load, validate, drop leaky cols, split).
      2. Build the ColumnTransformer preprocessor.
      3. Define the three models.
      4. Run stratified cross-validation on training data to select best model.
      5. Fit the best model on the full training set.
      6. Evaluate all three models on the held-out test set.
      7. Plot and save confusion matrices.
      8. Save the CV comparison CSV.
      9. Save the best fitted Pipeline via joblib.

    Parameters
    ----------
    raw_path : path to ai4i2020.csv
    results_dir : output directory for plots and CSVs
    models_dir : output directory for joblib artifacts
    random_state : reproducibility seed
    cv_folds : number of CV folds

    Returns
    -------
    dict with keys:
        'cv_results'      : pd.DataFrame (CV scores for all models)
        'test_metrics'    : dict of {model_name: {metric: value}}
        'best_model_name' : str
        'best_pipeline'   : fitted sklearn Pipeline
    """
    results_dir = Path(results_dir)
    models_dir  = Path(models_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Preprocessing ────────────────────────────────────────────────────
    logger.info("Step 1/6 — Running preprocessing …")
    prep_output = run_preprocessing(
        raw_path=raw_path,
        processed_dir="data/processed",
        random_state=random_state,
    )
    X_train = prep_output["X_train"]
    X_test  = prep_output["X_test"]
    y_train = prep_output["y_train"]
    y_test  = prep_output["y_test"]

    # ── 2. Build preprocessor and model list ─────────────────────────────────
    logger.info("Step 2/6 — Building preprocessor and models …")
    preprocessor = build_preprocessor()
    models       = build_models()

    # ── 3. Cross-validation on training set (model selection) ────────────────
    logger.info("Step 3/6 — Cross-validation for model selection …")
    best_name, cv_df = select_model_by_cv(
        X_train, y_train, preprocessor, models, cv_folds=cv_folds
    )

    # Save CV results CSV
    cv_csv_path = results_dir / "model_comparison_cv.csv"
    cv_df.to_csv(cv_csv_path, index=False)
    logger.info("CV results saved: %s", cv_csv_path)

    # ── 4. Fit all three models on full training set ──────────────────────────
    # We fit all three so we can show test-set confusion matrices for each.
    # Model SELECTION was already done via CV; test metrics are for reporting only.
    logger.info("Step 4/6 — Fitting all models on full training set …")
    fitted_pipelines: dict[str, Pipeline] = {}
    test_metrics:    dict[str, dict]    = {}

    for name, estimator in models.items():
        logger.info("Fitting: %s …", name)
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),   # fresh instance per model
            ("classifier", estimator),
        ])
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

    # ── 5. Evaluate all models on test set (reporting only) ──────────────────
    logger.info("Step 5/6 — Evaluating all models on test set …")
    for name, pipeline in fitted_pipelines.items():
        metrics = evaluate_on_test(pipeline, X_test, y_test, name)
        test_metrics[name] = metrics
        plot_confusion_matrix(pipeline, X_test, y_test, name, results_dir)

    # Save test-set metrics JSON
    metrics_path = results_dir / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    logger.info("Test metrics saved: %s", metrics_path)

    # ── 6. Save the best fitted pipeline via joblib ───────────────────────────
    logger.info("Step 6/6 — Saving best pipeline (%s) …", best_name)
    best_pipeline = fitted_pipelines[best_name]
    artifact_path = models_dir / "best_pipeline.joblib"
    joblib.dump(best_pipeline, artifact_path)
    logger.info("Best pipeline saved: %s", artifact_path)

    # Save model metadata alongside the artifact
    metadata = {
        "best_model_name": best_name,
        "selection_criterion": "PR-AUC (5-fold stratified CV on training set)",
        "random_state": random_state,
        "cv_folds": cv_folds,
        "feature_columns": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "target_column": TARGET_COLUMN,
        "test_metrics": test_metrics[best_name],
    }
    meta_path = models_dir / "model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Model metadata saved: %s", meta_path)

    # ── Summary ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("  Best model    : %s", best_name)
    logger.info("  Test Recall   : %.4f", test_metrics[best_name]["recall"])
    logger.info("  Test F1       : %.4f", test_metrics[best_name]["f1"])
    logger.info("  Test PR-AUC   : %.4f", test_metrics[best_name]["pr_auc"])
    logger.info("  Test ROC-AUC  : %.4f", test_metrics[best_name]["roc_auc"])
    logger.info("=" * 60)

    return {
        "cv_results":       cv_df,
        "test_metrics":     test_metrics,
        "best_model_name":  best_name,
        "best_pipeline":    best_pipeline,
    }


# ---------------------------------------------------------------------------
# Allow running as a script:  python -m src.train
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    raw = sys.argv[1] if len(sys.argv) > 1 else str(RAW_DATA_PATH)
    run_training(raw_path=raw)
