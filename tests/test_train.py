"""
tests/test_train.py
====================
Unit tests for src/train.py.

All tests use small synthetic DataFrames that mimic the AI4I 2020 schema.
No real dataset or trained artifacts are required.

Run with:
    pytest tests/test_train.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.preprocessing import (
    CATEGORICAL_FEATURES,
    LEAKY_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    drop_leaky_columns,
    separate_features,
    split_data,
)
from src.train import (
    RANDOM_STATE,
    build_models,
    build_preprocessor,
    evaluate_on_test,
    plot_confusion_matrix,
    select_model_by_cv,
)


# ============================================================
# Helpers
# ============================================================

def _make_valid_df(n: int = 300, failure_rate: float = 0.15) -> pd.DataFrame:
    """
    Synthetic DataFrame shaped like AI4I 2020 (post-leaky-drop is done later).
    Using 15 % failure rate so CV folds each contain enough positives.
    """
    rng = np.random.default_rng(seed=0)
    n_failures = max(10, int(n * failure_rate))
    labels = np.zeros(n, dtype=int)
    labels[:n_failures] = 1
    rng.shuffle(labels)

    return pd.DataFrame({
        "UDI": np.arange(1, n + 1),
        "Product ID": [f"M{i}" for i in range(n)],
        "Type": rng.choice(["L", "M", "H"], size=n),
        "Air temperature [K]": rng.uniform(295, 304, n),
        "Process temperature [K]": rng.uniform(305, 314, n),
        "Rotational speed [rpm]": rng.integers(1168, 2861, n).astype(float),
        "Torque [Nm]": rng.uniform(3.8, 76.6, n),
        "Tool wear [min]": rng.integers(0, 254, n).astype(float),
        TARGET_COLUMN: labels,
        "TWF": rng.integers(0, 2, n),
        "HDF": rng.integers(0, 2, n),
        "PWF": rng.integers(0, 2, n),
        "OSF": rng.integers(0, 2, n),
        "RNF": rng.integers(0, 2, n),
    })


def _get_split_data(n: int = 300) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return (X_train, X_test, y_train, y_test) from a synthetic dataset."""
    df = _make_valid_df(n=n)
    df_clean = drop_leaky_columns(df)
    X, y = separate_features(df_clean)
    return split_data(X, y, random_state=RANDOM_STATE)


# ============================================================
# Tests: build_preprocessor()
# ============================================================

class TestBuildPreprocessor:

    def test_returns_column_transformer(self) -> None:
        from sklearn.compose import ColumnTransformer
        preprocessor = build_preprocessor()
        assert isinstance(preprocessor, ColumnTransformer)

    def test_transformer_names(self) -> None:
        """Transformer names should be 'num' and 'cat'."""
        preprocessor = build_preprocessor()
        names = [t[0] for t in preprocessor.transformers]
        assert "num" in names
        assert "cat" in names

    def test_fit_transform_runs_without_error(self) -> None:
        """fit_transform should work on valid training data."""
        X_train, _, y_train, _ = _get_split_data()
        preprocessor = build_preprocessor()
        result = preprocessor.fit_transform(X_train)
        assert result.shape[0] == len(X_train)

    def test_output_has_more_columns_than_input_numerics(self) -> None:
        """After OHE the column count should exceed the original numeric count."""
        X_train, _, _, _ = _get_split_data()
        preprocessor = build_preprocessor()
        result = preprocessor.fit_transform(X_train)
        # numerics (5) + OHE of 'Type' (3 categories: L, M, H)
        assert result.shape[1] >= len(NUMERIC_FEATURES)


# ============================================================
# Tests: build_models()
# ============================================================

class TestBuildModels:

    def test_returns_three_models(self) -> None:
        models = build_models()
        assert len(models) == 3

    def test_expected_model_names_present(self) -> None:
        models = build_models()
        assert "Logistic Regression" in models
        assert "Decision Tree"       in models
        assert "Random Forest"       in models

    def test_all_models_have_fit_predict(self) -> None:
        models = build_models()
        for name, estimator in models.items():
            assert hasattr(estimator, "fit"),    f"{name} missing .fit()"
            assert hasattr(estimator, "predict"), f"{name} missing .predict()"

    def test_all_models_use_balanced_class_weight(self) -> None:
        """Every classifier should have class_weight='balanced' set."""
        models = build_models()
        for name, estimator in models.items():
            assert getattr(estimator, "class_weight", None) == "balanced", (
                f"{name} does not use class_weight='balanced'"
            )


# ============================================================
# Tests: select_model_by_cv()
# ============================================================

class TestSelectModelByCv:

    def test_returns_string_and_dataframe(self) -> None:
        X_train, _, y_train, _ = _get_split_data()
        preprocessor = build_preprocessor()
        models       = build_models()
        best_name, cv_df = select_model_by_cv(
            X_train, y_train, preprocessor, models, cv_folds=3
        )
        assert isinstance(best_name, str)
        assert isinstance(cv_df, pd.DataFrame)

    def test_best_name_is_one_of_the_models(self) -> None:
        X_train, _, y_train, _ = _get_split_data()
        preprocessor = build_preprocessor()
        models       = build_models()
        best_name, _ = select_model_by_cv(
            X_train, y_train, preprocessor, models, cv_folds=3
        )
        assert best_name in models

    def test_cv_df_has_row_per_model(self) -> None:
        X_train, _, y_train, _ = _get_split_data()
        preprocessor = build_preprocessor()
        models       = build_models()
        _, cv_df = select_model_by_cv(
            X_train, y_train, preprocessor, models, cv_folds=3
        )
        assert len(cv_df) == len(models)

    def test_cv_df_contains_pr_auc_column(self) -> None:
        X_train, _, y_train, _ = _get_split_data()
        preprocessor = build_preprocessor()
        models       = build_models()
        _, cv_df = select_model_by_cv(
            X_train, y_train, preprocessor, models, cv_folds=3
        )
        assert "average_precision_mean" in cv_df.columns

    def test_cv_df_sorted_by_pr_auc_descending(self) -> None:
        X_train, _, y_train, _ = _get_split_data()
        preprocessor = build_preprocessor()
        models       = build_models()
        _, cv_df = select_model_by_cv(
            X_train, y_train, preprocessor, models, cv_folds=3
        )
        scores = cv_df["average_precision_mean"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_reproduces_same_best_model_with_same_seed(self) -> None:
        X_train, _, y_train, _ = _get_split_data()
        best_a, _ = select_model_by_cv(
            X_train, y_train, build_preprocessor(), build_models(), cv_folds=3
        )
        best_b, _ = select_model_by_cv(
            X_train, y_train, build_preprocessor(), build_models(), cv_folds=3
        )
        assert best_a == best_b


# ============================================================
# Tests: evaluate_on_test()
# ============================================================

class TestEvaluateOnTest:

    def _fitted_pipeline(self) -> tuple[Pipeline, pd.DataFrame, pd.Series]:
        X_train, X_test, y_train, y_test = _get_split_data()
        preprocessor = build_preprocessor()
        from sklearn.ensemble import RandomForestClassifier
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=10,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ])
        pipeline.fit(X_train, y_train)
        return pipeline, X_test, y_test

    def test_returns_dict_with_all_metric_keys(self) -> None:
        pipeline, X_test, y_test = self._fitted_pipeline()
        metrics = evaluate_on_test(pipeline, X_test, y_test, "RF test")
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
            assert key in metrics, f"Metric '{key}' missing from result"

    def test_accuracy_in_valid_range(self) -> None:
        pipeline, X_test, y_test = self._fitted_pipeline()
        metrics = evaluate_on_test(pipeline, X_test, y_test, "RF test")
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_recall_in_valid_range(self) -> None:
        pipeline, X_test, y_test = self._fitted_pipeline()
        metrics = evaluate_on_test(pipeline, X_test, y_test, "RF test")
        assert 0.0 <= metrics["recall"] <= 1.0

    def test_f1_in_valid_range(self) -> None:
        pipeline, X_test, y_test = self._fitted_pipeline()
        metrics = evaluate_on_test(pipeline, X_test, y_test, "RF test")
        assert 0.0 <= metrics["f1"] <= 1.0

    def test_roc_auc_in_valid_range(self) -> None:
        pipeline, X_test, y_test = self._fitted_pipeline()
        metrics = evaluate_on_test(pipeline, X_test, y_test, "RF test")
        assert 0.0 <= metrics["roc_auc"] <= 1.0

    def test_pr_auc_in_valid_range(self) -> None:
        pipeline, X_test, y_test = self._fitted_pipeline()
        metrics = evaluate_on_test(pipeline, X_test, y_test, "RF test")
        assert 0.0 <= metrics["pr_auc"] <= 1.0


# ============================================================
# Tests: plot_confusion_matrix()
# ============================================================

class TestPlotConfusionMatrix:

    def test_saves_png_file(self, tmp_path: Path) -> None:
        X_train, X_test, y_train, y_test = _get_split_data()
        preprocessor = build_preprocessor()
        from sklearn.tree import DecisionTreeClassifier
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ])
        pipeline.fit(X_train, y_train)

        saved_path = plot_confusion_matrix(
            pipeline, X_test, y_test,
            model_name="Decision Tree",
            output_dir=tmp_path,
        )

        assert saved_path.exists(), "PNG file was not created"
        assert saved_path.suffix == ".png"

    def test_filename_contains_model_name(self, tmp_path: Path) -> None:
        X_train, X_test, y_train, y_test = _get_split_data()
        preprocessor = build_preprocessor()
        from sklearn.linear_model import LogisticRegression
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                class_weight="balanced",
                max_iter=500,
                random_state=RANDOM_STATE,
            )),
        ])
        pipeline.fit(X_train, y_train)

        saved_path = plot_confusion_matrix(
            pipeline, X_test, y_test,
            model_name="Logistic Regression",
            output_dir=tmp_path,
        )

        assert "logistic_regression" in saved_path.name


# ============================================================
# Tests: Pipeline integration — fit + save + load
# ============================================================

class TestPipelineSaveLoad:

    def test_joblib_save_and_reload(self, tmp_path: Path) -> None:
        """A fitted pipeline saved with joblib should produce identical predictions."""
        import joblib

        X_train, X_test, y_train, y_test = _get_split_data()
        preprocessor = build_preprocessor()
        from sklearn.ensemble import RandomForestClassifier
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=10,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ])
        pipeline.fit(X_train, y_train)

        artifact_path = tmp_path / "test_pipeline.joblib"
        joblib.dump(pipeline, artifact_path)
        assert artifact_path.exists()

        loaded = joblib.load(artifact_path)
        original_preds = pipeline.predict(X_test)
        loaded_preds   = loaded.predict(X_test)

        np.testing.assert_array_equal(
            original_preds, loaded_preds,
            err_msg="Loaded pipeline predictions differ from original",
        )

    def test_loaded_pipeline_has_predict_proba(self, tmp_path: Path) -> None:
        """Reloaded pipeline should support predict_proba for SHAP / ROC-AUC."""
        import joblib

        X_train, X_test, y_train, _ = _get_split_data()
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", build_models()["Random Forest"]),
        ])
        pipeline.fit(X_train, y_train)

        artifact_path = tmp_path / "pipeline.joblib"
        joblib.dump(pipeline, artifact_path)
        loaded = joblib.load(artifact_path)

        assert hasattr(loaded, "predict_proba")
        proba = loaded.predict_proba(X_test)
        assert proba.shape == (len(X_test), 2)
        # Each row should sum to ~1.0
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
