"""
tests/test_preprocessing.py
============================
Unit tests for src/preprocessing.py.

These tests use small in-memory DataFrames that mimic the structure
of the AI4I 2020 dataset. They do NOT require the real CSV file to be
present, so they can run in any environment that has the dependencies
installed.

Run with:
    pytest tests/test_preprocessing.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── import the module under test ────────────────────────────────────────────
from src.preprocessing import (
    CATEGORICAL_FEATURES,
    LEAKY_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    drop_leaky_columns,
    load_dataset,
    report_data_quality,
    run_preprocessing,
    separate_features,
    split_data,
    validate_columns,
)


# ============================================================
# Helpers — build minimal DataFrames that look like AI4I 2020
# ============================================================

def _make_valid_df(n: int = 200, failure_rate: float = 0.1) -> pd.DataFrame:
    """
    Create a minimal DataFrame that matches the AI4I 2020 schema.

    Parameters
    ----------
    n : int
        Number of rows (default 200).
    failure_rate : float
        Fraction of rows where Machine failure = 1 (default 0.1 = 10 %).
        Using 10 % rather than the real 3.4 % makes the stratified split
        reliable even on small synthetic samples.
    """
    rng = np.random.default_rng(seed=0)
    n_failures = max(2, int(n * failure_rate))   # at least 2 positives

    labels = np.zeros(n, dtype=int)
    labels[:n_failures] = 1
    rng.shuffle(labels)

    return pd.DataFrame(
        {
            "UDI": np.arange(1, n + 1),
            "Product ID": [f"M{i}" for i in range(n)],
            "Type": rng.choice(["L", "M", "H"], size=n),
            "Air temperature [K]": rng.uniform(295, 304, n),
            "Process temperature [K]": rng.uniform(305, 314, n),
            "Rotational speed [rpm]": rng.integers(1168, 2861, n),
            "Torque [Nm]": rng.uniform(3.8, 76.6, n),
            "Tool wear [min]": rng.integers(0, 254, n),
            TARGET_COLUMN: labels,
            "TWF": rng.integers(0, 2, n),
            "HDF": rng.integers(0, 2, n),
            "PWF": rng.integers(0, 2, n),
            "OSF": rng.integers(0, 2, n),
            "RNF": rng.integers(0, 2, n),
        }
    )


# ============================================================
# Tests: load_dataset()
# ============================================================

class TestLoadDataset:
    """Tests for the load_dataset() function."""

    def test_raises_for_nonexistent_path(self, tmp_path: Path) -> None:
        """load_dataset should raise FileNotFoundError for a missing file."""
        missing = tmp_path / "does_not_exist.csv"
        with pytest.raises(FileNotFoundError, match="Dataset file not found"):
            load_dataset(missing)

    def test_raises_for_empty_csv(self, tmp_path: Path) -> None:
        """load_dataset should raise ValueError for an empty CSV file."""
        empty_file = tmp_path / "empty.csv"
        empty_file.write_text("")           # completely empty
        with pytest.raises(ValueError):
            load_dataset(empty_file)

    def test_raises_for_headeronly_csv(self, tmp_path: Path) -> None:
        """load_dataset should raise ValueError if CSV has a header but no rows."""
        header_only = tmp_path / "header_only.csv"
        header_only.write_text("col_a,col_b\n")   # header only, no data rows
        with pytest.raises(ValueError, match="no rows"):
            load_dataset(header_only)

    def test_loads_valid_csv(self, tmp_path: Path) -> None:
        """load_dataset returns a non-empty DataFrame for a valid CSV."""
        df = _make_valid_df()
        csv_path = tmp_path / "ai4i2020.csv"
        df.to_csv(csv_path, index=False)

        result = load_dataset(csv_path)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)
        assert list(result.columns) == list(df.columns)

    def test_raises_for_directory_path(self, tmp_path: Path) -> None:
        """load_dataset should raise ValueError if path points to a directory."""
        with pytest.raises(ValueError, match="not a file"):
            load_dataset(tmp_path)   # tmp_path itself is a directory


# ============================================================
# Tests: validate_columns()
# ============================================================

class TestValidateColumns:
    """Tests for the validate_columns() function."""

    def test_passes_with_all_columns(self) -> None:
        """validate_columns should NOT raise when all required columns exist."""
        df = _make_valid_df()
        validate_columns(df)   # should complete without error

    def test_raises_for_missing_target(self) -> None:
        """validate_columns should raise ValueError if target column is missing."""
        df = _make_valid_df().drop(columns=[TARGET_COLUMN])
        with pytest.raises(ValueError, match="missing"):
            validate_columns(df)

    def test_raises_for_missing_feature(self) -> None:
        """validate_columns should raise ValueError for any missing required column."""
        df = _make_valid_df().drop(columns=["Torque [Nm]"])
        with pytest.raises(ValueError, match="missing"):
            validate_columns(df)

    def test_raises_when_multiple_columns_missing(self) -> None:
        """Error message should list ALL missing columns, not just the first."""
        df = _make_valid_df().drop(columns=["TWF", "HDF", "Torque [Nm]"])
        with pytest.raises(ValueError, match="missing"):
            validate_columns(df)

    def test_passes_with_extra_columns(self) -> None:
        """validate_columns should accept a DataFrame with extra, unexpected columns."""
        df = _make_valid_df()
        df["extra_col"] = 0
        validate_columns(df)   # extra columns are fine


# ============================================================
# Tests: report_data_quality()
# ============================================================

class TestReportDataQuality:
    """Tests for the report_data_quality() function."""

    def test_returns_dict_with_expected_keys(self) -> None:
        """report_data_quality should return a dict with the documented keys."""
        df = _make_valid_df()
        result = report_data_quality(df)
        for key in ("n_rows", "n_columns", "n_duplicates", "missing_counts", "target_counts"):
            assert key in result, f"Key '{key}' missing from result"

    def test_reports_correct_row_count(self) -> None:
        """n_rows should match the actual number of rows."""
        df = _make_valid_df(n=200)
        result = report_data_quality(df)
        assert result["n_rows"] == 200

    def test_detects_no_missing_values(self) -> None:
        """missing_counts should be empty for a complete dataset."""
        df = _make_valid_df()
        result = report_data_quality(df)
        assert result["missing_counts"] == {}

    def test_detects_missing_values(self) -> None:
        """missing_counts should record which columns have NaN values."""
        df = _make_valid_df()
        df.loc[0, "Torque [Nm]"] = float("nan")
        result = report_data_quality(df)
        assert "Torque [Nm]" in result["missing_counts"]
        assert result["missing_counts"]["Torque [Nm]"] == 1

    def test_detects_duplicates(self) -> None:
        """n_duplicates should count exact duplicate rows."""
        df = _make_valid_df()
        df_with_dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        result = report_data_quality(df_with_dup)
        assert result["n_duplicates"] >= 1

    def test_no_duplicates_in_clean_data(self) -> None:
        """n_duplicates should be 0 for a dataset with no exact duplicates."""
        df = _make_valid_df()
        result = report_data_quality(df)
        assert result["n_duplicates"] == 0

    def test_target_counts_sum_to_total(self) -> None:
        """target_counts values should sum to the total number of rows."""
        df = _make_valid_df(n=200)
        result = report_data_quality(df)
        assert sum(result["target_counts"].values()) == 200


# ============================================================
# Tests: drop_leaky_columns()
# ============================================================

class TestDropLeakyColumns:
    """Tests for the drop_leaky_columns() function."""

    def test_removes_identifier_columns(self) -> None:
        """UDI and Product ID should not appear in the returned DataFrame."""
        df = _make_valid_df()
        result = drop_leaky_columns(df)
        assert "UDI" not in result.columns
        assert "Product ID" not in result.columns

    def test_removes_all_sub_failure_flags(self) -> None:
        """All five sub-failure columns should be removed."""
        df = _make_valid_df()
        result = drop_leaky_columns(df)
        for col in LEAKY_COLUMNS:
            assert col not in result.columns, f"Column '{col}' should have been dropped"

    def test_target_column_is_kept(self) -> None:
        """The 'Machine failure' target column must NOT be dropped."""
        df = _make_valid_df()
        result = drop_leaky_columns(df)
        assert TARGET_COLUMN in result.columns

    def test_numeric_features_are_kept(self) -> None:
        """All numeric feature columns must be preserved."""
        df = _make_valid_df()
        result = drop_leaky_columns(df)
        for col in NUMERIC_FEATURES:
            assert col in result.columns, f"Numeric feature '{col}' should be kept"

    def test_categorical_features_are_kept(self) -> None:
        """All categorical feature columns must be preserved."""
        df = _make_valid_df()
        result = drop_leaky_columns(df)
        for col in CATEGORICAL_FEATURES:
            assert col in result.columns, f"Categorical feature '{col}' should be kept"

    def test_original_dataframe_not_modified(self) -> None:
        """drop_leaky_columns must not mutate the input DataFrame."""
        df = _make_valid_df()
        original_cols = list(df.columns)
        drop_leaky_columns(df)
        assert list(df.columns) == original_cols


# ============================================================
# Tests: separate_features()
# ============================================================

class TestSeparateFeatures:
    """Tests for the separate_features() function."""

    def _clean_df(self) -> pd.DataFrame:
        return drop_leaky_columns(_make_valid_df())

    def test_returns_tuple_of_correct_types(self) -> None:
        """separate_features should return (DataFrame, Series)."""
        X, y = separate_features(self._clean_df())
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_target_not_in_X(self) -> None:
        """'Machine failure' must NOT appear in the feature matrix X."""
        X, _ = separate_features(self._clean_df())
        assert TARGET_COLUMN not in X.columns

    def test_y_contains_only_binary_values(self) -> None:
        """Target series should contain only 0 and 1."""
        _, y = separate_features(self._clean_df())
        assert set(y.unique()).issubset({0, 1})

    def test_row_counts_match(self) -> None:
        """X and y must have the same number of rows."""
        df = self._clean_df()
        X, y = separate_features(df)
        assert len(X) == len(y) == len(df)

    def test_raises_if_target_already_dropped(self) -> None:
        """separate_features should raise ValueError if target column is absent."""
        df = self._clean_df().drop(columns=[TARGET_COLUMN])
        with pytest.raises(ValueError, match="Target column"):
            separate_features(df)


# ============================================================
# Tests: split_data()
# ============================================================

class TestSplitData:
    """Tests for the split_data() function."""

    def _get_X_y(self) -> tuple[pd.DataFrame, pd.Series]:
        df = drop_leaky_columns(_make_valid_df(n=200, failure_rate=0.1))
        return separate_features(df)

    def test_returns_four_objects(self) -> None:
        """split_data should return exactly four objects."""
        result = split_data(*self._get_X_y())
        assert len(result) == 4

    def test_correct_sizes_default_split(self) -> None:
        """Default 80/20 split: train ≈ 160 rows, test ≈ 40 rows for n=200."""
        X, y = self._get_X_y()
        X_train, X_test, y_train, y_test = split_data(X, y)
        assert len(X_train) == 160
        assert len(X_test) == 40
        assert len(y_train) == 160
        assert len(y_test) == 40

    def test_total_rows_preserved(self) -> None:
        """Train + test rows should equal the original row count."""
        X, y = self._get_X_y()
        X_train, X_test, y_train, y_test = split_data(X, y)
        assert len(X_train) + len(X_test) == len(X)

    def test_stratification_preserves_class_ratio(self) -> None:
        """Both splits should have approximately the same failure rate."""
        X, y = self._get_X_y()
        _, _, y_train, y_test = split_data(X, y)
        # Allow ±5 percentage-point tolerance
        assert abs(y_train.mean() - y_test.mean()) < 0.05

    def test_reproducible_with_same_seed(self) -> None:
        """Two calls with the same random_state must produce identical splits."""
        X, y = self._get_X_y()
        split_a = split_data(X, y, random_state=42)
        split_b = split_data(X, y, random_state=42)
        pd.testing.assert_frame_equal(split_a[0], split_b[0])   # X_train
        pd.testing.assert_series_equal(split_a[2], split_b[2])  # y_train

    def test_different_seeds_give_different_splits(self) -> None:
        """Different random seeds should produce different train sets."""
        X, y = self._get_X_y()
        X_train_42, _, _, _ = split_data(X, y, random_state=42)
        X_train_99, _, _, _ = split_data(X, y, random_state=99)
        # At least some rows differ
        assert not X_train_42.equals(X_train_99)

    def test_invalid_test_size_zero(self) -> None:
        """test_size=0 should raise ValueError."""
        X, y = self._get_X_y()
        with pytest.raises(ValueError, match="test_size"):
            split_data(X, y, test_size=0.0)

    def test_invalid_test_size_one(self) -> None:
        """test_size=1 should raise ValueError."""
        X, y = self._get_X_y()
        with pytest.raises(ValueError, match="test_size"):
            split_data(X, y, test_size=1.0)

    def test_no_overlap_between_train_and_test(self) -> None:
        """Train and test sets must not share any rows (by index)."""
        X, y = self._get_X_y()
        X_train, X_test, _, _ = split_data(X, y)
        overlap = set(X_train.index) & set(X_test.index)
        assert len(overlap) == 0


# ============================================================
# Tests: run_preprocessing() — end-to-end integration test
# ============================================================

class TestRunPreprocessing:
    """Integration tests for the full run_preprocessing() pipeline."""

    def _write_csv(self, tmp_path: Path, n: int = 200) -> Path:
        """Write a minimal AI4I-like CSV to a temp directory and return its path."""
        csv_path = tmp_path / "ai4i2020.csv"
        _make_valid_df(n=n).to_csv(csv_path, index=False)
        return csv_path

    def test_returns_all_expected_keys(self, tmp_path: Path) -> None:
        """run_preprocessing should return a dict with all documented keys."""
        csv_path = self._write_csv(tmp_path)
        result = run_preprocessing(
            raw_path=csv_path,
            processed_dir=tmp_path / "processed",
        )
        for key in ("X_train", "X_test", "y_train", "y_test", "quality_report"):
            assert key in result, f"Key '{key}' missing from result"

    def test_processed_files_are_saved(self, tmp_path: Path) -> None:
        """run_preprocessing should save four CSV files to processed_dir."""
        csv_path = self._write_csv(tmp_path)
        out_dir = tmp_path / "processed"
        run_preprocessing(raw_path=csv_path, processed_dir=out_dir)

        for name in ("X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"):
            assert (out_dir / name).exists(), f"{name} was not saved"

    def test_target_not_in_features(self, tmp_path: Path) -> None:
        """X_train and X_test must not contain the target column."""
        csv_path = self._write_csv(tmp_path)
        result = run_preprocessing(
            raw_path=csv_path,
            processed_dir=tmp_path / "processed",
        )
        assert TARGET_COLUMN not in result["X_train"].columns
        assert TARGET_COLUMN not in result["X_test"].columns

    def test_no_leaky_columns_in_output(self, tmp_path: Path) -> None:
        """None of the leaky columns should appear in X_train or X_test."""
        csv_path = self._write_csv(tmp_path)
        result = run_preprocessing(
            raw_path=csv_path,
            processed_dir=tmp_path / "processed",
        )
        for col in ["UDI", "Product ID"] + LEAKY_COLUMNS:
            assert col not in result["X_train"].columns
            assert col not in result["X_test"].columns

    def test_raises_for_missing_csv(self, tmp_path: Path) -> None:
        """run_preprocessing should propagate FileNotFoundError for missing CSV."""
        with pytest.raises(FileNotFoundError):
            run_preprocessing(
                raw_path=tmp_path / "nonexistent.csv",
                processed_dir=tmp_path / "processed",
            )

    def test_creates_processed_dir_if_missing(self, tmp_path: Path) -> None:
        """run_preprocessing should create processed_dir if it doesn't exist."""
        csv_path = self._write_csv(tmp_path)
        out_dir = tmp_path / "new" / "nested" / "processed"
        run_preprocessing(raw_path=csv_path, processed_dir=out_dir)
        assert out_dir.exists()
