"""
src/preprocessing.py
====================
Preprocessing pipeline for the AI4I 2020 Predictive Maintenance dataset.

What this module does (step by step):
  1. load_dataset()       – Read the CSV from disk with clear error messages.
  2. validate_columns()   – Check all required columns are present.
  3. report_data_quality()– Print missing-value and duplicate counts.
  4. drop_leaky_columns() – Remove identifiers and sub-failure columns that
                            directly reveal the target (data leakage).
  5. separate_features()  – Split X (features) and y (target).
  6. split_data()         – Stratified train/test split (80/20, seed 42).
  7. run_preprocessing()  – Convenience function that chains all steps.

Usage (from project root, after activating your virtual environment):
    python -m src.preprocessing

The processed splits are saved to data/processed/ as CSV files.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Logging setup — shows INFO-level messages with timestamps
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column definitions (based on official UCI schema)
# ---------------------------------------------------------------------------

# The column that we are trying to predict.
TARGET_COLUMN: str = "Machine failure"

# These columns must be present in the raw CSV.
REQUIRED_COLUMNS: list[str] = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    TARGET_COLUMN,
    "TWF",   # Tool Wear Failure  ─┐
    "HDF",   # Heat Dissipation Failure │ Sub-failure flags — dropped
    "PWF",   # Power Failure            │ before modelling to prevent
    "OSF",   # Overstrain Failure       │ data leakage.
    "RNF",   # Random Failure          ─┘
]

# Columns that identify individual rows — not useful as features.
IDENTIFIER_COLUMNS: list[str] = ["UDI", "Product ID"]

# Sub-failure flags: each is a direct cause of 'Machine failure'.
# Including them would let the model "cheat" (data leakage).
LEAKY_COLUMNS: list[str] = ["TWF", "HDF", "PWF", "OSF", "RNF"]

# Numeric feature columns kept for modelling.
NUMERIC_FEATURES: list[str] = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# Categorical feature columns kept for modelling.
CATEGORICAL_FEATURES: list[str] = ["Type"]


# ---------------------------------------------------------------------------
# Step 1: Load the CSV
# ---------------------------------------------------------------------------

def load_dataset(filepath: str | Path) -> pd.DataFrame:
    """
    Load the AI4I 2020 dataset from a local CSV file.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file (e.g. 'data/raw/ai4i2020.csv').

    Returns
    -------
    pd.DataFrame
        The raw dataset as a DataFrame.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the given path.
    ValueError
        If the file is empty or cannot be parsed as CSV.
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(
            f"\n[ERROR] Dataset file not found: {path.resolve()}\n"
            f"  → Download 'ai4i2020.csv' from:\n"
            f"    https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset\n"
            f"  → Place it at:  data/raw/ai4i2020.csv\n"
        )

    if not path.is_file():
        raise ValueError(f"[ERROR] Path is not a file: {path.resolve()}")

    logger.info("Loading dataset from: %s", path.resolve())

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(
            f"[ERROR] Could not parse '{path}' as a CSV file.\n"
            f"  → Original error: {exc}"
        ) from exc

    if df.empty:
        raise ValueError(
            f"[ERROR] The file '{path}' was loaded but contains no rows.\n"
            f"  → Make sure you downloaded the correct file."
        )

    logger.info("Loaded %d rows × %d columns.", len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# Step 2: Validate columns
# ---------------------------------------------------------------------------

def validate_columns(df: pd.DataFrame) -> None:
    """
    Check that all required columns exist in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded raw dataset.

    Raises
    ------
    ValueError
        If one or more required columns are missing.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(
            f"[ERROR] The dataset is missing {len(missing)} required column(s):\n"
            f"  → Missing: {missing}\n"
            f"  → Found columns: {list(df.columns)}\n"
            f"  → Make sure you are using the official AI4I 2020 CSV from UCI."
        )

    logger.info("Column validation passed. All %d required columns present.", len(REQUIRED_COLUMNS))


# ---------------------------------------------------------------------------
# Step 3: Report data quality
# ---------------------------------------------------------------------------

def report_data_quality(df: pd.DataFrame) -> dict:
    """
    Print a human-readable data quality summary.

    Returns a dictionary so results can be used programmatically
    (e.g. in unit tests or the dashboard).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict with keys:
        'n_rows'         – total rows
        'n_columns'      – total columns
        'n_duplicates'   – exact duplicate rows
        'missing_counts' – {column: count} for columns with missing values
        'target_counts'  – {0: n, 1: n} value counts of target column
    """
    n_rows, n_cols = df.shape
    n_duplicates = int(df.duplicated().sum())

    missing_counts: dict[str, int] = {
        col: int(df[col].isna().sum())
        for col in df.columns
        if df[col].isna().any()
    }

    target_counts: dict[int, int] = df[TARGET_COLUMN].value_counts().to_dict()
    failure_pct = 100.0 * target_counts.get(1, 0) / n_rows

    logger.info("─" * 50)
    logger.info("DATA QUALITY REPORT")
    logger.info("  Rows      : %d", n_rows)
    logger.info("  Columns   : %d", n_cols)
    logger.info("  Duplicates: %d", n_duplicates)

    if missing_counts:
        logger.warning("  Missing values found:")
        for col, count in missing_counts.items():
            logger.warning("    %-35s %d", col, count)
    else:
        logger.info("  Missing values: None ✓")

    logger.info(
        "  Target distribution → No failure: %d  |  Failure: %d  (%.1f%% positive)",
        target_counts.get(0, 0),
        target_counts.get(1, 0),
        failure_pct,
    )
    logger.info("─" * 50)

    return {
        "n_rows": n_rows,
        "n_columns": n_cols,
        "n_duplicates": n_duplicates,
        "missing_counts": missing_counts,
        "target_counts": target_counts,
    }


# ---------------------------------------------------------------------------
# Step 4: Remove leaky and identifier columns
# ---------------------------------------------------------------------------

def drop_leaky_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns that cause data leakage or are not useful features.

    Removed columns:
      - Identifiers  : 'UDI', 'Product ID'  (row IDs, not predictive)
      - Sub-failures : 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'
          These flags directly cause 'Machine failure' = 1.
          Including them lets the model cheat; they are unavailable
          at real prediction time before a failure event is recorded.

    Parameters
    ----------
    df : pd.DataFrame  (raw dataset, after validation)

    Returns
    -------
    pd.DataFrame  (copy without leaky columns)
    """
    cols_to_drop = IDENTIFIER_COLUMNS + LEAKY_COLUMNS
    existing = [c for c in cols_to_drop if c in df.columns]
    dropped = df.drop(columns=existing)

    logger.info(
        "Dropped %d leaky / identifier columns: %s",
        len(existing),
        existing,
    )
    logger.info(
        "Remaining columns (%d): %s",
        len(dropped.columns),
        list(dropped.columns),
    )
    return dropped


# ---------------------------------------------------------------------------
# Step 5: Separate features from target
# ---------------------------------------------------------------------------

def separate_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split the DataFrame into features (X) and the target label (y).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that still contains TARGET_COLUMN.

    Returns
    -------
    X : pd.DataFrame
        Input features (all columns except the target).
    y : pd.Series
        Binary target column ('Machine failure': 0 or 1).

    Raises
    ------
    ValueError
        If TARGET_COLUMN is not present in df.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"[ERROR] Target column '{TARGET_COLUMN}' not found in DataFrame.\n"
            f"  → Call drop_leaky_columns() before separate_features(), "
            f"not the other way around."
        )

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)

    logger.info(
        "Features (X): %d rows × %d columns | Target (y): %d rows",
        X.shape[0], X.shape[1], len(y),
    )
    logger.info("Feature columns: %s", list(X.columns))
    return X, y


# ---------------------------------------------------------------------------
# Step 6: Train / test split
# ---------------------------------------------------------------------------

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split X and y into training and test sets.

    Stratification is used so both splits maintain the same
    proportion of positive (failure) examples as the full dataset.
    This is important because only ~3.4 % of rows are failures.

    Parameters
    ----------
    X : pd.DataFrame  – feature matrix
    y : pd.Series     – binary target
    test_size : float – fraction reserved for testing (default 0.2 = 20 %)
    random_state : int – random seed for reproducibility (default 42)

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    if not (0.0 < test_size < 1.0):
        raise ValueError(
            f"[ERROR] test_size must be between 0 and 1, got: {test_size}"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,          # preserves class ratio in both splits
    )

    logger.info(
        "Train/Test split (%.0f%%/%.0f%%, seed=%d, stratified):",
        (1 - test_size) * 100,
        test_size * 100,
        random_state,
    )
    logger.info(
        "  Train → %d rows  (failures: %d, %.1f%%)",
        len(y_train),
        y_train.sum(),
        100.0 * y_train.mean(),
    )
    logger.info(
        "  Test  → %d rows  (failures: %d, %.1f%%)",
        len(y_test),
        y_test.sum(),
        100.0 * y_test.mean(),
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Step 7: Convenience function — run the full pipeline
# ---------------------------------------------------------------------------

def run_preprocessing(
    raw_path: str | Path = "data/raw/ai4i2020.csv",
    processed_dir: str | Path = "data/processed",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Run the entire preprocessing pipeline end-to-end.

    Steps performed:
      1. Load the dataset from raw_path.
      2. Validate all required columns.
      3. Report data quality.
      4. Drop leaky / identifier columns.
      5. Separate X and y.
      6. Perform stratified train/test split.
      7. Save the four splits to processed_dir as CSV files.

    Parameters
    ----------
    raw_path : str or Path
        Location of the raw CSV (default: 'data/raw/ai4i2020.csv').
    processed_dir : str or Path
        Output directory for processed splits (default: 'data/processed').
    test_size : float
        Fraction for test split (default 0.2).
    random_state : int
        Reproducibility seed (default 42).

    Returns
    -------
    dict with keys:
        'X_train', 'X_test', 'y_train', 'y_test', 'quality_report'
    """
    logger.info("=" * 55)
    logger.info("STARTING PREPROCESSING PIPELINE")
    logger.info("=" * 55)

    # 1. Load
    df = load_dataset(raw_path)

    # 2. Validate
    validate_columns(df)

    # 3. Report quality
    quality = report_data_quality(df)

    # 4. Drop leaky columns
    df_clean = drop_leaky_columns(df)

    # 5. Separate X / y
    X, y = separate_features(df_clean)

    # 6. Split
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=test_size, random_state=random_state
    )

    # 7. Save splits
    out_dir = Path(processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(out_dir / "X_train.csv", index=True)
    X_test.to_csv(out_dir / "X_test.csv", index=True)
    y_train.to_csv(out_dir / "y_train.csv", index=True, header=True)
    y_test.to_csv(out_dir / "y_test.csv", index=True, header=True)

    logger.info("Processed files saved to: %s", out_dir.resolve())
    logger.info("  X_train.csv  X_test.csv  y_train.csv  y_test.csv")
    logger.info("=" * 55)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 55)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "quality_report": quality,
    }


# ---------------------------------------------------------------------------
# Allow running as a script:  python -m src.preprocessing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Accept an optional path argument: python -m src.preprocessing path/to/file.csv
    raw = sys.argv[1] if len(sys.argv) > 1 else "data/raw/ai4i2020.csv"
    run_preprocessing(raw_path=raw)
