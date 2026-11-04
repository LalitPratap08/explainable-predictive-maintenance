"""
dashboard/app.py
================
Streamlit dashboard for the Predictive Maintenance Using Machine Learning project.

Run from the project root (where models/, results/, src/ live):
    streamlit run dashboard/app.py

Requirements:
    - streamlit
    - joblib
    - pandas
    - numpy
    - scikit-learn  (must match the version used to train the saved pipeline)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration — must be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Paths  (all relative to the project root so the app works from any machine)
# ---------------------------------------------------------------------------
PROJECT_ROOT   = Path(__file__).resolve().parent.parent   # …/explainable-predictive-maintenance
MODELS_DIR     = PROJECT_ROOT / "models"
RESULTS_DIR    = PROJECT_ROOT / "results"

MODEL_PATH     = MODELS_DIR  / "best_pipeline.joblib"
METADATA_PATH  = MODELS_DIR  / "model_metadata.json"
TEST_METRICS_PATH = RESULTS_DIR / "test_metrics.json"
CV_CSV_PATH    = RESULTS_DIR / "model_comparison_cv.csv"

# Confusion-matrix PNGs saved by train.py  (one per model)
CONFUSION_MATRIX_IMAGES = {
    "Random Forest":      RESULTS_DIR / "confusion_matrix_random_forest.png",
    "Decision Tree":      RESULTS_DIR / "confusion_matrix_decision_tree.png",
    "Logistic Regression": RESULTS_DIR / "confusion_matrix_logistic_regression.png",
}

# Feature columns expected by the pipeline (must match preprocessing.py)
NUMERIC_FEATURES    = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
CATEGORICAL_FEATURES = ["Type"]
ALL_FEATURES         = NUMERIC_FEATURES + CATEGORICAL_FEATURES   # order matters!


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading trained pipeline …")
def load_pipeline():
    """Load the best trained sklearn Pipeline from disk (cached across reruns)."""
    import joblib  # import here to keep the top-level imports light
    return joblib.load(MODEL_PATH)


def load_json(path: Path) -> dict | None:
    """Read a JSON file and return its contents, or None if the file is missing."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        st.warning(f"⚠️ Could not parse `{path.name}`: {exc}")
        return None


def fmt(value: float, pct: bool = True) -> str:
    """Format a metric value as a percentage string (e.g. 97.95 %)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"{value * 100:.2f} %" if pct else f"{value:.4f}"


def metric_card(label: str, value: str, help_text: str = "") -> None:
    """Render a single metric using st.metric (compact, highlighted)."""
    st.metric(label=label, value=value, help=help_text)


# ---------------------------------------------------------------------------
# Custom CSS — clean, professional look
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
        /* ── Main container padding ── */
        .main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

        /* ── Section header styling ── */
        h2 { color: #1f4e79; border-bottom: 2px solid #1f4e79; padding-bottom: 0.25rem; }
        h3 { color: #2e75b6; }

        /* ── Metric card emphasis ── */
        div[data-testid="metric-container"] {
            background: #f0f6ff;
            border: 1px solid #c9d9f0;
            border-radius: 8px;
            padding: 0.6rem 0.8rem;
        }

        /* ── Info / warning boxes ── */
        .stAlert { border-radius: 8px; }

        /* ── Prediction result boxes ── */
        .pred-failure {
            background: #fff0f0;
            border: 2px solid #d32f2f;
            border-radius: 10px;
            padding: 1rem 1.5rem;
            font-size: 1.1rem;
            color: #b71c1c;
        }
        .pred-ok {
            background: #f0fff4;
            border: 2px solid #388e3c;
            border-radius: 10px;
            padding: 1rem 1.5rem;
            font-size: 1.1rem;
            color: #1b5e20;
        }
        .pred-proba {
            background: #fffde7;
            border: 1px solid #f9a825;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            margin-top: 0.75rem;
            font-size: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ===========================================================================
# SIDEBAR — project info & quick nav
# ===========================================================================

with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg",
        width=90,
    )
    st.title("🔧 Predictive Maintenance")
    st.caption("AI4I 2020 Dataset  ·  ML Pipeline")
    st.divider()

    st.markdown("**Navigate to:**")
    st.markdown("- 📊 Model Metrics")
    st.markdown("- 📈 Model Comparison")
    st.markdown("- 🖼️ Confusion Matrices")
    st.markdown("- 🤖 Live Prediction")
    st.divider()

    # Show pipeline load status in sidebar
    if MODEL_PATH.exists():
        st.success("✅ Pipeline loaded")
    else:
        st.error(f"❌ Pipeline not found:\n`{MODEL_PATH}`")

    if METADATA_PATH.exists():
        meta = load_json(METADATA_PATH)
        if meta:
            st.info(f"**Best model:** {meta.get('best_model_name', '—')}")
            st.caption(f"Selection criterion: {meta.get('selection_criterion', '—')}")
    else:
        st.warning("⚠️ `model_metadata.json` not found.")


# ===========================================================================
# MAIN CONTENT
# ===========================================================================

# ── Page title ──────────────────────────────────────────────────────────────
st.title("🔧 Predictive Maintenance Using Machine Learning")
st.markdown(
    "An end-to-end ML dashboard for predicting machine failure on the "
    "[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601)."
)
st.divider()


# ===========================================================================
# SECTION 1 — Best-model metrics
# ===========================================================================

st.header("📊 Best Model — Evaluation Metrics")

metadata    = load_json(METADATA_PATH)
test_metrics_all = load_json(TEST_METRICS_PATH)

if metadata is None:
    st.error(
        "⚠️ `models/model_metadata.json` not found. "
        "Please run `python -m src.train` to generate it."
    )
else:
    best_model_name = metadata.get("best_model_name", "Unknown")
    best_metrics    = metadata.get("test_metrics", {})

    st.subheader(f"🏆 Selected Model: {best_model_name}")
    st.caption(
        f"**Selection criterion:** {metadata.get('selection_criterion', '—')}  |  "
        f"**CV folds:** {metadata.get('cv_folds', '—')}  |  "
        f"**Random state:** {metadata.get('random_state', '—')}"
    )

    # ── 6 metric cards in one row ───────────────────────────────────────────
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        metric_card(
            "Accuracy",
            fmt(best_metrics.get("accuracy")),
            "Overall fraction of correct predictions on the held-out test set.",
        )
    with col2:
        metric_card(
            "Precision",
            fmt(best_metrics.get("precision")),
            "Of all predicted failures, how many were real failures? "
            "(TP / (TP + FP))",
        )
    with col3:
        metric_card(
            "Recall",
            fmt(best_metrics.get("recall")),
            "Of all actual failures, how many did the model catch? "
            "(TP / (TP + FN))",
        )
    with col4:
        metric_card(
            "F1-Score",
            fmt(best_metrics.get("f1")),
            "Harmonic mean of Precision and Recall. "
            "Useful when both false positives and false negatives matter.",
        )
    with col5:
        metric_card(
            "PR-AUC",
            fmt(best_metrics.get("pr_auc")),
            "Area under the Precision-Recall curve. "
            "The primary selection criterion for this imbalanced dataset.",
        )
    with col6:
        metric_card(
            "ROC-AUC",
            fmt(best_metrics.get("roc_auc")),
            "Area under the ROC curve. "
            "Measures the model's ability to separate the two classes.",
        )

st.divider()


# ===========================================================================
# SECTION 2 — Model comparison (cross-validation results)
# ===========================================================================

st.header("📈 Model Comparison — Cross-Validation Results")

if not CV_CSV_PATH.exists():
    st.error(
        "⚠️ `results/model_comparison_cv.csv` not found. "
        "Please run `python -m src.train` to generate it."
    )
else:
    cv_df = pd.read_csv(CV_CSV_PATH)

    # ── Pretty display table ────────────────────────────────────────────────
    # Keep the mean columns (mean ± std would be very wide); show std in a tooltip
    display_cols = {
        "Model":                    "Model",
        "accuracy_mean":            "Accuracy (mean)",
        "precision_mean":           "Precision (mean)",
        "recall_mean":              "Recall (mean)",
        "f1_mean":                  "F1 (mean)",
        "roc_auc_mean":             "ROC-AUC (mean)",
        "average_precision_mean":   "PR-AUC (mean) ★",
    }

    # Keep only columns that actually exist in the CSV
    existing_cols = [c for c in display_cols if c in cv_df.columns]
    cv_display = cv_df[existing_cols].copy()
    cv_display = cv_display.rename(columns=display_cols)

    # Format numeric columns as percentages
    numeric_cols_display = [v for k, v in display_cols.items() if k != "Model"]
    for col in numeric_cols_display:
        if col in cv_display.columns:
            cv_display[col] = cv_display[col].apply(
                lambda x: f"{x * 100:.2f} %" if pd.notna(x) else "N/A"
            )

    st.markdown(
        "5-fold stratified cross-validation on the **training set only**. "
        "★ PR-AUC is the primary model-selection criterion (robust to class imbalance)."
    )
    st.dataframe(cv_display, use_container_width=True, hide_index=True)

    # ── Also show the full table with std columns ────────────────────────────
    with st.expander("🔍 Show full CV results (mean ± std for all metrics)"):
        st.dataframe(cv_df.round(4), use_container_width=True, hide_index=True)

    # ── Also show test-set metrics for all models ────────────────────────────
    if test_metrics_all:
        st.markdown("#### Test-Set Metrics (all three models)")
        st.caption("Reported for reference only. Model selection was based on CV, not the test set.")
        test_df = pd.DataFrame(test_metrics_all).T.reset_index()
        test_df = test_df.rename(columns={
            "index":   "Model",
            "accuracy":  "Accuracy",
            "precision": "Precision",
            "recall":    "Recall",
            "f1":        "F1",
            "roc_auc":   "ROC-AUC",
            "pr_auc":    "PR-AUC",
        })
        # Format as percentages
        for col in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]:
            if col in test_df.columns:
                test_df[col] = test_df[col].apply(
                    lambda x: f"{x * 100:.2f} %" if pd.notna(x) else "N/A"
                )
        st.dataframe(test_df, use_container_width=True, hide_index=True)

st.divider()


# ===========================================================================
# SECTION 3 — Confusion matrices
# ===========================================================================

st.header("🖼️ Confusion Matrices (Test Set)")
st.caption(
    "Each matrix shows how the model classified the held-out test examples. "
    "Rows = true labels, Columns = predicted labels."
)

available_images = {
    name: path for name, path in CONFUSION_MATRIX_IMAGES.items() if path.exists()
}

if not available_images:
    st.error(
        "⚠️ No confusion-matrix PNG files found in `results/`. "
        "Please run `python -m src.train` to generate them."
    )
else:
    img_cols = st.columns(len(available_images))
    for col, (model_name, img_path) in zip(img_cols, available_images.items()):
        with col:
            is_best = (metadata is not None and model_name == metadata.get("best_model_name"))
            label = f"**{model_name}**" + (" 🏆 *(selected)*" if is_best else "")
            st.markdown(label)
            st.image(str(img_path), use_container_width=True)

    if len(available_images) < len(CONFUSION_MATRIX_IMAGES):
        missing = set(CONFUSION_MATRIX_IMAGES) - set(available_images)
        st.warning(
            f"⚠️ Confusion matrices not found for: {', '.join(missing)}. "
            "Re-run `python -m src.train` to regenerate all images."
        )

st.divider()


# ===========================================================================
# SECTION 4 — Live prediction
# ===========================================================================

st.header("🤖 Live Prediction")
st.markdown(
    "Enter sensor readings below and click **Predict** to run the trained "
    f"**{metadata.get('best_model_name', 'best') if metadata else 'best'}** pipeline."
)

if not MODEL_PATH.exists():
    st.error(
        f"⚠️ Trained pipeline not found at `{MODEL_PATH}`. "
        "Please run `python -m src.train` to train and save the model."
    )
else:
    # Try to load the pipeline (cached)
    pipeline = None
    try:
        pipeline = load_pipeline()
    except Exception as exc:
        st.error(
            f"❌ Failed to load the pipeline from `{MODEL_PATH}`.\n\n"
            f"**Error:** `{exc}`\n\n"
            "Make sure scikit-learn and joblib versions match those used during training."
        )

    if pipeline is not None:
        # ── Input form ──────────────────────────────────────────────────────
        with st.form("prediction_form"):
            st.subheader("Enter Machine Sensor Readings")

            form_col1, form_col2 = st.columns(2)

            with form_col1:
                machine_type = st.selectbox(
                    "Machine Type",
                    options=["L", "M", "H"],
                    index=0,
                    help=(
                        "Product quality variant: "
                        "L = Low (50 %), M = Medium (30 %), H = High (20 %)."
                    ),
                )
                air_temp = st.number_input(
                    "Air Temperature [K]",
                    min_value=290.0,
                    max_value=310.0,
                    value=298.1,
                    step=0.1,
                    format="%.1f",
                    help="Ambient air temperature in Kelvin. Typical range: 295 – 304 K.",
                )
                process_temp = st.number_input(
                    "Process Temperature [K]",
                    min_value=300.0,
                    max_value=320.0,
                    value=308.6,
                    step=0.1,
                    format="%.1f",
                    help="Process (coolant) temperature in Kelvin. Typically 10 K above air temp.",
                )

            with form_col2:
                rot_speed = st.number_input(
                    "Rotational Speed [rpm]",
                    min_value=1000,
                    max_value=3000,
                    value=1551,
                    step=1,
                    help="Spindle rotational speed in revolutions per minute. Typical: 1168 – 2886 rpm.",
                )
                torque = st.number_input(
                    "Torque [Nm]",
                    min_value=1.0,
                    max_value=100.0,
                    value=42.8,
                    step=0.1,
                    format="%.1f",
                    help="Torque applied to the spindle in Newton-metres. Typical: 3.8 – 76.6 Nm.",
                )
                tool_wear = st.number_input(
                    "Tool Wear [min]",
                    min_value=0,
                    max_value=300,
                    value=0,
                    step=1,
                    help=(
                        "Cumulative tool-wear time in minutes. "
                        "Increases by 2–5 min per cycle depending on machine type. "
                        "Typical max: ~250 min."
                    ),
                )

            submitted = st.form_submit_button(
                "🔮 Predict Machine Failure",
                use_container_width=True,
                type="primary",
            )

        # ── Prediction logic ─────────────────────────────────────────────────
        if submitted:
            # Build a one-row DataFrame with the same column names as training data
            input_data = pd.DataFrame(
                [{
                    "Air temperature [K]":      air_temp,
                    "Process temperature [K]":   process_temp,
                    "Rotational speed [rpm]":    rot_speed,
                    "Torque [Nm]":               torque,
                    "Tool wear [min]":           tool_wear,
                    "Type":                      machine_type,
                }]
            )

            st.markdown("#### Prediction Result")

            try:
                prediction = pipeline.predict(input_data)[0]  # 0 = no failure, 1 = failure

                # ── Probability (if the pipeline supports it) ─────────────────
                failure_prob = None
                if hasattr(pipeline, "predict_proba"):
                    try:
                        proba = pipeline.predict_proba(input_data)[0]
                        failure_prob = proba[1]   # probability of class 1 (failure)
                    except Exception:
                        failure_prob = None

                # ── Display result ────────────────────────────────────────────
                if prediction == 1:
                    st.markdown(
                        "<div class='pred-failure'>"
                        "⚠️ <b>Machine Failure Predicted</b><br>"
                        "The model predicts a <b>failure</b> is likely under these conditions. "
                        "Consider scheduling preventive maintenance."
                        "</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<div class='pred-ok'>"
                        "✅ <b>No Failure Predicted</b><br>"
                        "The model predicts the machine will operate <b>normally</b> "
                        "under these conditions."
                        "</div>",
                        unsafe_allow_html=True,
                    )

                if failure_prob is not None:
                    pct = failure_prob * 100
                    bar_color = "#d32f2f" if failure_prob >= 0.5 else "#388e3c"
                    st.markdown(
                        f"<div class='pred-proba'>"
                        f"📊 <b>Failure probability:</b> {pct:.1f} %"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    st.progress(
                        float(failure_prob),
                        text=f"Failure probability: {pct:.1f} %",
                    )

                # ── Show the submitted values ─────────────────────────────────
                with st.expander("📋 Submitted input values"):
                    st.dataframe(input_data, use_container_width=True, hide_index=True)

            except Exception as exc:
                st.error(
                    f"❌ Prediction failed.\n\n"
                    f"**Error:** `{exc}`\n\n"
                    "Make sure the pipeline was trained with the same scikit-learn version. "
                    "If you recently updated packages, re-run `python -m src.train`."
                )


# ===========================================================================
# FOOTER
# ===========================================================================

st.divider()
st.markdown(
    "<div style='text-align:center; color:#888; font-size:0.85rem;'>"
    "Predictive Maintenance · AI4I 2020 Dataset · "
    "Built with <a href='https://streamlit.io' target='_blank'>Streamlit</a> · "
    "Model: scikit-learn Pipeline (Random Forest)"
    "</div>",
    unsafe_allow_html=True,
)

