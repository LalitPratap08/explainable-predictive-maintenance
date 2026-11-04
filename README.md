# Explainable Machine Learning for Predictive Maintenance

> **Academic prototype** — Predictions from this system do not replace qualified engineering or maintenance decisions.

## Project Overview

This project builds a software-only Machine Learning application that predicts equipment failure from historical sensor and operational data, and explains those predictions using Explainable AI (XAI) techniques.

The goal is to demonstrate how modern ML interpretability tools (SHAP) can make black-box models more trustworthy and actionable in an industrial maintenance context — without requiring physical equipment, IoT hardware, or real-time sensor streams.

---

## Dataset

**AI4I 2020 Predictive Maintenance Dataset**
- **Source**: UCI Machine Learning Repository — <https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset>
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Records**: 10,000 synthetic data points modelled on real industrial equipment
- **Target column**: `Machine failure` (binary: 0 = no failure, 1 = failure)
- **Class imbalance**: ~3.4% positive (failure) class — addressed with class-weight balancing and PR-AUC evaluation

### Citation
> Stephan Matzka. (2020). AI4I 2020 Predictive Maintenance Dataset. UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C

### How to obtain the dataset
The dataset must be placed manually if automatic download fails:

1. Visit <https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset>
2. Download `ai4i2020.csv`
3. Place it at `data/raw/ai4i2020.csv`

---

## Technology Stack

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10+ | Core language |
| Pandas | ≥2.0 | Data manipulation |
| NumPy | ≥1.24 | Numerical operations |
| Scikit-learn | ≥1.3 | ML pipeline, models, evaluation |
| SHAP | ≥0.44 | Explainable AI |
| Matplotlib / Seaborn | ≥3.7 / ≥0.12 | Static visualisations |
| Plotly | ≥5.18 | Interactive charts in dashboard |
| Streamlit | ≥1.32 | Web dashboard |
| Joblib | ≥1.3 | Model persistence |
| Pytest | ≥7.4 | Automated testing |
| Jupyter | ≥1.0 | Experimental notebooks |

---

## Project Structure

```
explainable-predictive-maintenance/
├── data/
│   ├── raw/            ← Place ai4i2020.csv here
│   └── processed/      ← Generated after preprocessing
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_explainability.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing.py   ← Data loading, validation, feature engineering
│   ├── train.py           ← Model training and comparison
│   ├── predict.py         ← Inference on new readings
│   └── explain.py         ← SHAP global and local explanations
├── dashboard/
│   └── app.py             ← Streamlit dashboard
├── models/                ← Saved pipeline artifacts (.joblib)
├── results/               ← Plots, metrics, experiment outputs
├── tests/                 ← Pytest test suite
├── requirements.txt
├── .gitignore
├── README.md
└── LICENSE
```

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Maaz4484/Zero-Deps.git
cd explainable-predictive-maintenance
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Place the dataset
Download `ai4i2020.csv` from UCI (link above) and copy it to:
```
data/raw/ai4i2020.csv
```

---

## ML Pipeline

The pipeline is implemented in stages to prevent data leakage:

1. **Load & Validate** — schema checks, duplicate removal, type validation
2. **Exploratory Data Analysis** — class distribution, correlations, feature histograms
3. **Preprocessing** — StandardScaler for numerics, OneHotEncoder for categoricals; fitted on training split only
4. **Train / Test Split** — stratified 80/20 split, `random_state=42`
5. **Model Training & Comparison**:
   - Logistic Regression (baseline)
   - Decision Tree
   - Random Forest *(primary candidate)*
6. **Evaluation** — F1, Recall, Precision, ROC-AUC, PR-AUC, Confusion Matrix
7. **Model Selection** — based on validation PR-AUC (not test-set performance)
8. **SHAP Explainability** — global feature importance + local per-prediction explanations
9. **Artifact Saving** — full sklearn Pipeline saved via Joblib

---

## Evaluation Strategy

Given the severe class imbalance (~3.4% positives), evaluation focuses on:

- **PR-AUC** — primary selection metric (robust to imbalance)
- **Recall** — priority metric (missed failures are costly)
- **F1-score** — balanced view of precision and recall
- **ROC-AUC** — secondary metric
- **Confusion matrix** — to inspect false positive / false negative trade-off

> Accuracy alone is **not** used for model selection.

---

## Explainability

SHAP (SHapley Additive exPlanations) is used to explain model predictions:

- **Global**: Feature importance bar plot, SHAP summary plot
- **Local**: Waterfall plot for individual predictions showing feature contributions

**Important note**: Feature contribution values indicate statistical associations learned from the training data. They do not establish causal relationships between features and equipment failures.

---

## Running the Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard requires trained model artifacts in `models/`. Run the training pipeline first.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Reproducing Experiments

```bash
# Step 1 — Preprocess data
python -m src.preprocessing

# Step 2 — Train and compare models
python -m src.train

# Step 3 — Generate SHAP explanations
python -m src.explain

# Step 4 — Launch dashboard
streamlit run dashboard/app.py
```

---

## Implementation Status

| Stage | Description | Status |
|-------|-------------|--------|
| 1 | Repository inspection and planning | ✅ Complete |
| 2 | Project structure, requirements, .gitignore | ✅ Complete |
| 3 | Dataset loading, EDA, preprocessing | 🔄 In Progress — `src/preprocessing.py` complete, tests written |
| 4 | Model training, evaluation, artifact saving | 🔄 In Progress — `src/train.py` complete, tests written |
| 5 | SHAP explainability and experiment outputs | 🔲 Pending |
| 6 | Streamlit dashboard | 🔲 Pending |
| 7 | Tests, validation, final documentation | 🔲 Pending |

---

## Known Limitations

- The AI4I 2020 dataset is **synthetically generated** and may not reflect all real-world failure patterns.
- With only ~340 positive samples, model generalisation is limited.
- SHAP explanations for tree ensembles are exact; for linear models they are approximate.
- No real-time sensor integration — this is a batch-prediction prototype.
- Early-warning prediction (time-to-failure) is not implemented; only binary failure classification.
- Results should be interpreted in the context of the training distribution only.

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

The AI4I 2020 dataset is licensed under CC BY 4.0 — attribution required.

---



