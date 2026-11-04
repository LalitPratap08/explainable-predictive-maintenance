# Predictive Maintenance Using Machine Learning

A machine learning project that predicts potential machine failures using operational sensor data. It compares multiple classification models and provides an interactive Streamlit dashboard for exploring model performance and making predictions.

## Project Overview

Unexpected machine failures can cause production delays, maintenance costs, and equipment downtime.

This project uses the **AI4I 2020 Predictive Maintenance Dataset** to train machine learning models that identify potential machine failures based on operating conditions.

The project includes data preprocessing, model training, evaluation, automated testing, and an interactive dashboard.

## Features

- Data cleaning and preprocessing
- Handling of imbalanced target classes
- Stratified train-test splitting
- Comparison of three machine learning models
- 5-fold stratified cross-validation
- Model evaluation using multiple classification metrics
- Confusion matrix visualisations
- Saved trained model for reuse
- Interactive Streamlit prediction dashboard
- Automated tests using Pytest

## Machine Learning Models

The following models are compared:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

Models are compared using **Precision, Recall, F1-score, ROC-AUC, and PR-AUC**.

Random Forest was selected as the best model based on cross-validation PR-AUC.

## Dataset

**AI4I 2020 Predictive Maintenance Dataset**

Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)

- 10,000 observations
- 14 original columns
- Target variable: `Machine failure`
- Failure cases represent approximately 3.4% of the dataset.

The raw dataset is not included in this repository. Download it from UCI and place `ai4i2020.csv` inside:

`data/raw/ai4i2020.csv`

## Model Performance

### Cross-validation results

| Model | Mean PR-AUC | Standard Deviation |
|---|---:|---:|
| Logistic Regression | 0.4409 | 0.0299 |
| Decision Tree | 0.5282 | 0.0484 |
| Random Forest | **0.7388** | 0.0367 |

### Test set results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8245 | 0.1418 | 0.8235 | 0.2419 | 0.9070 | 0.3818 |
| Decision Tree | 0.9585 | 0.4324 | 0.7059 | 0.5363 | 0.8454 | 0.5746 |
| Random Forest | **0.9795** | **0.7213** | 0.6471 | **0.6822** | **0.9612** | **0.7551** |

Random Forest achieved the highest test PR-AUC and was selected as the best overall model.

Because the dataset is imbalanced, accuracy alone is not sufficient to evaluate failure prediction performance.

## Project Structure

```text
explainable-predictive-maintenance/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── best_pipeline.joblib
│   └── model_metadata.json
│
├── notebooks/
│
├── results/
│   ├── confusion_matrix_decision_tree.png
│   ├── confusion_matrix_logistic_regression.png
│   ├── confusion_matrix_random_forest.png
│   ├── model_comparison_cv.csv
│   └── test_metrics.json
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   └── train.py
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   └── test_train.py
│
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/LalitPratap08/explainable-predictive-maintenance.git
cd explainable-predictive-maintenance
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download the AI4I 2020 dataset from the UCI Machine Learning Repository and place the CSV file at:

```text
data/raw/ai4i2020.csv
```

## Running the Project

### Train the models

```bash
python -m src.train
```

This runs preprocessing, model training, cross-validation, evaluation, and model saving.

### Launch the Streamlit dashboard

```bash
python -m streamlit run dashboard/app.py
```

Open the local URL displayed in your terminal, usually:

`http://localhost:8501`

### Run automated tests

```bash
python -m pytest tests/ -v
```

**Current test status:** 67 tests passed.

## Dashboard

The Streamlit dashboard provides:

- Model performance overview
- Cross-validation comparison
- Confusion matrices
- Interactive machine failure prediction

## Limitations

- The project uses the AI4I 2020 dataset, which is synthetic.
- Failure events are significantly less frequent than normal operating conditions.
- Model performance on this dataset does not guarantee equivalent performance on real industrial equipment.
- Predictions are intended for academic demonstration and should not be used as the sole basis for real maintenance decisions.

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib
- Pytest

## Author

**Lalit Pratap**

GitHub: [@LalitPratap08](https://github.com/LalitPratap08)

---

*Developed as an individual academic machine learning project.*