# Australian Rain Prediction App 🌧️

An end-to-end Machine Learning project that predicts whether it will rain tomorrow in Australia based on today's meteorological data. This repository showcases the evolution from experimental data science to production-ready ML engineering and includes an interactive web interface.

## ✨ Project Overview

* **Goal:** Build and evaluate a binary classification model to predict `RainTomorrow` (Yes/No).
* **Dataset:** ~10 years of daily weather observations from various Australian meteorological stations (sourced from Kaggle).
* **Target Variable:** `RainTomorrow` (Binary classification).

---

## 🛠️ Two-Approach Methodology

To demonstrate best practices in ML engineering, this project implements and compares two distinct workflows:

### 1. Manual Feature Engineering Workflow (Experimental)
* **Data Splitting:** Chronological split based on years (Train: <2015, Validation: 2015, Test: >2015) to prevent temporal data leakage.
* **Preprocessing:** Manual handling of missing data (`SimpleImputer`), numerical scaling (`MinMaxScaler`), and categorical encoding (`OneHotEncoder`) executed independently across data splits.
* **Artifacts:** Intermediate processed data is stored as lightweight `.parquet` files, and the standalone model is saved as `aussie_rain.joblib`.

### 2. Production-Ready Pipeline Workflow (Recommended)
* **Architecture:** Fully refactored using Scikit-Learn's `Pipeline` and `ColumnTransformer`.
* **Encapsulation:** Combines all preprocessing stages and the `LogisticRegression(solver='liblinear')` classifier into a single, immutable pipeline asset.
* **Benefits:** Complete elimination of data leakage risks, robust inference handling for unknown categories, and simplified model deployment.
* **Artifacts:** Serialized as a single, clean production asset: `aussie_rain_pipeline.joblib`.

---

## 💻 Interactive Streamlit Interface

The project features a web application built with **Streamlit** (`app.py`) that allows users to:
* Input local weather metrics (Temperature, Humidity, Wind Speed, Rain Status, etc.) via interactive web components.
* Trigger instant model inference using the pre-trained production pipeline (`aussie_rain_pipeline.joblib`).
* View the rain prediction alongside its calculated probability score.

---

## 📂 Repository Structure

```text
STREAMLIT-AU-RAIN-FORECAST/
├── data/
│   ├── weatherAUS.csv                 # Raw meteorological dataset
│   └── *.parquet                      # Processed train/val/test splits (Manual workflow)
├── models/
│   ├── aussie_rain.joblib             # Manual workflow trained model
│   └── aussie_rain_pipeline.joblib    # Production pipeline (Pre-processor + Classifier)
├── notebooks/
│   └── AU-Rain_Prediction-Model.ipynb # Jupyter Notebook containing both workflows
├── app.py                             # Streamlit web application (Root execution)
├── requirements.txt                   # Python package dependencies
└── runtime.txt                        # Python version specification (3.12)
```

---

## 🚀 Quick Start & Installation

### Prerequisites
* **Python 3.12**

### 1. Clone the repository
```bash
git clone https://github.com
cd streamlit-au-rain-forecast
```

### 2. Set up a virtual environment & Install dependencies
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the Streamlit Application
```bash
streamlit run app.py
```

---

## 📊 Tech Stack

* **Language & UI:** Python 3.12, Streamlit
* **ML & Data Science:** Scikit-Learn, Pandas, NumPy, PyArrow, Joblib
* **Visualization:** Matplotlib, Seaborn