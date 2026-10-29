# Australian Rain Prediction App 🌧️

An end-to-end Machine Learning project that predicts whether it will rain tomorrow in Australia based on today's meteorological observations. The repository includes an experimental modelling workflow, a scikit-learn preprocessing and prediction pipeline, and an interactive Streamlit application.

## ✨ Project Overview

- **Goal:** Build and evaluate a binary classification model to predict `RainTomorrow` (`Yes`/`No`).
- **Dataset:** Approximately 10 years of daily weather observations from Australian weather stations, sourced from Kaggle.
- **Target Variable:** `RainTomorrow` (binary classification).

## 🛠️ Methodology

The project includes two modelling workflows.

### 1. Manual Preprocessing Workflow (Experimental)

- **Data Splitting:** Chronological split by year:
  - Training set: before 2015
  - Validation set: 2015
  - Test set: after 2015
- **Preprocessing:** Missing-value imputation (`SimpleImputer`), numerical scaling (`MinMaxScaler`), and categorical encoding (`OneHotEncoder`).
- **Artifacts:** Processed data splits are stored as `.parquet` files, and the trained model is saved as `models/aussie_rain.joblib`.

This workflow is retained for experimentation and model comparison.

### 2. Scikit-Learn Pipeline Workflow (Used by the App)

The recommended workflow combines preprocessing and classification into a single scikit-learn pipeline.

**Numerical Features:**
- `SimpleImputer(strategy="mean")` replaces missing values with the mean calculated from the training data.
- `MinMaxScaler()` scales numerical features.

**Categorical Features:**
- `SimpleImputer(strategy="constant", fill_value="missing")` replaces missing values with the constant category `"missing"`.
- `OneHotEncoder(handle_unknown="ignore")` converts categorical features into numerical representations and handles previously unseen categories.

**Classifier:**
- `LogisticRegression(solver="liblinear")`.

**Model Artifact:**
- The complete fitted pipeline is serialized as `models/aussie_rain_pipeline.joblib`.

Combining preprocessing and classification ensures that the same fitted transformations are applied during training and inference. The application loads the complete pipeline, so preprocessing does not need to be performed separately in `app.py`.

## 💻 Interactive Streamlit Application

The project includes an interactive web application built with **Streamlit** (`app.py`).

Users can:

- Enter weather observations, including temperature, humidity, rainfall, wind speed, cloud cover, and atmospheric pressure.
- Select a weather station location and wind directions.
- Leave numerical fields empty when a value is unknown.
- Select **"Don't know"** for categorical fields when information is unavailable.
- Submit the weather observations to the trained machine learning pipeline.
- View the predicted rain/no-rain class and the estimated probability of rain tomorrow.

### Missing Values Handling

The application allows users to leave unknown numerical values empty or select `"Don't know"` for categorical fields.

Before prediction, categorical `"Don't know"` selections are converted to missing values (`None`).

The saved pipeline handles missing values automatically:

- Numerical missing values are replaced with the mean using `SimpleImputer(strategy="mean")`.
- Categorical missing values are replaced with `"missing"` using `SimpleImputer(strategy="constant", fill_value="missing")`.

The application passes the original input features directly to the pipeline. Scaling and one-hot encoding are performed internally by the fitted preprocessing steps.

> **Note:** The prediction is a machine-learning estimate and is not an official weather-service forecast.

## 📂 Repository Structure

```text
STREAMLIT-AU-RAIN-FORECAST/
├── data/
│   ├── weatherAUS.csv
│   └── *.parquet
├── models/
│   ├── aussie_rain.joblib
│   └── aussie_rain_pipeline.joblib
├── notebooks/
│   └── AU-Rain_Prediction-Model.ipynb
├── app.py
├── requirements.txt
└── runtime.txt
```

### Main Files

- `data/weatherAUS.csv` — raw Australian weather dataset.
- `data/*.parquet` — processed data splits used by the manual preprocessing workflow.
- `models/aussie_rain.joblib` — model artifact from the experimental workflow.
- `models/aussie_rain_pipeline.joblib` — complete fitted preprocessing and prediction pipeline used by the Streamlit application.
- `notebooks/AU-Rain_Prediction-Model.ipynb` — Jupyter Notebook containing the data analysis, preprocessing, model training, and evaluation workflows.
- `app.py` — Streamlit web application.
- `requirements.txt` — Python package dependencies.
- `runtime.txt` — Python runtime specification, if required by the deployment platform.

The application expects the trained pipeline to be available at:

`models/aussie_rain_pipeline.joblib`

## 🚀 Quick Start & Installation

### Prerequisites

- Python 3.12
- Git
- Conda or Python virtual environment
- Trained pipeline file: `models/aussie_rain_pipeline.joblib`

### 1. Clone the Repository

Replace `<YOUR_REPOSITORY_URL>` with the actual URL of your GitHub repository.

```bash
git clone <YOUR_REPOSITORY_URL>
cd streamlit-AU-rain-forecast
```

If you have already cloned the repository, open the project folder in VS Code and continue from the next step.

### 2. Create and Activate a Virtual Environment

#### Option A: Conda (Used During Development)

Create the environment:

```bash
conda create --name ds-course python=3.12 -y
```

Activate it:

```bash
conda activate ds-course
```

If the `ds-course` environment already exists, simply activate it:

```bash
conda activate ds-course
```

#### Option B: Python Virtual Environment

On Linux, macOS, or WSL:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Choose one environment setup option, not both.

### 3. Install Dependencies

Make sure the intended environment is activated and that you are in the project root directory (the folder containing `requirements.txt`).

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

### 4. Run the Streamlit Application

Start the application:

```bash
python -m streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open this URL in your web browser to access the application.

## 🔧 Troubleshooting

**Streamlit command not found**

Make sure the correct environment is activated and install the dependencies:

```bash
conda activate ds-course
python -m pip install -r requirements.txt
```

You can launch Streamlit using:

```bash
python -m streamlit run app.py
```

**Pipeline model not found**

Make sure the following file exists:

```text
models/aussie_rain_pipeline.joblib
```

**Prediction or feature mismatch error**

Ensure that the saved pipeline was trained using the same feature names and input format expected by `app.py`.

**Missing values handling**

Ensure that the saved pipeline includes the numerical and categorical imputers described in the preprocessing section.

## 📊 Tech Stack

- **Programming Language:** Python 3.12
- **Web Application:** Streamlit
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Model Serialization:** Joblib
- **Data Storage:** PyArrow, Parquet
- **Visualization:** Matplotlib, Seaborn