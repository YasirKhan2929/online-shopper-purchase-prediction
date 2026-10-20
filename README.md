# Online Shopper Purchase Prediction

An end-to-end machine learning project that predicts online shoppers' purchase intention and provides business insights through an interactive analytics dashboard.

## Project Overview

Online stores receive many visitor sessions, but not every visitor makes a purchase. This project explores shopping behaviour, trains a classification model, and provides an interface for estimating the likelihood of a purchase.

The project includes:
- Exploratory data analysis and data preparation
- Machine learning model training and evaluation
- Purchase probability predictions through a FastAPI REST API
- An interactive Streamlit prediction interface
- A Streamlit business analytics dashboard
- A demo online store that sends sample browsing-session data to the prediction API

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Joblib
- FastAPI
- Uvicorn
- Streamlit
- Requests

## Dataset

The project uses the Online Shoppers Purchasing Intention dataset.

- Total sessions: 12,330
- Sessions resulting in a purchase: 1,908
- Sessions without a purchase: 10,422
- Overall purchase rate: approximately 15.47%

The dataset includes browsing activity, page-view counts, session durations, visitor type, month, and other session attributes.

## Project Structure

```text
project1/
├── app.py
├── analytics.py
├── api.py
├── store_demo.html
├── data/
│   ├── raw/
│   │   └── online_shoppers_intention.csv
│   └── processed/
│       └── online_shoppers_prepared.csv
├── models/
│   └── purchase_prediction_model.joblib
├── notebooks/
├── src/
│   ├── 01_explore_data.py
│   ├── 02_analyze_data.py
│   ├── 03_prepare_data.py
│   ├── 04_train_model.py
│   ├── 05_compare_models.py
│   ├── 06_tune_threshold.py
│   ├── 07_final_evaluation.py
│   ├── 08_save_model.py
│   ├── 09_predict.py
│   ├── 10_compare_validation.py
│   ├── 11_final_model_comparison.py
│   ├── 12_investigate_pagevalues.py
│   └── 13_compare_pagevalues.py
└── tests/
```

## Installation

### 1. Clone the repository

```bash
git clone [https://github.com/YasirKhan2929/online-shopper-purchase-prediction.git]
cd project1
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install pandas numpy scikit-learn matplotlib joblib fastapi "uvicorn[standard]" streamlit requests
```

### 4. Prepare the required files

Place `online_shoppers_intention.csv` in `data/raw/` and ensure the trained model artifact is available at:

```text
models/purchase_prediction_model.joblib
```

## How to Run

Open separate VS Code terminals from the project root.

### Start the prediction API

```bash
python -m uvicorn api:app --reload
```

API documentation:

http://127.0.0.1:8000/docs

Health check:

http://127.0.0.1:8000/health

### Open the demo online store

http://127.0.0.1:8000/store

The demo store sends sample browsing-session information to the prediction API. Its session measurements are illustrative and should not be treated as real customer analytics.

### Start the prediction interface

```bash
python -m streamlit run app.py
```

### Start the business analytics dashboard

```bash
python -m streamlit run analytics.py
```

If port 8501 is already occupied, use:

```bash
python -m streamlit run analytics.py --server.port 8502
```

## Model Evaluation

The saved Logistic Regression model uses a classification threshold of 0.60.

Previously recorded held-out test results:

| Metric | Result |
|---|---:|
| Precision | Approximately 0.30 |
| Recall | Approximately 0.56 |
| F1-score | Approximately 0.39 |
| ROC-AUC | Approximately 0.73 |

These results indicate that the model identifies some purchasing sessions but also produces false-positive predictions. It should not be considered production-ready without further evaluation and improvement.

## Important Considerations

- A predicted purchase probability is an estimate, not a guarantee.
- The selected classification threshold affects precision and recall.
- Historical dataset patterns do not necessarily represent current shoppers.
- The demo store uses illustrative session measurements; it does not collect or analyse actual customer traffic.
- Feature availability must be checked before using the model in a real store.
- Further work should include testing, monitoring, and evaluating the model on newer data.

## Future Improvements

- Add automated tests for API endpoints and input validation.
- Compare additional classification models and improve prediction quality.
- Add model-performance monitoring.
- Connect the demo store to persistent session storage.
- Deploy the API and dashboards to a hosting platform.

## Author

Yasir Khan

Machine Learning | Data Analytics | Python | AI Applications
