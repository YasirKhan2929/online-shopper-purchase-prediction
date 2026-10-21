# Online Shopper Purchase Prediction

An end-to-end machine learning project that predicts online shoppers' purchase intention and provides business insights through interactive dashboards.

The project combines data exploration, model training and evaluation, a prediction API, a Streamlit interface, a business analytics dashboard, and a demo online store.

## Project Overview

Online stores receive many visitor sessions, but not every visitor makes a purchase. This project explores shopping behaviour, trains a classification model, and provides an interface for estimating the likelihood of a purchase based on session attributes.

Key features:

- Exploratory data analysis and data preparation
- Machine learning model training and evaluation
- Purchase probability predictions through a FastAPI REST API
- An interactive Streamlit prediction interface
- A Streamlit business analytics dashboard
- A demo online store that sends sample browsing-session data to the prediction API
- Automated API tests for endpoint behaviour, input validation, and error handling

## Technology Stack

- **Language:** Python
- **Data processing:** Pandas, NumPy
- **Machine learning:** Scikit-learn
- **Visualization:** Matplotlib
- **Model persistence:** Joblib
- **API:** FastAPI, Uvicorn
- **User interfaces and dashboards:** Streamlit
- **HTTP requests:** Requests
- **Testing:** Pytest, FastAPI TestClient

## Dataset

The project uses the **Online Shoppers Purchasing Intention** dataset.

| Dataset statistic | Value |
|---|---:|
| Total sessions | 12,330 |
| Sessions resulting in a purchase | 1,908 |
| Sessions without a purchase | 10,422 |
| Overall purchase rate | Approximately 15.47% |

The dataset includes browsing activity, page-view counts, session durations, visitor type, month, and other session attributes.

The saved prediction model uses 16 input features. `PageValues` is excluded from the saved model to avoid relying on a feature that may introduce target leakage or may not be available at prediction time.

## Project Structure

```text
project1/
├── app.py
├── analytics.py
├── api.py
├── store_demo.html
├── README.md
├── .gitignore
├── data/
│   ├── raw/
│   │   └── online_shoppers_intention.csv
│   └── processed/
│       └── online_shoppers_prepared.csv
├── documentation/
├── models/
│   └── purchase_prediction_model.joblib
├── notebooks/
├── screenshots/
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
    └── test_api.py
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YasirKhan2929/online-shopper-purchase-prediction.git
cd online-shopper-purchase-prediction
```

If your local project folder has a different name, change into that folder before running the commands below.

### 2. Create and activate a virtual environment

Create the environment:

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation because of the execution policy, consult the official Python/PowerShell guidance for your environment or use Command Prompt activation instead:

```bat
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
python -m pip install pandas numpy scikit-learn matplotlib joblib fastapi "uvicorn[standard]" streamlit requests pytest httpx
```

### 4. Check required project files

Ensure the dataset is available at:

```text
data/raw/online_shoppers_intention.csv
```

Ensure the saved model artifact is available at:

```text
models/purchase_prediction_model.joblib
```

The prediction API loads the saved model when it starts. If the model file is missing or invalid, the API will not start successfully.

## How to Run

Run the API and each Streamlit app in separate terminals from the project root. Activate the same virtual environment in each terminal if you created one.

### Start the prediction API

```bash
python -m uvicorn api:app --reload
```

API root:

http://127.0.0.1:8000/

Interactive API documentation:

http://127.0.0.1:8000/docs

Health check:

http://127.0.0.1:8000/health

### Open the demo online store

With the API running, open:

http://127.0.0.1:8000/store

The demo store sends sample browsing-session information to the prediction API. Its session measurements are illustrative and should not be treated as real customer analytics.

### Start the prediction interface

In a separate terminal:

```bash
python -m streamlit run app.py
```

### Start the business analytics dashboard

In another terminal:

```bash
python -m streamlit run analytics.py
```

If port `8501` is already occupied, choose another port:

```bash
python -m streamlit run analytics.py --server.port 8502
```

## Automated Testing

The project includes automated API tests covering:

- API health and home endpoints
- Valid purchase predictions
- Missing required fields
- Invalid bounce-rate and exit-rate values
- Negative page counts and durations
- Invalid month and visitor-type categories
- Invalid input types
- Consistent predictions for identical inputs
- Safe handling of unexpected prediction errors
- Availability of the API documentation

Run the tests from the project root:

```bash
python -m pytest tests/test_api.py -v
```

**Latest verified result:** 15 tests passed.

A Starlette deprecation warning related to the test client was present during that run; all tests passed despite the warning. The warning is worth reviewing during future dependency maintenance.

## Model Evaluation

The saved Logistic Regression model uses a classification threshold of `0.60`.

Previously recorded held-out test results:

| Metric | Result |
|---|---:|
| Precision | Approximately 0.30 |
| Recall | Approximately 0.56 |
| F1-score | Approximately 0.39 |
| ROC-AUC | Approximately 0.73 |

These are previously recorded results and have not been recalculated as part of this README update. They indicate that the model identifies some purchasing sessions but also produces false-positive predictions. Further evaluation and improvement are needed before considering real-world deployment.

The classification threshold affects the balance between precision and recall. Model performance should be judged in the context of the business objective and on data that is kept separate from model training and tuning.

## Important Considerations

- A predicted purchase probability is an estimate, not a guarantee that a visitor will purchase.
- The classification threshold affects predicted labels and the balance between precision and recall.
- Historical dataset patterns may not represent current shoppers or a particular online store.
- The demo store uses illustrative session measurements; it does not collect or analyse actual customer traffic.
- Input features must be available, correctly measured, and consistent with training data before the model is used in a real store.
- The model and API should undergo further validation, monitoring, and evaluation on newer data before real-world use.
- The current project is a portfolio/demo application, not a production-ready purchasing decision system.

## Future Improvements

- Expand automated test coverage as new features are introduced.
- Compare additional classification models and improve prediction quality.
- Add model-performance monitoring and data-drift checks.
- Improve dependency management with a pinned `requirements.txt` file.
- Add reproducible model-training instructions and document the evaluation split.
- Connect the demo store to persistent session storage if a realistic end-to-end demo is needed.
- Deploy the API and dashboards to a hosting platform.
- Add project screenshots and an architecture diagram to the documentation.

## Author

**Yasir Khan**

Machine Learning | Data Analytics | Python | AI Applications

GitHub: https://github.com/YasirKhan2929
