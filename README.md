# Customer360
## AI-Based Customer Lifetime Value Prediction and Retention Intelligence System

Customer360 is an end-to-end customer analytics platform that uses machine learning and behavioural analytics to estimate customer lifetime value (CLV), identify customers at risk of churn, segment customer behaviour, and recommend targeted retention actions. It combines a reproducible Python analytics pipeline with an interactive Streamlit dashboard and a FastAPI service.

> **Data notice:** The included dataset is synthetic and generated for demonstration. Model results are illustrative and must not be presented as validated performance on real customers.

## Key capabilities

- **Customer Lifetime Value Prediction:** Estimate future customer value from historical purchase behaviour, recency, frequency, monetary value, tenure, and engagement features.
- **Churn Risk Prediction:** Estimate the probability that a customer will become inactive/churn within the defined observation framework.
- **Customer Segmentation:** Group customers using behavioural and value-related features with K-Means clustering.
- **Retention Intelligence:** Generate transparent, rule-based retention suggestions using CLV, churn risk, and customer segment.
- **Explainable outputs:** Show the strongest model features and human-readable reason codes where available.
- **Interactive dashboard:** Explore KPIs, customer segments, CLV distributions, churn risk, and individual customer profiles.
- **API access:** Serve customer-level predictions through FastAPI.

## Architecture

```text
Synthetic / Business Customer Data
              |
              v
     Data Validation & Cleaning
              |
              v
       Feature Engineering
              |
       +------+------+
       |             |
       v             v
  CLV Regression  Churn Classification
       |             |
       +------+------+
              |
              v
    Customer Segmentation
              |
              v
  Retention Rules & Recommendations
              |
       +------+------+
       |             |
       v             v
 Streamlit UI     FastAPI Service
```

## Technology stack

- Python, Pandas, NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- FastAPI and Uvicorn
- Joblib
- Pytest

## Repository structure

```text
Customer360/
├── api/
│   └── main.py
├── config/
│   └── config.yaml
├── data/
│   ├── processed/
│   └── sample/
│       └── customers.csv
├── docs/
│   ├── 01_project_overview.md
│   ├── 02_problem_statement.md
│   ├── 03_objectives_and_scope.md
│   ├── 04_literature_and_concepts.md
│   ├── 05_system_architecture.md
│   ├── 06_methodology.md
│   ├── 07_dataset_dictionary.md
│   ├── 08_feature_engineering.md
│   ├── 09_clv_prediction.md
│   ├── 10_churn_prediction.md
│   ├── 11_customer_segmentation.md
│   ├── 12_retention_intelligence.md
│   ├── 13_model_evaluation.md
│   ├── 14_dashboard_guide.md
│   ├── 15_api_guide.md
│   ├── 16_testing.md
│   ├── 17_deployment.md
│   ├── 18_privacy_and_responsible_use.md
│   ├── 19_limitations_and_future_work.md
│   └── 20_references.md
├── notebooks/
│   ├── 01_customer_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_clv_modeling.ipynb
│   ├── 04_churn_modeling.ipynb
│   └── 05_customer_segmentation.ipynb
├── scripts/
│   ├── generate_data.py
│   └── run_pipeline.py
├── src/customer360/
│   ├── __init__.py
│   ├── data.py
│   ├── features.py
│   ├── models.py
│   ├── segmentation.py
│   ├── retention.py
│   └── pipeline.py
├── tests/
│   ├── test_features.py
│   └── test_retention.py
├── dashboard/
│   └── app.py
├── .env.example
├── .gitignore
├── LICENSE
├── requirements.txt
└── pyproject.toml
```

## Quick start

### 1. Create an environment

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate the sample dataset

```bash
python scripts/generate_data.py
```

The generator writes `data/sample/customers.csv`.

### 4. Train models and create outputs

```bash
python scripts/run_pipeline.py
```

This creates model artifacts and scored customer data under `artifacts/` and `data/processed/`.

### 5. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

### 6. Launch the API

```bash
uvicorn api.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation.

## Run tests

```bash
pytest -q
```

## API example

After starting the API, send a request to `POST /predict`:

```json
{
  "recency_days": 18,
  "frequency": 12,
  "monetary_value": 1850.0,
  "tenure_months": 14,
  "avg_order_value": 154.17,
  "support_tickets": 1,
  "engagement_score": 0.72
}
```

The API returns predicted CLV, churn probability, a risk label, and a retention recommendation. Train the pipeline before starting the API.

## Methodology

1. Generate or load customer-level behavioural data.
2. Validate data types, missing values, and ranges.
3. Engineer RFM and engagement features.
4. Train a regression model for CLV estimation.
5. Train a classification model for churn-risk estimation.
6. Cluster customers using scaled behavioural features.
7. Apply transparent retention rules.
8. Evaluate models and inspect outputs.
9. Present insights in the dashboard and API.

## Evaluation

The pipeline reports:
- CLV regression: MAE, RMSE, and R².
- Churn classification: ROC-AUC, precision, recall, F1-score, and confusion matrix.

Use a chronological or otherwise leakage-aware split when adapting this project to production data. The demonstration data is synthetic; high scores on it do not establish real-world effectiveness.

## Responsible use

Customer scores should support—not replace—human decision-making. Validate models for bias, monitor drift, protect personal information, obtain appropriate consent, and avoid using sensitive attributes or proxies for unfair treatment. Do not upload real customer data to a public repository.

## Roadmap

- Add time-aware CLV modelling and probabilistic customer value.
- Add uplift modelling to estimate incremental retention campaign impact.
- Integrate CRM and commerce data through governed connectors.
- Add model monitoring, drift detection, and retraining workflows.
- Add SHAP-based explanations and campaign experiment tracking.

## Author

**Mohamed Riyaz S.**  
Computer Science & Engineering | Business Analytics

## License

MIT License. See [LICENSE](LICENSE).
