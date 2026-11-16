# Customer360
## AI-Based Customer Lifetime Value Prediction and Retention Intelligence System

Customer360 is an end-to-end customer analytics platform that uses machine learning and behavioural analytics to estimate customer lifetime value (CLV), identify customers at risk of churn, segment customer behaviour, and recommend targeted retention actions. It combines a reproducible Python analytics pipeline with an interactive Streamlit dashboard and a FastAPI service.

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

## **Quick Start**

- Create and activate a Python virtual environment.
- Install the required dependencies using `requirements.txt`.
- Generate the sample customer dataset using the data generation script.
- Execute the end-to-end pipeline to preprocess data and train machine-learning models.
- Generate model artifacts and customer predictions.
- Access the processed datasets and trained outputs in the `artifacts/` and `data/processed/` directories.

## **API Usage – Customer Intelligence & Prediction**

The `/predict` API endpoint accepts customer behavioural, transactional, and engagement data through a POST request to generate actionable customer intelligence.

**Key input parameters:**

- **Recency:** Days since the customer's last purchase.
- **Frequency:** Number of customer transactions.
- **Monetary Value:** Total customer spending.
- **Tenure:** Duration of the customer relationship.
- **Average Order Value:** Average spending per transaction.
- **Support Tickets:** Number of customer support interactions.
- **Engagement Score:** Customer engagement level.

**Expected API outputs:**

- Predicted Customer Lifetime Value (CLV).
- Customer churn probability.
- Customer risk classification.
- AI-driven customer retention recommendations.


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
Integrated M.Tech, Computer Science and Engineering (Business Analytics)  
VIT Chennai
