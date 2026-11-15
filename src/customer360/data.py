from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = [
    "customer_id", "recency_days", "frequency", "monetary_value",
    "tenure_months", "avg_order_value", "support_tickets",
    "engagement_score", "churned", "future_value"
]

def load_customers(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if df["customer_id"].duplicated().any():
        raise ValueError("customer_id must be unique")
    return df
