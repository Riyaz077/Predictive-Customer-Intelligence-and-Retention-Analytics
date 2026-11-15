import numpy as np
import pandas as pd

NUMERIC_INPUTS = [
    "recency_days", "frequency", "monetary_value", "tenure_months",
    "avg_order_value", "support_tickets", "engagement_score"
]

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-conscious customer behaviour features."""
    x = df[NUMERIC_INPUTS].copy()
    x["purchase_frequency_per_month"] = x["frequency"] / x["tenure_months"].clip(lower=1)
    x["value_per_month"] = x["monetary_value"] / x["tenure_months"].clip(lower=1)
    x["recency_frequency_ratio"] = x["recency_days"] / (x["frequency"] + 1)
    x["engagement_value"] = x["engagement_score"] * np.log1p(x["monetary_value"].clip(lower=0))
    return x.replace([np.inf, -np.inf], 0).fillna(0)
