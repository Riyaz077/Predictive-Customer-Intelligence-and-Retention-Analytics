import pandas as pd
from src.customer360.features import build_features

def test_feature_engineering_returns_expected_columns():
    df = pd.DataFrame([{
        "recency_days": 10, "frequency": 5, "monetary_value": 500,
        "tenure_months": 10, "avg_order_value": 100,
        "support_tickets": 1, "engagement_score": 0.5
    }])
    result = build_features(df)
    assert "purchase_frequency_per_month" in result
    assert result.iloc[0]["purchase_frequency_per_month"] == 0.5
