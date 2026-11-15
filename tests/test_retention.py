from src.customer360.retention import retention_recommendation, risk_label

def test_high_value_high_risk_recommendation():
    strategy, _ = retention_recommendation(2500, 0.85, 0.4)
    assert strategy == "High-value recovery"

def test_risk_labels():
    assert risk_label(0.8) == "High"
    assert risk_label(0.5) == "Medium"
    assert risk_label(0.2) == "Low"
