from pathlib import Path
import numpy as np
import pandas as pd

def generate_customers(n=2500, seed=42):
    rng = np.random.default_rng(seed)
    recency = np.clip(rng.gamma(2.2, 24, n), 1, 240).astype(int)
    frequency = np.clip(rng.poisson(9, n) + 1, 1, 60)
    tenure = rng.integers(2, 49, n)
    aov = np.clip(rng.lognormal(4.1, 0.65, n), 10, 1500)
    monetary = frequency * aov
    support = rng.poisson(1.2, n)
    engagement = np.clip(rng.beta(2.2, 2.0, n), 0, 1)
    # Synthetic churn mechanism, for demonstration only.
    logit = -2.1 + 0.018 * recency - 0.10 * frequency + 0.55 * support - 1.4 * engagement
    p = 1 / (1 + np.exp(-logit))
    churned = rng.binomial(1, p)
    future_value = np.maximum(
        0, monetary * rng.uniform(0.35, 1.2, n)
        + frequency * aov * engagement * 0.25
        - recency * aov * 0.025
        + rng.normal(0, 80, n)
    )
    return pd.DataFrame({
        "customer_id": [f"C{idx:05d}" for idx in range(1, n + 1)],
        "recency_days": recency,
        "frequency": frequency,
        "monetary_value": monetary.round(2),
        "tenure_months": tenure,
        "avg_order_value": aov.round(2),
        "support_tickets": support,
        "engagement_score": engagement.round(3),
        "churned": churned,
        "future_value": future_value.round(2)
    })

if __name__ == "__main__":
    path = Path("data/sample/customers.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    generate_customers().to_csv(path, index=False)
    print(f"Generated {path}")
