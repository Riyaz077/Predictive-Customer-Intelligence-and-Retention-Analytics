from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from .data import load_customers
from .features import build_features
from .models import (train_clv_model, train_churn_model,
                     regression_metrics, classification_metrics)
from .segmentation import fit_segments
from .retention import retention_recommendation, risk_label

def run_pipeline(data_path="data/sample/customers.csv", output_dir="artifacts",
                 processed_path="data/processed/customer_scores.csv"):
    df = load_customers(data_path)
    X = build_features(df)
    indices = range(len(df))
    train_idx, test_idx = train_test_split(
        list(indices), test_size=0.2, random_state=42, stratify=df["churned"]
    )
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]

    clv = train_clv_model(X_train, df["future_value"].iloc[train_idx])
    churn = train_churn_model(X_train, df["churned"].iloc[train_idx])
    clv_pred = clv.predict(X_test)
    churn_prob = churn.predict_proba(X_test)[:, 1]
    churn_pred = (churn_prob >= 0.5).astype(int)

    metrics = {
        "clv_regression": regression_metrics(df["future_value"].iloc[test_idx], clv_pred),
        "churn_classification": classification_metrics(
            df["churned"].iloc[test_idx], churn_prob, churn_pred
        )
    }

    segmenter, labels = fit_segments(df)
    scored = df.copy()
    scored["predicted_clv"] = clv.predict(X)
    scored["churn_probability"] = churn.predict_proba(X)[:, 1]
    scored["churn_risk"] = scored["churn_probability"].map(risk_label)
    scored["segment"] = labels
    recommendations = scored.apply(
        lambda r: retention_recommendation(
            float(r["predicted_clv"]), float(r["churn_probability"]),
            float(r["engagement_score"])
        ), axis=1
    )
    scored["retention_strategy"] = [v[0] for v in recommendations]
    scored["recommended_action"] = [v[1] for v in recommendations]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    Path(processed_path).parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(processed_path, index=False)
    joblib.dump(clv, out / "clv_model.joblib")
    joblib.dump(churn, out / "churn_model.joblib")
    joblib.dump(segmenter, out / "segmenter.joblib")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics, scored
