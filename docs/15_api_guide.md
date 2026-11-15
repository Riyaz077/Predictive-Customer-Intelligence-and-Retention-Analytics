# API Guide

Start the API with `uvicorn api.main:app --reload`. The POST `/predict` endpoint accepts recency_days, frequency, monetary_value, tenure_months, avg_order_value, support_tickets, and engagement_score. Models must exist in the artifacts directory. Interactive documentation is available at `/docs`.
