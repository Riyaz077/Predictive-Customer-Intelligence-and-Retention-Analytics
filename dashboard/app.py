from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer360", page_icon="📊", layout="wide")
st.title("Customer360")
st.caption("AI-Based Customer Lifetime Value Prediction and Retention Intelligence")

path = Path("data/processed/customer_scores.csv")
if not path.exists():
    st.warning("Scored data not found. Run `python scripts/run_pipeline.py` first.")
    st.stop()

df = pd.read_csv(path)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{len(df):,}")
c2.metric("Average predicted CLV", f"{df.predicted_clv.mean():,.0f}")
c3.metric("High churn-risk customers", f"{(df.churn_risk == 'High').sum():,}")
c4.metric("Average churn probability", f"{df.churn_probability.mean():.1%}")

tab1, tab2, tab3 = st.tabs(["Overview", "Segments & Risk", "Customer Explorer"])
with tab1:
    left, right = st.columns(2)
    with left:
        st.subheader("Predicted CLV distribution")
        fig, ax = plt.subplots()
        ax.hist(df["predicted_clv"], bins=30)
        ax.set_xlabel("Predicted CLV")
        ax.set_ylabel("Customers")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.subheader("Churn risk distribution")
        st.bar_chart(df["churn_risk"].value_counts())
with tab2:
    st.subheader("Customer segment sizes")
    st.bar_chart(df["segment"].value_counts().sort_index())
    st.subheader("Average CLV by segment")
    st.bar_chart(df.groupby("segment")["predicted_clv"].mean())
    st.dataframe(
        df.groupby("churn_risk").agg(
            customers=("customer_id", "count"),
            avg_clv=("predicted_clv", "mean"),
            avg_probability=("churn_probability", "mean")
        ).round(2), use_container_width=True
    )
with tab3:
    selected = st.selectbox("Select customer", df["customer_id"].tolist())
    row = df.loc[df.customer_id == selected].iloc[0]
    a, b, c = st.columns(3)
    a.metric("Predicted CLV", f"{row.predicted_clv:,.2f}")
    b.metric("Churn probability", f"{row.churn_probability:.1%}")
    c.metric("Risk", row.churn_risk)
    st.write("**Retention strategy:**", row.retention_strategy)
    st.write("**Recommended action:**", row.recommended_action)
    st.json({
        "recency_days": int(row.recency_days),
        "frequency": int(row.frequency),
        "monetary_value": float(row.monetary_value),
        "tenure_months": int(row.tenure_months),
        "engagement_score": float(row.engagement_score),
        "segment": int(row.segment)
    })

st.caption("Demonstration only: included customer data is synthetic. Validate before business use.")
