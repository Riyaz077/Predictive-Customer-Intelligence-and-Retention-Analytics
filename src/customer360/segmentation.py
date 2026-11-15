from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

SEGMENT_FEATURES = [
    "recency_days", "frequency", "monetary_value",
    "tenure_months", "engagement_score"
]

def fit_segments(df, n_clusters=4, random_state=42):
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state))
    ])
    labels = pipeline.fit_predict(df[SEGMENT_FEATURES])
    return pipeline, labels
