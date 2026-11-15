from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import numpy as np

def train_clv_model(X_train, y_train, random_state=42):
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=250, min_samples_leaf=2,
            random_state=random_state, n_jobs=-1
        ))
    ])
    model.fit(X_train, y_train)
    return model

def train_churn_model(X_train, y_train, random_state=42):
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=250, min_samples_leaf=2,
            class_weight="balanced", random_state=random_state, n_jobs=-1
        ))
    ])
    model.fit(X_train, y_train)
    return model

def regression_metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred))
    }

def classification_metrics(y_true, probabilities, predictions):
    result = {
        "Precision": float(precision_score(y_true, predictions, zero_division=0)),
        "Recall": float(recall_score(y_true, predictions, zero_division=0)),
        "F1": float(f1_score(y_true, predictions, zero_division=0))
    }
    if len(set(y_true)) > 1:
        result["ROC_AUC"] = float(roc_auc_score(y_true, probabilities))
    else:
        result["ROC_AUC"] = None
    return result
