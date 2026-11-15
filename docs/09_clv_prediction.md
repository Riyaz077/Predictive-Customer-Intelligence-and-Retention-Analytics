# CLV Prediction

CLV is treated as a regression target using the synthetic future_value field. The baseline uses a RandomForestRegressor inside a scikit-learn pipeline with median imputation. The output is an estimate for demonstration and should be recalibrated against an agreed business definition and prediction horizon.
