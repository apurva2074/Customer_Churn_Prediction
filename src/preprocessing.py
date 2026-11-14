
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def create_preprocessor():
    """
    Creates a preprocessing pipeline for customer churn data.
    """

    numerical_features = [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    binary_features = [
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling"
    ]

    categorical_features = [
        "gender",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod"
    ]

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    binary_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            drop="if_binary",
            handle_unknown="ignore"
        ))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_features),
        ("binary", binary_pipeline, binary_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    return preprocessor
