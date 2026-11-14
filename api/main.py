
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal

from src.predict import predict_churn


# ==========================================
# 1. CREATE FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn using Machine Learning.",
    version="1.0.0"
)


# ==========================================
# 2. CUSTOMER INPUT VALIDATION
# ==========================================

class CustomerInput(BaseModel):

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]

    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]

    tenure: int = Field(ge=0)

    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal[
        "Yes", "No", "No phone service"
    ]

    InternetService: Literal[
        "DSL", "Fiber optic", "No"
    ]

    OnlineSecurity: Literal[
        "Yes", "No", "No internet service"
    ]

    OnlineBackup: Literal[
        "Yes", "No", "No internet service"
    ]

    DeviceProtection: Literal[
        "Yes", "No", "No internet service"
    ]

    TechSupport: Literal[
        "Yes", "No", "No internet service"
    ]

    StreamingTV: Literal[
        "Yes", "No", "No internet service"
    ]

    StreamingMovies: Literal[
        "Yes", "No", "No internet service"
    ]

    Contract: Literal[
        "Month-to-month", "One year", "Two year"
    ]

    PaperlessBilling: Literal["Yes", "No"]

    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]

    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


# ==========================================
# 3. HOME ENDPOINT
# ==========================================

@app.get("/")
def home():

    return {
        "message": "Customer Churn Prediction API is running",
        "docs": "/docs"
    }


# ==========================================
# 4. HEALTH CHECK ENDPOINT
# ==========================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model": "Tuned Random Forest",
        "threshold": 0.45
    }


# ==========================================
# 5. PREDICTION ENDPOINT
# ==========================================

@app.post("/predict")
def predict(customer: CustomerInput):

    customer_data = customer.model_dump()

    result = predict_churn(customer_data)

    return result
