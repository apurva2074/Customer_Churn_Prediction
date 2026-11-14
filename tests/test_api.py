
from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


# ==========================================
# TEST 1: HOME ENDPOINT
# ==========================================

def test_home_endpoint():

    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == (
        "Customer Churn Prediction API is running"
    )


# ==========================================
# TEST 2: HEALTH ENDPOINT
# ==========================================

def test_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model"] == "Tuned Random Forest"
    assert response.json()["threshold"] == 0.45


# ==========================================
# TEST 3: VALID CUSTOMER PREDICTION
# ==========================================

def test_valid_customer_prediction():

    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.50,
        "TotalCharges": 171.00
    }

    response = client.post("/predict", json=customer)

    assert response.status_code == 200

    result = response.json()

    assert result["churn_prediction"] in [0, 1]
    assert result["churn_status"] in [
        "Likely to Churn",
        "Not Likely to Churn"
    ]

    assert 0 <= result["churn_probability"] <= 1
    assert result["threshold_used"] == 0.45


# ==========================================
# TEST 4: INVALID CUSTOMER INPUT
# ==========================================

def test_invalid_gender():

    customer = {
        "gender": "Unknown"
    }

    response = client.post("/predict", json=customer)

    assert response.status_code == 422


# ==========================================
# TEST 5: MISSING REQUIRED FIELDS
# ==========================================

def test_missing_customer_fields():

    customer = {}

    response = client.post("/predict", json=customer)

    assert response.status_code == 422
