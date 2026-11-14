
import joblib
import pandas as pd

from pathlib import Path


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "artifacts" / "random_forest_tuned.joblib"

CHURN_THRESHOLD = 0.45


# ==========================================
# 2. LOAD TRAINED MODEL
# ==========================================

model = joblib.load(MODEL_PATH)


# ==========================================
# 3. PREDICTION FUNCTION
# ==========================================

def predict_churn(customer_data):

    """
    Predict whether a customer is likely to churn.

    Parameters:
        customer_data: Dictionary containing customer details.

    Returns:
        Dictionary containing churn prediction and probability.
    """

    # Convert dictionary into DataFrame
    customer_df = pd.DataFrame([customer_data])

    # Predict churn probability
    churn_probability = model.predict_proba(
        customer_df
    )[0][1]

    # Apply our selected threshold
    prediction = int(
        churn_probability >= CHURN_THRESHOLD
    )

    return {
        "churn_prediction": prediction,
        "churn_status": (
            "Likely to Churn"
            if prediction == 1
            else "Not Likely to Churn"
        ),
        "churn_probability": round(
            float(churn_probability), 4
        ),
        "threshold_used": CHURN_THRESHOLD
    }


# ==========================================
# 4. LOCAL TESTING
# ==========================================

if __name__ == "__main__":

    sample_customer = {
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

    result = predict_churn(sample_customer)

    print("\nCUSTOMER CHURN PREDICTION")
    print("=" * 40)

    for key, value in result.items():
        print(f"{key}: {value}")
