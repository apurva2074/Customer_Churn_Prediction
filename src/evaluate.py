
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from src.train import load_data, prepare_data


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "artifacts" / "churn_knn.joblib"


# ==========================================
# 2. EVALUATE MODEL
# ==========================================

def evaluate_model():

    # Load dataset
    df = load_data()

    # Separate features and target
    X, y = prepare_data(df)

    # Recreate the same test split used during training
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Load trained pipeline
    model = joblib.load(MODEL_PATH)

    # Generate predictions
    y_pred = model.predict(X_test)

    # Get churn probabilities for ROC-AUC
    y_prob = model.predict_proba(X_test)[:, 1]

    # ======================================
    # 3. EVALUATION METRICS
    # ======================================

    print("\nMODEL EVALUATION")
    print("=" * 40)

    print("\nAccuracy:")
    print(accuracy_score(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nROC-AUC Score:")
    print(roc_auc_score(y_test, y_prob))


# ==========================================
# 4. EXECUTION
# ==========================================

if __name__ == "__main__":
    evaluate_model()
