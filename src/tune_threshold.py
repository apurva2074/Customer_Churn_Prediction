
import joblib
import numpy as np

from pathlib import Path
from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from src.train import load_data, prepare_data


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

MODEL_PATHS = {
    "Tuned Logistic Regression":
        ARTIFACT_DIR / "logistic_tuned.joblib",

    "Balanced Logistic Regression":
        ARTIFACT_DIR / "churn_pipeline_balanced.joblib",

    "Tuned Random Forest":
        ARTIFACT_DIR / "random_forest_tuned.joblib",

    "Tuned Gradient Boosting":
        ARTIFACT_DIR / "gradient_boosting_tuned.joblib"
}


# ==========================================
# 2. LOAD AND SPLIT DATA
# ==========================================

df = load_data()
X, y = prepare_data(df)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 3. CROSS-VALIDATION SETTINGS
# ==========================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

TARGET_RECALL = 0.80

thresholds = np.arange(0.05, 0.96, 0.01)

results = []


# ==========================================
# 4. GENERATE OOF PROBABILITIES
# ==========================================

for model_name, model_path in MODEL_PATHS.items():

    print(f"\nEvaluating: {model_name}")

    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        continue

    saved_model = joblib.load(model_path)

    # Clone the pipeline so each fold trains independently
    model = clone(saved_model)

    # Generate out-of-fold probabilities on training data
    oof_probabilities = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1
    )[:, 1]

    # ======================================
    # 5. SELECT THRESHOLD
    # ======================================

    threshold_results = []

    for threshold in thresholds:

        oof_predictions = (
            oof_probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_train,
            oof_predictions,
            zero_division=0
        )

        recall = recall_score(
            y_train,
            oof_predictions,
            zero_division=0
        )

        threshold_results.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall
        })

    # Prefer thresholds meeting the target recall
    eligible = [
        result for result in threshold_results
        if result["recall"] >= TARGET_RECALL
    ]

    if eligible:
        selected = max(
            eligible,
            key=lambda result: (
                result["precision"],
                result["recall"]
            )
        )
    else:
        # If target recall is impossible, maximize recall
        selected = max(
            threshold_results,
            key=lambda result: (
                result["recall"],
                result["precision"]
            )
        )

    selected_threshold = selected["threshold"]

    print(f"Selected threshold: {selected_threshold:.2f}")
    print(f"CV precision: {selected['precision']:.4f}")
    print(f"CV recall: {selected['recall']:.4f}")

    # ======================================
    # 6. EVALUATE ON HELD-OUT TEST DATA
    # ======================================

    # Fit a fresh copy on all training data
    final_model = clone(saved_model)
    final_model.fit(X_train, y_train)

    test_probabilities = final_model.predict_proba(X_test)[:, 1]

    test_predictions = (
        test_probabilities >= selected_threshold
    ).astype(int)

    test_precision = precision_score(
        y_test, test_predictions, zero_division=0
    )

    test_recall = recall_score(
        y_test, test_predictions, zero_division=0
    )

    test_f1 = f1_score(
        y_test, test_predictions, zero_division=0
    )

    test_accuracy = accuracy_score(
        y_test, test_predictions
    )

    test_auc = roc_auc_score(
        y_test, test_probabilities
    )

    print("\nTEST RESULTS")
    print("-" * 40)
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1-score: {test_f1:.4f}")
    print(f"ROC-AUC: {test_auc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, test_predictions))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        test_predictions,
        zero_division=0
    ))

    results.append({
        "model": model_name,
        "threshold": selected_threshold,
        "cv_precision": selected["precision"],
        "cv_recall": selected["recall"],
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "test_accuracy": test_accuracy,
        "test_roc_auc": test_auc
    })


# ==========================================
# 7. FINAL COMPARISON
# ==========================================

print("\n\nTHRESHOLD COMPARISON")
print("=" * 90)

for result in results:
    print(
        f"\n{result['model']}"
        f"\nThreshold: {result['threshold']:.2f}"
        f" | CV Recall: {result['cv_recall']:.3f}"
        f" | Test Recall: {result['test_recall']:.3f}"
        f" | Test Precision: {result['test_precision']:.3f}"
        f" | Test F1: {result['test_f1']:.3f}"
        f" | Test ROC-AUC: {result['test_roc_auc']:.3f}"
    )
