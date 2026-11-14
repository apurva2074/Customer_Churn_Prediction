
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from src.preprocessing import create_preprocessor
from sklearn.neighbors import KNeighborsClassifier

# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACT_DIR / "churn_knn.joblib"

# ==========================================
# 2. LOAD DATASET
# ==========================================

def load_data():

    df = pd.read_csv(DATA_PATH)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Convert target into binary values
    df["Churn"] = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    return df


# ==========================================
# 3. PREPARE FEATURES AND TARGET
# ==========================================

def prepare_data(df):

    X = df.drop(columns=["Churn", "customerID"])
    y = df["Churn"]

    return X, y


# ==========================================
# 4. TRAIN MODEL
# ==========================================

def train_model():

    df = load_data()

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Create complete ML pipeline
    # Create complete ML pipeline
    model_pipeline = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("classifier", KNeighborsClassifier(
            n_neighbors=7,weights="distance",
            metric="minkowski",
            p=2,
            n_jobs=-1
        ))
        
    ])

    # Train
    model_pipeline.fit(X_train, y_train)

    # Save complete pipeline
    joblib.dump(model_pipeline, MODEL_PATH)

    print("Model training completed!")
    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")
    print(f"Model saved at: {MODEL_PATH}")


# ==========================================
# 5. EXECUTION
# ==========================================

if __name__ == "__main__":
    train_model()
