
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score

from src.train import load_data, prepare_data
from src.preprocessing import create_preprocessor


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACT_DIR / "logistic_tuned.joblib"


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
# 3. CREATE PIPELINE
# ==========================================

pipeline = Pipeline([
    ("preprocessor", create_preprocessor()),
    ("classifier", LogisticRegression(
        max_iter=10000,
        random_state=42
    ))
])


# ==========================================
# 4. DEFINE HYPERPARAMETERS
# ==========================================

param_grid = {
    "classifier__C": [0.01, 0.1, 1, 10, 100],
    "classifier__solver": ["lbfgs", "liblinear"],
    "classifier__class_weight": [None, "balanced"]
}


# ==========================================
# 5. CROSS-VALIDATION
# ==========================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    verbose=1
)


# ==========================================
# 6. TRAIN AND FIND BEST PARAMETERS
# ==========================================

grid_search.fit(X_train, y_train)

print("\nBEST PARAMETERS:")
print(grid_search.best_params_)

print("\nBEST CROSS-VALIDATION ROC-AUC:")
print(grid_search.best_score_)


# ==========================================
# 7. FINAL TEST EVALUATION
# ==========================================

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print("\nFINAL TEST CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))

print("\nFINAL TEST ROC-AUC:")
print(roc_auc_score(y_test, y_prob))


# ==========================================
# 8. SAVE BEST MODEL
# ==========================================

joblib.dump(best_model, MODEL_PATH)

print(f"\nTuned model saved at: {MODEL_PATH}")
