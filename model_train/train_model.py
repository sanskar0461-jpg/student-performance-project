"""
Student Performance Prediction - Training Pipeline
---------------------------------------------------
Reads Student_performance_data__.csv (same folder), trains a Random Forest classifier,
prints evaluation metrics, and exports artifacts for a FastAPI backend.

Outputs:
    student_model.joblib   - trained RandomForestClassifier
    scaler.joblib          - fitted StandardScaler
    imputer.joblib         - fitted SimpleImputer (mean) for missing values at inference
    feature_columns.json   - exact feature order expected by the model
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "Student_performance_data _.csv"
MODEL_PATH = BASE_DIR / "student_model.joblib"
SCALER_PATH = BASE_DIR / "scaler.joblib"
IMPUTER_PATH = BASE_DIR / "imputer.joblib"
FEATURES_PATH = BASE_DIR / "feature_columns.json"

TARGET = "Passed"
RANDOM_STATE = 42

# This dataset has no 'Passed' column, so it is derived from GradeClass
# (0=A, 1=B, 2=C, 3=D, 4=F): Passed = 1 unless the student got an F.
# GPA and GradeClass are dropped from the features because they ARE the
# outcome (GradeClass is computed from GPA) - keeping them would leak the
# answer and make the model useless for real prediction.
DROP_COLUMNS = ["StudentID", "GPA", "GradeClass"]


def main() -> None:
    # 1. Data loading ---------------------------------------------------------
    if not DATASET_PATH.exists():
        sys.exit(f"ERROR: '{DATASET_PATH.name}' not found in {BASE_DIR}")

    df = pd.read_csv(DATASET_PATH)
    print(f"Loaded dataset: {df.shape[0]} rows x {df.shape[1]} columns")

    if TARGET not in df.columns:
        if "GradeClass" not in df.columns:
            sys.exit(f"ERROR: neither '{TARGET}' nor 'GradeClass' found. Columns: {list(df.columns)}")
        df[TARGET] = (df["GradeClass"] != 4).astype(int)
        print(f"Derived '{TARGET}' from GradeClass:\n{df[TARGET].value_counts().to_string()}\n")

    # 2. Separate X / y -------------------------------------------------------
    df = df.dropna(subset=[TARGET])  # rows without a label are unusable
    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET] + [c for c in DROP_COLUMNS if c in df.columns]).apply(pd.to_numeric, errors="coerce")
    feature_columns = X.columns.tolist()

    print(f"Features: {feature_columns}")
    print(f"Missing values per column:\n{X.isna().sum().to_string()}\n")

    # 3. Train/test split (80/20) --------------------------------------------
    # Split BEFORE imputing/scaling so test data never influences the fitted
    # imputer/scaler (avoids data leakage).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    # Preprocessing: mean imputation, then standard scaling (fit on train only)
    imputer = SimpleImputer(strategy="mean")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    # 4. Model development ----------------------------------------------------
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    # 5. Performance evaluation ----------------------------------------------
    y_pred = model.predict(X_test_scaled)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred), "\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # 6. Model export ---------------------------------------------------------
    joblib.dump(model, MODEL_PATH, compress=3)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(imputer, IMPUTER_PATH)
    FEATURES_PATH.write_text(json.dumps(feature_columns, indent=2))

    print("Saved artifacts:")
    for p in (MODEL_PATH, SCALER_PATH, IMPUTER_PATH, FEATURES_PATH):
        print(f"  {p.name:<22} {p.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
