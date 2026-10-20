"""
Student Performance Prediction API (FastAPI)

Expects these files in the same folder as this script:
    student_model.joblib, scaler.joblib, imputer.joblib, feature_columns.json
"""

import json
import os
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prisma import Prisma

BASE_DIR = Path(__file__).resolve().parent

# --- Load artifacts once at startup -----------------------------------------
try:
    model = joblib.load(BASE_DIR / "student_model.joblib")
    scaler = joblib.load(BASE_DIR / "scaler.joblib")
    imputer = joblib.load(BASE_DIR / "imputer.joblib")
    FEATURES = json.loads((BASE_DIR / "feature_columns.json").read_text())
except FileNotFoundError as e:
    raise RuntimeError(
        f"Missing model artifact: {e.filename}. Copy the 4 exported files "
        "from your training folder into backend/."
    ) from e

# --- Database Lifecycle -----------------------------------------------------
db = Prisma()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.connect()
    yield
    await db.disconnect()

# --- App + CORS -------------------------------------------------------------
app = FastAPI(
    title="Student Performance Prediction API", 
    version="1.0.0", 
    lifespan=lifespan  # Attach the database lifecycle manager
)

origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request / response schemas ---------------------------------------------
class StudentInput(BaseModel):
    Age: Optional[float] = Field(None, ge=10, le=30)
    Gender: Optional[int] = Field(None, ge=0, le=1)
    Ethnicity: Optional[int] = Field(None, ge=0, le=3)
    ParentalEducation: Optional[int] = Field(None, ge=0, le=4)
    StudyTimeWeekly: Optional[float] = Field(None, ge=0, le=40)
    Absences: Optional[int] = Field(None, ge=0, le=100)
    Tutoring: Optional[int] = Field(None, ge=0, le=1)
    ParentalSupport: Optional[int] = Field(None, ge=0, le=4)
    Extracurricular: Optional[int] = Field(None, ge=0, le=1)
    Sports: Optional[int] = Field(None, ge=0, le=1)
    Music: Optional[int] = Field(None, ge=0, le=1)
    Volunteering: Optional[int] = Field(None, ge=0, le=1)

class PredictionOut(BaseModel):
    passed: bool
    prediction: int
    pass_probability: float
    fail_probability: float

# --- Routes -----------------------------------------------------------------
@app.get("/")
def root():
    return {"service": "Student Performance Prediction API", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok", "features": FEATURES}

# Note: This function must be `async def` because Prisma database calls are asynchronous
@app.post("/predict", response_model=PredictionOut)
async def predict(student: StudentInput):
    data = student.model_dump()
    row = pd.DataFrame([[data.get(col) for col in FEATURES]], columns=FEATURES, dtype=float)
    
    try:
        X = scaler.transform(imputer.transform(row))
        proba = model.predict_proba(X)[0]
        pred = int(model.predict(X)[0])
    except Exception as exc:  
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    classes = list(model.classes_)
    is_passed = bool(pred == 1)
    pass_prob = round(float(proba[classes.index(1)]), 4)
    fail_prob = round(float(proba[classes.index(0)]), 4)

    # Save the prediction to the database
    try:
        await db.prediction.create(
            data={
                "age": data.get("Age"),
                "gender": data.get("Gender"),
                "ethnicity": data.get("Ethnicity"),
                "parentalEducation": data.get("ParentalEducation"),
                "studyTimeWeekly": data.get("StudyTimeWeekly"),
                "absences": data.get("Absences"),
                "tutoring": data.get("Tutoring"),
                "parentalSupport": data.get("ParentalSupport"),
                "extracurricular": data.get("Extracurricular"),
                "sports": data.get("Sports"),
                "music": data.get("Music"),
                "volunteering": data.get("Volunteering"),
                "passed": is_passed,
                "pass_probability": pass_prob
            }
        )
    except Exception as db_exc:
        # We print the error but do not fail the request if the DB insert fails
        print(f"Database logging failed: {db_exc}")

    return PredictionOut(
        passed=is_passed,
        prediction=pred,
        pass_probability=pass_prob,
        fail_probability=fail_prob,
    )