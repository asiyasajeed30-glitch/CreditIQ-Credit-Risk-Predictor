from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List
import joblib
import numpy as np
import pandas as pd
import os

app = FastAPI(title="Credit Risk Predictor", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Load saved objects
model = joblib.load(os.path.join(BASE_DIR, "logistic_model.pkl"))
pca   = joblib.load(os.path.join(BASE_DIR, "pca_transform.pkl"))

scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

# Label encoding maps (alphabetical = sklearn LabelEncoder order)
SEX_MAP      = {"female": 0, "male": 1}
HOUSING_MAP  = {"free": 0, "own": 1, "rent": 2}
SAVING_MAP   = {"unknown": 0, "little": 1, "moderate": 2, "quite rich": 3, "rich": 4}
CHECKING_MAP = {"unknown": 0, "little": 1, "moderate": 2, "rich": 3}

# Purpose columns in alphabetical order (pd.get_dummies order)
PURPOSE_COLS = [
    "business",
    "car",
    "domestic appliances",
    "education",
    "furniture/equipment",
    "radio/TV",
    "repairs",
    "vacation/others",
]

# Feature order matching training
FEATURE_ORDER = [
    "Age", "Sex", "Job", "Housing",
    "Saving accounts", "Checking account",
    "Credit amount", "Duration",
] + PURPOSE_COLS


# Pydantic schemas
class CreditInput(BaseModel):
    age: int              = Field(..., ge=18, le=100, example=35)
    sex: str              = Field(..., example="male")
    job: int              = Field(..., ge=0, le=3, example=2)
    housing: str          = Field(..., example="own")
    saving_accounts: str  = Field(..., example="little")
    checking_account: str = Field(..., example="moderate")
    credit_amount: float  = Field(..., gt=0, example=5000)
    duration: int         = Field(..., gt=0, example=24)
    purpose: str          = Field(..., example="car")


class BatchInput(BaseModel):
    records: List[CreditInput]


# Core prediction function
def encode_and_predict(record: CreditInput) -> dict:
    # Step 1: Label-encode categoricals
    row = {
        "Age":              record.age,
        "Sex":              SEX_MAP.get(record.sex.strip().lower(), 1),
        "Job":              record.job,
        "Housing":          HOUSING_MAP.get(record.housing.strip().lower(), 1),
        "Saving accounts":  SAVING_MAP.get(record.saving_accounts.strip().lower(), 1),
        "Checking account": CHECKING_MAP.get(record.checking_account.strip().lower(), 1),
        "Credit amount":    record.credit_amount,
        "Duration":         record.duration,
    }

    # Step 2: One-hot encode Purpose
    purpose_clean = record.purpose.strip().lower()
    for col in PURPOSE_COLS:
        row[col] = 1 if col == purpose_clean else 0

    # Step 3: Build DataFrame in exact column order
    df = pd.DataFrame([row], columns=FEATURE_ORDER)

    # Step 4: Scale if scaler is available
    if scaler is not None:
        X = pd.DataFrame(scaler.transform(df), columns=FEATURE_ORDER)
    else:
        X = df

    # Step 5: PCA transform
    X_pca = pca.transform(X)

    # Step 6: Predict
    label = int(model.predict(X_pca)[0])

    proba = model.predict_proba(X_pca)[0]
    confidence = round(float(max(proba)) * 100, 2)

    return {
        "risk":          "good" if label == 1 else "bad",
        "risk_label":    label,
        "confidence":    confidence,
        "input_summary": {
            "age":           record.age,
            "sex":           record.sex,
            "credit_amount": record.credit_amount,
            "duration":      record.duration,
            "purpose":       record.purpose,
        },
    }


# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/predict")
async def predict_single(data: CreditInput):
    return encode_and_predict(data)


@app.post("/predict/batch")
async def predict_batch(data: BatchInput):
    results = []
    for i, record in enumerate(data.records):
        res = encode_and_predict(record)
        res["record_index"] = i + 1
        results.append(res)
    return {"total": len(results), "predictions": results}


@app.get("/health")
async def health():
    return {"status": "ok", "model": "LogisticRegression + PCA(n=5)"}
