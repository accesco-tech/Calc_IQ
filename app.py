import os
import numpy as np
import tensorflow as tf
import joblib

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# -------------------------------------------------
# 1. PATH CONFIG
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "monthly_budget_model.keras")
INPUT_SCALER_PATH = os.path.join(BASE_DIR, "input_scaler.pkl")
OUTPUT_SCALER_PATH = os.path.join(BASE_DIR, "output_scaler.pkl")

# -------------------------------------------------
# 2. LOAD MODEL & SCALERS
# -------------------------------------------------
for path in [MODEL_PATH, INPUT_SCALER_PATH, OUTPUT_SCALER_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")

model = tf.keras.models.load_model(MODEL_PATH)
input_scaler = joblib.load(INPUT_SCALER_PATH)
output_scaler = joblib.load(OUTPUT_SCALER_PATH)

# -------------------------------------------------
# 3. FASTAPI APP
# -------------------------------------------------
app = FastAPI(
    title="Monthly Budget AI",
    description="AI-powered monthly budget generator",
    version="2.0.0"
)

# REQUIRED FOR BROWSER FETCH + RENDER
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static + Templates
app.mount("/static", StaticFiles(directory="templates/static"), name="static")
templates = Jinja2Templates(directory="templates")

# -------------------------------------------------
# 4. SCHEMAS
# -------------------------------------------------
class BudgetInput(BaseModel):
    income: float = Field(..., gt=0)
    rent: float = Field(..., ge=0)
    cost_index: float = Field(..., ge=0.7, le=1.6)
    family_members: int = Field(..., ge=1, le=10)

class BudgetOutput(BaseModel):
    grocery_and_essentials: float
    education: float
    healthcare: float
    transport: float
    travel: float
    savings: float
    entertainment: float

# -------------------------------------------------
# 5. ROUTES
# -------------------------------------------------

# ✅ FIX: Allow HEAD for Render health check
@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_model=BudgetOutput)
def predict_budget(data: BudgetInput):
    input_array = np.array([[
        data.income,
        data.rent,
        data.cost_index,
        data.family_members
    ]])

    scaled_input = input_scaler.transform(input_array)
    scaled_output = model.predict(scaled_input, verbose=0)
    output = output_scaler.inverse_transform(scaled_output)[0]

    output = np.maximum(output, 0)

    return {
        "grocery_and_essentials": round(float(output[0]), 2),
        "education": round(float(output[1]), 2),
        "healthcare": round(float(output[2]), 2),
        "transport": round(float(output[3]), 2),
        "travel": round(float(output[4]), 2),
        "savings": round(float(output[5]), 2),
        "entertainment": round(float(output[6]), 2),
    }


@app.get("/health")
def health():
    return {"status": "ok"}
