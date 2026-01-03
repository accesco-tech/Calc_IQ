from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles  # Added for image loading
from pydantic import BaseModel, Field
import numpy as np
import tensorflow as tf
import joblib

# -------------------------------------------------
# LOAD MODEL & SCALERS
# -------------------------------------------------
MODEL_PATH = "monthly_budget_model.keras"
INPUT_SCALER_PATH = "input_scaler.pkl"
OUTPUT_SCALER_PATH = "output_scaler.pkl"

model = tf.keras.models.load_model(MODEL_PATH)
input_scaler = joblib.load(INPUT_SCALER_PATH)
output_scaler = joblib.load(OUTPUT_SCALER_PATH)

# -------------------------------------------------
# FASTAPI APP
# -------------------------------------------------
app = FastAPI(
    title="Monthly Budget AI",
    description="AI-powered monthly budget generator",
    version="1.0.0"
)

# -------------------------------------------------
# STATIC FILES CONFIG (Added for image loading)
# -------------------------------------------------
# This links your 'templates/static' folder to the '/static' URL path
app.mount("/static", StaticFiles(directory="templates/static"), name="static")

# -------------------------------------------------
# TEMPLATE CONFIG
# -------------------------------------------------
templates = Jinja2Templates(directory="templates")

# -------------------------------------------------
# INPUT SCHEMA
# -------------------------------------------------
class BudgetInput(BaseModel):
    income: float = Field(..., gt=0, description="Monthly income")
    rent: float = Field(..., ge=0, description="Monthly rent")
    cost_index: float = Field(..., ge=0.7, le=1.6, description="City cost index")
    family_members: int = Field(..., ge=1, le=10, description="Number of family members")

# -------------------------------------------------
# OUTPUT SCHEMA
# -------------------------------------------------
class BudgetOutput(BaseModel):
    grocery_and_essentials: float
    education: float
    healthcare: float
    transport: float
    travel: float
    savings: float
    entertainment: float

# -------------------------------------------------
# INDEX PAGE
# -------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# -------------------------------------------------
# PREDICTION ENDPOINT
# -------------------------------------------------
@app.post("/predict", response_model=BudgetOutput)
def predict_budget(data: BudgetInput):

    # Prepare input array
    input_array = np.array([[
        data.income,
        data.rent,
        data.cost_index,
        data.family_members
    ]])

    # Scale input
    scaled_input = input_scaler.transform(input_array)

    # Predict
    scaled_output = model.predict(scaled_input, verbose=0)

    # Inverse scale output
    output = output_scaler.inverse_transform(scaled_output)[0]

    # Clamp negative values
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
