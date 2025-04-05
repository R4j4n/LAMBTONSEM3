# api.py - Modified to return the force plot as an image
import base64
import io
import pickle
from typing import Any, Dict, List

import catboost
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# Import Pydantic models
from models import CADRawInput, HealthCheck, PredictionResult

# Create FastAPI app
app = FastAPI(
    title="CAD Prediction API",
    description="API for predicting Coronary Artery Disease risk and explaining predictions",
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global variables for loaded models and data
cat_model = None
preprocessor = None
cat_explainer = None
column_names = None
catord_cols = None


# Load the saved model and explainer on startup
@app.on_event("startup")
def load_model():
    global cat_model, preprocessor, cat_explainer, column_names, catord_cols

    try:
        # Load CatBoost model
        cat_model = catboost.CatBoostClassifier()
        cat_model.load_model("models/cat_model.cbm")

        # Load preprocessor
        with open("models/preprocessor.pkl", "rb") as f:
            preprocessor = pickle.load(f)

        # Load SHAP explainer
        with open("models/cat_explainer.pkl", "rb") as f:
            cat_explainer = pickle.load(f)

        # Load column names
        with open("models/column_names.pkl", "rb") as f:
            column_names = pickle.load(f)

        # Load categorical column names
        with open("models/catord_cols.pkl", "rb") as f:
            catord_cols = pickle.load(f)

        print("Model and related objects loaded successfully")

    except Exception as e:
        print(f"Error loading model: {str(e)}")


# Health check endpoint
@app.get("/health", response_model=HealthCheck)
def health_check():
    """Check if the API and model are running properly"""
    status = HealthCheck(
        status="ok",
        model_loaded=cat_model is not None,
        preprocessor_loaded=preprocessor is not None,
        explainer_loaded=cat_explainer is not None,
    )
    return status


# Endpoint that returns sample input data
@app.get("/sample", response_model=CADRawInput)
def get_sample_input():
    """Returns a sample input that can be used for testing the API"""
    return CADRawInput()


# Main prediction endpoint
@app.post("/predict", response_model=PredictionResult)
async def predict(data: CADRawInput):
    """
    Make a CAD prediction based on patient data.
    Returns probability, prediction class, and waterfall plot.
    """
    # Check if model is loaded
    if cat_model is None or preprocessor is None or cat_explainer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Preprocess the input data
        processed_data = preprocess_input(data)

        # Make prediction
        prob = cat_model.predict_proba(processed_data)[0, 1]
        probability = prob * 100
        prediction = "CAD" if prob >= 0.5 else "Normal"

        # Generate SHAP values for explanation
        shap_values = cat_explainer(processed_data)

        # Create waterfall plot
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(shap_values[0], show=False)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        waterfall_plot = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close()

        return PredictionResult(
            probability=probability,
            prediction=prediction,
            waterfall_plot=waterfall_plot,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# New response model for force plot as image
class ForceplotImageResponse(BaseModel):
    force_plot_image: str  # Base64 encoded image


# Modified endpoint to return force plot as an image
@app.post("/force_plot_image", response_model=ForceplotImageResponse)
async def force_plot_image(data: CADRawInput):
    """Generate and return a SHAP force plot as an image for the provided data"""
    # Check if model is loaded
    if cat_model is None or preprocessor is None or cat_explainer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Preprocess the input data
        processed_data = preprocess_input(data)

        # Generate SHAP values
        shap_values = cat_explainer(processed_data)

        # Create a force plot as matplotlib figure
        plt.figure(figsize=(24, 12))  # Force plots typically need to be wider
        shap.force_plot(
            base_value=shap_values.base_values[0],
            shap_values=shap_values.values[0],
            features=processed_data.iloc[0],
            feature_names=list(processed_data.columns),
            matplotlib=True,
            show=False,
        )
        
        # Save the plot to a buffer
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=600, bbox_inches="tight")
        buf.seek(0)
        
        # Encode the image
        force_plot_image = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.savefig("test.png")
        plt.close()

        return ForceplotImageResponse(force_plot_image=force_plot_image)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Force plot image error: {str(e)}")


# Keep the original HTML force plot endpoint for backward compatibility
@app.post("/force_plot", response_class=HTMLResponse)
async def force_plot(data: CADRawInput):
    """Generate and return a SHAP force plot for the provided data (HTML version)"""
    # Check if model is loaded
    if cat_model is None or preprocessor is None or cat_explainer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Preprocess the input data
        processed_data = preprocess_input(data)

        # Generate SHAP values
        shap_values = cat_explainer(processed_data)

        # Create force plot HTML
        shap.initjs()
        force_plot_html = shap.force_plot(shap_values[0], matplotlib=False)

        # Construct a complete HTML page with the force plot
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>SHAP Force Plot</title>
            <script src="https://cdn.jsdelivr.net/npm/shap@latest/dist/bundles/shap.js"></script>
        </head>
        <body>
            <h2>SHAP Force Plot Explanation</h2>
            <div id="plot"></div>
            <script>
                var plot = {force_plot_html};
                shap.initjs();
                shap.force_plot(plot, document.getElementById("plot"));
            </script>
        </body>
        </html>
        """

        return html_content

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Force plot error: {str(e)}")


def preprocess_input(data: CADRawInput) -> pd.DataFrame:
    """
    Convert the Pydantic model input to the format expected by the model
    """
    # Convert to dictionary and then to DataFrame
    data_dict = data.dict(exclude={"Cath"})  # Exclude the target variable if present

    # First create a DataFrame with the raw input features
    input_df = pd.DataFrame([data_dict])

    # Fix column names to match what the preprocessor expects
    # Replace spaces with underscores and rename columns as needed
    input_df.columns = [col.replace(" ", "_") for col in input_df.columns]

    # Handle binary encoding for Sex
    if "Sex" in input_df.columns:
        input_df["Sex_Male"] = (input_df["Sex"] == "Male").astype(int)
        input_df.drop("Sex", axis=1, inplace=True)

    # Add "_1" suffix to binary columns to match preprocessor's expected format
    binary_cols = [
        "DM",
        "HTN",
        "Current_Smoker",
        "EX_Smoker",
        "FH",
        "Obesity",
        "CRF",
        "CVA",
        "Airway_disease",
        "Thyroid_Disease",
        "CHF",
        "DLP",
        "Edema",
        "Weak_Peripheral_Pulse",
        "Lung_rales",
        "Systolic_Murmur",
        "Diastolic_Murmur",
        "Typical_Chest_Pain",
        "Dyspnea",
        "Atypical",
        "Nonanginal",
        "Exertional_CP",
        "LowTH_Ang",
        "Q_Wave",
        "St_Elevation",
        "St_Depression",
        "Tinversion",
        "LVH",
        "Poor_R_Progression",
    ]

    for col in binary_cols:
        if col in input_df.columns:
            input_df[f"{col}_1"] = input_df[col]
            input_df.drop(col, axis=1, inplace=True)

    # Convert EF-TTE to EF_TTE if needed
    if "EF-TTE" in input_df.columns:
        input_df["EF_TTE"] = input_df["EF-TTE"]
        input_df.drop("EF-TTE", axis=1, inplace=True)

    # Standardize numeric features
    numeric_cols = [
        "Age",
        "Weight",
        "Length",
        "BMI",
        "BP",
        "PR",
        "FBS",
        "CR",
        "TG",
        "LDL",
        "HDL",
        "BUN",
        "ESR",
        "HB",
        "K",
        "Na",
        "WBC",
        "Lymph",
        "Neut",
        "PLT",
        "EF_TTE",
    ]

    # Use the loaded preprocessor to standardize the numerical variables
    # and ensure we have all the required columns in the correct order

    # First, make sure all the expected columns are present in the input DataFrame
    for col in column_names:
        if col not in input_df.columns:
            input_df[col] = 0  # Default value for missing columns

    # Reorder columns to match the expected order
    input_df = input_df[column_names]

    return input_df


# Run the application
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)