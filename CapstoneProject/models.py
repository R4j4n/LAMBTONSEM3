# models.py
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class CADRawInput(BaseModel):
    """Pydantic model for raw CAD prediction input data"""

    # Patient demographics
    Age: int = Field(50, description="Patient age in years")
    Weight: float = Field(89.0, description="Patient weight in kg")
    Length: float = Field(159.0, description="Patient height in cm")
    Sex: str = Field("Female", description="Patient sex (Male or Female)")
    BMI: float = Field(35.2, description="Body Mass Index")

    # Binary medical history features (0=No, 1=Yes)
    DM: int = Field(0, description="Diabetes Mellitus")
    HTN: int = Field(0, description="Hypertension")
    Current_Smoker: int = Field(0, description="Current smoker status")
    EX_Smoker: int = Field(0, description="Former smoker status")
    FH: int = Field(1, description="Family History of CAD")
    Obesity: int = Field(1, description="Obesity")
    CRF: int = Field(0, description="Chronic Renal Failure")
    CVA: int = Field(0, description="Cerebrovascular Accident")
    Airway_disease: int = Field(0, description="Airway disease")
    Thyroid_Disease: int = Field(0, description="Thyroid Disease")
    CHF: int = Field(0, description="Congestive Heart Failure")
    DLP: int = Field(1, description="Dyslipidemia")

    # Vital signs
    BP: int = Field(110, description="Blood Pressure (systolic)")
    PR: int = Field(65, description="Pulse Rate")

    # Physical examination findings (0=No, 1=Yes)
    Edema: int = Field(0, description="Edema")
    Weak_Peripheral_Pulse: int = Field(0, description="Weak Peripheral Pulse")
    Lung_rales: int = Field(0, description="Lung rales")
    Systolic_Murmur: int = Field(0, description="Systolic Murmur")
    Diastolic_Murmur: int = Field(0, description="Diastolic Murmur")

    # Symptoms (0=No, 1=Yes)
    Typical_Chest_Pain: int = Field(1, description="Typical Chest Pain")
    Dyspnea: int = Field(1, description="Shortness of Breath")
    Atypical: int = Field(0, description="Atypical Chest Pain")
    Nonanginal: int = Field(0, description="Nonanginal Pain")
    Exertional_CP: int = Field(0, description="Exertional Chest Pain")
    LowTH_Ang: int = Field(0, description="Low Threshold Angina")

    # Functional status
    Function_Class: int = Field(2, description="NYHA Functional Class (1-4)")

    # ECG findings (0=No, 1=Yes)
    Q_Wave: int = Field(0, description="Q Wave on ECG")
    St_Elevation: int = Field(0, description="ST Elevation on ECG")
    St_Depression: int = Field(0, description="ST Depression on ECG")
    Tinversion: int = Field(0, description="T-wave Inversion on ECG")
    LVH: int = Field(0, description="Left Ventricular Hypertrophy")
    Poor_R_Progression: int = Field(0, description="Poor R Progression on ECG")

    # Laboratory values
    FBS: float = Field(90.0, description="Fasting Blood Sugar")
    CR: float = Field(0.9, description="Creatinine")
    TG: float = Field(95.0, description="Triglycerides")
    LDL: float = Field(140.0, description="Low-Density Lipoprotein")
    HDL: float = Field(35.0, description="High-Density Lipoprotein")
    BUN: float = Field(18.0, description="Blood Urea Nitrogen")
    ESR: float = Field(14.0, description="Erythrocyte Sedimentation Rate")
    HB: float = Field(12.9, description="Hemoglobin")
    K: float = Field(4.3, description="Potassium")
    Na: float = Field(143.0, description="Sodium")
    WBC: float = Field(7900.0, description="White Blood Cell count")
    Lymph: float = Field(30.0, description="Lymphocyte count (%)")
    Neut: float = Field(65.0, description="Neutrophil count (%)")
    PLT: float = Field(260.0, description="Platelet count")

    # Cardiac imaging
    EF_TTE: float = Field(55.0, description="Ejection Fraction from TTE (%)")
    Region_RWMA: int = Field(0, description="Regional Wall Motion Abnormality")
    VHD: int = Field(
        0, description="Valvular Heart Disease (0=None, 1=Mild, 2=Moderate, 3=Severe)"
    )

    # Optional output label (not required for prediction)
    Cath: Optional[str] = Field(
        None, description="Catheterization result (Normal or CAD)"
    )


# Output model for prediction results
class PredictionResult(BaseModel):
    probability: float = Field(..., description="Probability of CAD as percentage")
    prediction: str = Field(..., description="Prediction class (CAD or Normal)")
    waterfall_plot: str = Field(
        ..., description="Base64 encoded SHAP waterfall plot image"
    )


# Simple response model for health check
class HealthCheck(BaseModel):
    status: str
    model_loaded: bool
    preprocessor_loaded: bool
    explainer_loaded: bool
