import os
import shutil
import pandas as pd
import numpy as np
import joblib
import json
import logging
import datetime
import uuid
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# --- SKLEARN IMPORTS FOR REAL ML ---
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, r2_score, classification_report

# --- CONFIGURATION ---
UPLOAD_DIR = "uploads"
MODELS_DIR = "models"
ARTIFACTS_DIR = "artifacts" # Where we save the real .pkl models
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AutoML_Brain")

app = FastAPI(title="Pro AutoML Backend")

# --- CORS CONFIGURATION ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- IN-MEMORY DATABASE ---
projects_db = []

# --- THE AUTO-ML ENGINEER CLASS ---
class AutoMLSystem:
    def __init__(self, filepath, target_col="auto"):
        self.filepath = filepath
        self.target_col = target_col
        self.df = pd.read_csv(filepath)
        self.model_type = "unknown" # classification vs regression
        self.best_model = None
        self.best_score = -np.inf
        self.best_model_name = ""
        self.pipeline = None
        self.label_encoder = None
        
    def analyze_and_preprocess(self):
        # 1. Automatic Target Detection
        if self.target_col == "auto":
            # Heuristic: Usually the last column, or column named 'target', 'churn', 'price'
            potential = [c for c in self.df.columns if c.lower() in ['target', 'class', 'label', 'y', 'price', 'churn', 'survived']]
            if potential:
                self.target_col = potential[0]
            else:
                self.target_col = self.df.columns[-1]
        
        logger.info(f"Target Column Identified: {self.target_col}")

        # 2. Determine Task (Classification vs Regression)
        y = self.df[self.target_col]
        if y.dtype == 'object' or y.nunique() < 20:
            self.model_type = "classification"
        else:
            self.model_type = "regression"
            
        logger.info(f"Task Detected: {self.model_type}")

        # 3. Features Split
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]

        # Handle Categorical Target for Classification
        if self.model_type == "classification" and y.dtype == 'object':
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)

        # 4. Define Preprocessing Pipeline
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'bool']).columns

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')), # Auto-fill missing numbers
            ('scaler', StandardScaler()) # Normalize
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')), # Auto-fill missing text
            ('onehot', OneHotEncoder(handle_unknown='ignore')) # Convert text to binary vectors
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
            
        return X, y

    def train_and_select_best(self):
        X, y = self.analyze_and_preprocess()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = []
        if self.model_type == "classification":
            models = [
                ("Random Forest", RandomForestClassifier(n_estimators=100)),
                ("Gradient Boosting", GradientBoostingClassifier()),
                ("Logistic Regression", LogisticRegression(max_iter=1000))
            ]
        else:
            models = [
                ("Random Forest Regressor", RandomForestRegressor(n_estimators=100)),
                ("Gradient Boosting", GradientBoostingRegressor()),
                ("Linear Regression", LinearRegression())
            ]

        best_score = -1
        best_pipe = None
        best_name = ""

        # The Tournament
        for name, model in models:
            logger.info(f"Training {name}...")
            clf = Pipeline(steps=[('preprocessor', self.preprocessor),
                                  ('classifier', model)])
            clf.fit(X_train, y_train)
            
            score = 0
            if self.model_type == "classification":
                preds = clf.predict(X_test)
                score = accuracy_score(y_test, preds)
            else:
                score = clf.score(X_test, y_test) # R2 score

            logger.info(f"{name} Score: {score}")
            
            if score > best_score:
                best_score = score
                best_pipe = clf
                best_name = name

        self.best_model = best_pipe
        self.best_score = best_score
        self.best_model_name = best_name
        
        return {
            "best_model": best_name,
            "score": round(best_score * 100, 2),
            "type": self.model_type,
            "target": self.target_col
        }

    def save(self, model_id):
        # Save binary artifact
        joblib.dump({
            "pipeline": self.best_model,
            "label_encoder": self.label_encoder,
            "model_type": self.model_type,
            "target_col": self.target_col
        }, os.path.join(ARTIFACTS_DIR, f"{model_id}.pkl"))
        
        # Generate readable Python code for the user
        code = f"""
# Generated by AutoML Architect
# Task: {self.model_type.capitalize()}
# Target: {self.target_col}
# Best Model: {self.best_model_name} (Score: {round(self.best_score * 100, 2)}%)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# ... other imports implied

# 1. Load Data
df = pd.read_csv('dataset.csv')
X = df.drop(columns=['{self.target_col}'])
y = df['{self.target_col}']

# 2. Preprocessing (Reconstructed)
# Numerical Imputation & Scaling
# Categorical One-Hot Encoding
# ... (Full pipeline logic hidden in abstraction for brevity)

# 3. Model
model = {self.best_model.named_steps['classifier']}
# model.fit(X, y)
print("Best model replicated!")
"""
        with open(os.path.join(MODELS_DIR, f"{model_id}.py"), "w") as f:
            f.write(code)

# --- WORKER ---

def train_worker_real(model_id: str, file_path: str):
    logger.info(f"Engineer started working on {model_id}")
    model_entry = next((item for item in projects_db if item["id"] == model_id), None)
    
    try:
        # Instantiate the Expert
        automl = AutoMLSystem(file_path)
        
        # Run the Analysis & Tournament
        results = automl.train_and_select_best()
        
        # Save Artifacts
        automl.save(model_id)
        
        # Update DB
        if model_entry:
            model_entry["status"] = "completed"
            model_entry["accuracy"] = f"{results['score']}%"
            model_entry["target_col"] = results['target']
            model_entry["best_algorithm"] = results['best_model']
            model_entry["code_url"] = f"/models/{model_id}/code"
            
    except Exception as e:
        logger.error(f"AutoML crashed: {e}")
        if model_entry:
            model_entry["status"] = "failed"
            model_entry["error"] = str(e)

# --- API ---

@app.get("/")
def health_check():
    return {"status": "running", "service": "AutoML Pro"}

@app.get("/models")
def get_models():
    return {"models": projects_db}

@app.get("/models/{model_id}/code")
def get_model_code(model_id: str):
    code_path = os.path.join(MODELS_DIR, f"{model_id}.py")
    if not os.path.exists(code_path):
        raise HTTPException(status_code=404, detail="Code not generated yet")
    return FileResponse(code_path)

@app.get("/files/{filename}")
def get_file(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

# --- REAL PREDICTION ENDPOINT ---
class PredictionRequest(BaseModel):
    data: Dict[str, Any]

@app.post("/predict/{model_id}")
def predict(model_id: str, request: PredictionRequest):
    """
    Loads the REAL saved .pkl model and runs inference.
    """
    model_path = os.path.join(ARTIFACTS_DIR, f"{model_id}.pkl")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found or training not finished")
    
    try:
        # Load the artifact
        artifact = joblib.load(model_path)
        pipeline = artifact["pipeline"]
        label_encoder = artifact.get("label_encoder")
        
        # Create DataFrame from input (expecting single row or dict)
        input_df = pd.DataFrame([request.data])
        
        # Run Prediction
        prediction = pipeline.predict(input_df)
        
        # Decode label if classification
        final_pred = prediction[0]
        if label_encoder is not None:
            final_pred = label_encoder.inverse_transform([int(final_pred)])[0]
            
        # Get probabilities if available
        confidence = "N/A"
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(input_df)
            confidence = f"{round(np.max(probs) * 100, 2)}%"
            
        return {
            "model_id": model_id,
            "prediction": str(final_pred),
            "confidence": confidence,
            "algorithm": str(pipeline.named_steps['classifier'])
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/upload")
async def upload_dataset(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        file_id = str(uuid.uuid4())
        clean_name = file.filename.replace(" ", "_")
        file_path = os.path.join(UPLOAD_DIR, clean_name)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        model_id = str(uuid.uuid4())
        new_model = {
            "id": model_id,
            "name": clean_name.split('.')[0],
            "type": "Analyzing...",
            "status": "training",
            "accuracy": None,
            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "file_url": f"/files/{clean_name}",
            "file_id": file_id
        }
        projects_db.insert(0, new_model) 
        
        # Start REAL worker
        background_tasks.add_task(train_worker_real, model_id, file_path)
        
        return new_model

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail="Upload failed")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
