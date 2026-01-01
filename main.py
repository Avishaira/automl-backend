import os
import logging
import io
import json
import uuid
import shutil
import glob
import pickle
import uvicorn
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from google import genai
from google.genai import types

# ML Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, r2_score

# Initialize Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AutoML_Brain")

app = FastAPI()

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- FILE SYSTEM STORAGE ---
# Uses a local directory. Note: On Render free tier, this wipes on restart.
# For persistence, you'd need a persistent disk or cloud storage (S3/GCS).
BASE_DIR = Path("models_storage")
try:
    BASE_DIR.mkdir(exist_ok=True)
    logger.info(f"Storage initialized at {BASE_DIR.absolute()}")
except Exception as e:
    logger.error(f"Failed to create storage dir: {e}")

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    message: str

class ConfigResponse(BaseModel):
    suggested_target: str
    suggested_features: List[str]
    task_type: str

class UserConfig(BaseModel):
    target: str
    features: List[str]
    task_type: str

class PredictionRequest(BaseModel):
    input_data: Dict[str, Any]

class TrainResponse(BaseModel):
    status: str
    iterations_run: int
    best_score: float
    best_algorithm_code: str

# --- Helper Functions ---
def get_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY missing.")
    return genai.Client(api_key=api_key)

def load_model_context(model_id: str):
    """Loads metadata and data for a specific model."""
    model_dir = BASE_DIR / model_id
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    
    context = {"id": model_id, "dir": model_dir}
    
    # Load Metadata
    meta_path = model_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            context["meta"] = json.load(f)
    else:
        context["meta"] = {"created_at": str(datetime.now()), "status": "new"}

    # Load DF if exists
    csv_path = model_dir / "original.csv"
    if csv_path.exists():
        context["df"] = pd.read_csv(csv_path)
    else:
        context["df"] = None
        
    return context

def save_metadata(model_id: str, meta: dict):
    with open(BASE_DIR / model_id / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

def clean_data(df, target, features):
    X = df[features].copy()
    y = df[target].copy()
    y = y.dropna()
    X = X.loc[y.index]

    imputer = SimpleImputer(strategy='mean')
    encoders = {}

    num_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols = X.select_dtypes(exclude=[np.number]).columns

    if len(num_cols) > 0:
        X[num_cols] = imputer.fit_transform(X[num_cols])

    if len(cat_cols) > 0:
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = X[col].astype(str)
            X[col] = le.fit_transform(X[col])
            encoders[col] = le
            
    return X, y, imputer, encoders

# --- Endpoints ---

@app.get("/")
async def root():
    return {"status": "alive", "version": "2.0", "storage": str(BASE_DIR)}

# 1. LIST MODELS
@app.get("/models/list")
async def list_models():
    models = []
    if not BASE_DIR.exists():
        return []
        
    for path in BASE_DIR.iterdir():
        if path.is_dir():
            meta_path = path / "metadata.json"
            if meta_path.exists():
                try:
                    with open(meta_path) as f:
                        models.append(json.load(f))
                except Exception as e:
                    logger.error(f"Error reading meta for {path}: {e}")
                    
    # Sort by date descending
    models.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    return models

# 2. CREATE MODEL
@app.post("/models/create")
async def create_model():
    model_id = str(uuid.uuid4())[:8]
    model_dir = BASE_DIR / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    
    meta = {
        "id": model_id,
        "name": f"Model {model_id}",
        "created_at": datetime.now().isoformat(),
        "status": "waiting_for_data",
        "best_score": None
    }
    save_metadata(model_id, meta)
    return meta

# 3. UPLOAD DATASET (Per Model)
@app.post("/models/{model_id}/upload")
async def upload_dataset(model_id: str, file: UploadFile = File(...)):
    ctx = load_model_context(model_id)
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="CSV only")
    
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    
    # Save Original
    df.to_csv(ctx['dir'] / "original.csv", index=False)
    
    # Update Meta
    ctx['meta']['status'] = "data_uploaded"
    ctx['meta']['rows'] = len(df)
    ctx['meta']['cols'] = list(df.columns)
    save_metadata(model_id, ctx['meta'])
    
    return {
        "message": "Upload successful",
        "rows": len(df),
        "columns": list(df.columns),
        "preview": df.head(3).to_dict()
    }

# 4. ANALYZE (Per Model)
@app.post("/models/{model_id}/analyze", response_model=ConfigResponse)
async def analyze_data(model_id: str):
    ctx = load_model_context(model_id)
    if ctx['df'] is None:
        raise HTTPException(status_code=400, detail="No data uploaded for this model")

    client = get_gemini_client()
    buffer = io.StringIO()
    ctx['df'].info(buf=buffer)
    
    prompt = f"""
    Analyze dataset structure:
    {buffer.getvalue()}
    Sample:
    {ctx['df'].head().to_markdown()}
    Return JSON: {{ "suggested_target": "str", "suggested_features": ["str"], "task_type": "classification"|"regression" }}
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        return ConfigResponse(**json.loads(response.text))
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 5. SET CONFIG
@app.post("/models/{model_id}/config")
async def set_config(model_id: str, config: UserConfig):
    ctx = load_model_context(model_id)
    ctx['meta'].update({
        "target": config.target,
        "features": config.features,
        "task_type": config.task_type,
        "status": "configured"
    })
    save_metadata(model_id, ctx['meta'])
    return {"message": "Config saved"}

# 6. TRAIN (Per Model + Artifact Saving)
@app.post("/models/{model_id}/train", response_model=TrainResponse)
async def train_loop(model_id: str, iterations: int = 3):
    ctx = load_model_context(model_id)
    meta = ctx['meta']
    
    if "target" not in meta:
        raise HTTPException(status_code=400, detail="Not configured")

    try:
        X, y, imputer, encoders = clean_data(ctx['df'], meta['target'], meta['features'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Save Cleaning Artifacts
        with open(ctx['dir'] / "clean_objects.pkl", "wb") as f:
            pickle.dump({"imputer": imputer, "encoders": encoders}, f)
            
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Data Prep Error: {e}")

    client = get_gemini_client()
    best_score = -float("inf")
    best_model = None
    best_code = ""
    history = []
    
    for i in range(iterations):
        prompt = f"""
        Write Python code to instantiate a scikit-learn model for {meta['task_type']}.
        Data Shape: {X.shape}
        Goal: Maximize {'Accuracy' if meta['task_type'] == 'classification' else 'R2 Score'}.
        Previous attempts: {history}
        Current Best: {best_score}
        Return ONLY code to instantiate 'model'. No imports.
        Example: model = RandomForestClassifier(n_estimators=100)
        """
        
        try:
            resp = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.7)
            )
            code = resp.text.strip().replace("```python", "").replace("```", "").strip()
            
            scope = {
                "RandomForestClassifier": RandomForestClassifier,
                "RandomForestRegressor": RandomForestRegressor,
                "GradientBoostingClassifier": GradientBoostingClassifier,
                "GradientBoostingRegressor": GradientBoostingRegressor,
                "LinearRegression": LinearRegression,
                "LogisticRegression": LogisticRegression,
                "SVC": SVC, "SVR": SVR
            }
            exec(code, {}, scope)
            model = scope.get("model")
            
            if model:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                score = accuracy_score(y_test, preds) if meta['task_type'] == 'classification' else r2_score(y_test, preds)
                
                history.append({"code": code, "score": score})
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_code = code
                
        except Exception as e:
            logger.error(f"Loop {i} failed: {e}")

    # Save Best Model Artifacts
    if best_model:
        with open(ctx['dir'] / "best_model.pkl", "wb") as f:
            pickle.dump(best_model, f)
        
        with open(ctx['dir'] / "model_code.py", "w") as f:
            f.write(best_code)
            
        ctx['meta']['best_score'] = best_score
        ctx['meta']['status'] = "trained"
        ctx['meta']['best_code'] = best_code
        save_metadata(model_id, ctx['meta'])

    return TrainResponse(
        status="success", iterations_run=iterations,
        best_score=best_score, best_algorithm_code=best_code
    )

# 7. PREDICT (Loads from Disk)
@app.post("/models/{model_id}/predict")
async def make_prediction(model_id: str, request: PredictionRequest):
    ctx = load_model_context(model_id)
    
    # Load Artifacts
    try:
        with open(ctx['dir'] / "best_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open(ctx['dir'] / "clean_objects.pkl", "rb") as f:
            clean_objs = pickle.load(f)
            imputer = clean_objs['imputer']
            encoders = clean_objs['encoders']
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Model not trained yet")

    try:
        input_df = pd.DataFrame([request.input_data])
        
        # Apply Saved Cleaning Logic
        num_cols = input_df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            input_df[num_cols] = imputer.transform(input_df[num_cols])
            
        for col, le in encoders.items():
            if col in input_df.columns:
                # Handle safe transform
                input_df[col] = input_df[col].astype(str).map(
                    lambda s: le.transform([s])[0] if s in le.classes_ else 0
                )
        
        # Reorder
        input_df = input_df[ctx['meta']['features']]
        pred = model.predict(input_df)
        return {"prediction": float(pred[0]), "model_used": ctx['meta'].get('best_code')}
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 8. CHAT (Context Aware)
@app.post("/models/{model_id}/chat")
async def chat_endpoint(model_id: str, request: ChatRequest):
    ctx = load_model_context(model_id)
    client = get_gemini_client()
    
    meta = ctx['meta']
    system_context = f"You are an expert Data Science Assistant analyzing Model ID: {model_id}."
    
    if meta.get('rows'):
        system_context += f"\nData: {meta['rows']} rows, Columns: {meta.get('cols')}"
    if meta.get('best_score'):
        system_context += f"\nBest Score: {meta['best_score']:.4f}\nCode: {meta.get('best_code')}"
        
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=f"{system_context}\n\nUser: {request.message}",
        config=types.GenerateContentConfig(temperature=0.7)
    )
    return {"response": response.text}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    logger.info("Starting Backend...")
    uvicorn.run(app, host="0.0.0.0", port=port)
