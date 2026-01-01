import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uuid
import pandas as pd
import numpy as np
import io
import os
from datetime import datetime
import json

# ML Imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

app = FastAPI(title="AutoML Backend")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- In-Memory Storage (Replace with DB for production) ---
# Structure: { model_id: { metadata, data: df, model: sklearn_model, encoders: {}, config: {} } }
models_storage = {}

# --- Pydantic Models ---
class ModelCreate(BaseModel):
    name: Optional[str] = None

class ModelConfig(BaseModel):
    target: str
    features: List[str]
    task_type: str  # 'classification' or 'regression'

class PredictRequest(BaseModel):
    input_data: Dict[str, Any]

class ChatRequest(BaseModel):
    message: str

# --- Helper Functions ---
def get_model_store(model_id: str):
    if model_id not in models_storage:
        raise HTTPException(status_code=404, detail="Model not found")
    return models_storage[model_id]

def serialize_model_metadata(model_id: str, data: dict):
    return {
        "id": model_id,
        "name": data.get("name", "Untitled Model"),
        "status": data.get("status", "created"),
        "created_at": data.get("created_at"),
        "best_score": data.get("best_score"),
        "best_code": data.get("best_code")
    }

# --- Endpoints ---

@app.get("/")
def health_check():
    return {"status": "online", "service": "AutoML Backend"}

@app.get("/models/list")
def list_models():
    # Sort by creation time desc
    sorted_models = sorted(
        [serialize_model_metadata(mid, m) for mid, m in models_storage.items()],
        key=lambda x: x['created_at'],
        reverse=True
    )
    return sorted_models

@app.post("/models/create")
def create_model():
    model_id = str(uuid.uuid4())[:8]
    models_storage[model_id] = {
        "name": f"Model {model_id}",
        "status": "created",
        "created_at": datetime.now().isoformat(),
        "data": None,
        "model": None,
        "config": None,
        "best_score": None,
        "best_code": None
    }
    return serialize_model_metadata(model_id, models_storage[model_id])

@app.post("/models/{model_id}/upload")
async def upload_dataset(model_id: str, file: UploadFile = File(...)):
    store = get_model_store(model_id)
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        # Basic cleanup: drop empty cols/rows
        df.dropna(how='all', inplace=True)
        df.dropna(axis=1, how='all', inplace=True)
        
        # Store in memory
        store['data'] = df
        store['status'] = 'uploaded'
        
        # Return preview info
        preview = df.head(5).fillna('').to_dict(orient='records')
        columns = list(df.columns)
        
        return {
            "rows": len(df),
            "columns": columns,
            "preview": preview
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse CSV: {str(e)}")

@app.post("/models/{model_id}/analyze")
def analyze_data(model_id: str):
    store = get_model_store(model_id)
    df = store.get('data')
    
    if df is None:
        raise HTTPException(status_code=400, detail="No data uploaded")
    
    # Simple heuristic to guess target and task
    columns = list(df.columns)
    
    # Heuristic: Target is often the last column or named explicitly
    potential_targets = [c for c in columns if 'target' in c.lower() or 'label' in c.lower() or 'class' in c.lower() or 'churn' in c.lower() or 'price' in c.lower()]
    suggested_target = potential_targets[0] if potential_targets else columns[-1]
    
    # Heuristic: Task type based on target cardinality
    target_series = df[suggested_target]
    unique_count = target_series.nunique()
    is_numeric = pd.api.types.is_numeric_dtype(target_series)
    
    task_type = "regression"
    if unique_count < 20 or not is_numeric:
        task_type = "classification"
    
    suggested_features = [c for c in columns if c != suggested_target]
    
    return {
        "task_type": task_type,
        "suggested_target": suggested_target,
        "suggested_features": suggested_features
    }

@app.post("/models/{model_id}/config")
def save_config(model_id: str, config: ModelConfig):
    store = get_model_store(model_id)
    store['config'] = config.dict()
    return {"status": "configured"}

@app.post("/models/{model_id}/train")
def train_model(model_id: str, iterations: int = 3):
    store = get_model_store(model_id)
    df = store.get('data')
    config = store.get('config')
    
    if df is None or config is None:
        raise HTTPException(status_code=400, detail="Data or config missing")
    
    target = config['target']
    features = config['features']
    task = config['task_type']
    
    try:
        # 1. Preprocessing
        X = df[features].copy()
        y = df[target].copy()
        
        # Handle Missing Values
        num_imputer = SimpleImputer(strategy='mean')
        cat_imputer = SimpleImputer(strategy='most_frequent')
        
        # Handle Encoding
        encoders = {}
        for col in X.columns:
            if X[col].dtype == 'object' or not pd.api.types.is_numeric_dtype(X[col]):
                # Fill missing for cat
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown')
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                encoders[col] = le
            else:
                # Fill missing for num
                X[col] = X[col].fillna(X[col].mean())
        
        # Encode target if classification
        target_encoder = None
        if task == 'classification' and (y.dtype == 'object' or not pd.api.types.is_numeric_dtype(y)):
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y.astype(str))
            
        # 2. Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 3. Model Selection & Training (Simplified AutoML)
        best_model = None
        best_score = -1 if task == 'classification' else -float('inf')
        best_algo_name = ""
        best_code_snippet = ""
        
        candidates = []
        if task == 'classification':
            candidates = [
                ('RandomForest', RandomForestClassifier(n_estimators=50)),
                ('GradientBoosting', GradientBoostingClassifier(n_estimators=50)),
                ('LogisticRegression', LogisticRegression(max_iter=1000))
            ]
        else:
            candidates = [
                ('RandomForest', RandomForestRegressor(n_estimators=50)),
                ('GradientBoosting', GradientBoostingRegressor(n_estimators=50)),
                ('LinearRegression', LinearRegression())
            ]
            
        for name, model in candidates:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            if task == 'classification':
                score = accuracy_score(y_test, preds)
                # Improve best logic
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_algo_name = name
            else:
                score = r2_score(y_test, preds)
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_algo_name = name

        # 4. Generate Code Snippet
        if task == 'classification':
             import_stmt = f"from sklearn.ensemble import {best_algo_name}Classifier" if 'Forest' in best_algo_name or 'Boosting' in best_algo_name else "from sklearn.linear_model import LogisticRegression"
        else:
             import_stmt = f"from sklearn.ensemble import {best_algo_name}Regressor" if 'Forest' in best_algo_name or 'Boosting' in best_algo_name else "from sklearn.linear_model import LinearRegression"

        best_code_snippet = f"""
import pandas as pd
{import_stmt}
from sklearn.model_selection import train_test_split

# Load Data
df = pd.read_csv('dataset.csv')
X = df[{features}]
y = df['{target}']

# Basic Preprocessing (Simplified)
X = pd.get_dummies(X)
X.fillna(0, inplace=True)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize Model
model = {best_model.__class__.__name__}(n_estimators=100) # Example params
model.fit(X_train, y_train)

print(f"Model Score: {best_score}")
"""

        # 5. Save Artifacts
        store['model'] = best_model
        store['encoders'] = encoders
        store['target_encoder'] = target_encoder
        store['best_score'] = best_score
        store['best_code'] = best_code_snippet.strip()
        store['status'] = 'trained'
        
        return {
            "status": "success",
            "best_score": best_score,
            "best_algorithm": best_algo_name,
            "best_algorithm_code": best_code_snippet.strip()
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/models/{model_id}/predict")
def predict(model_id: str, request: PredictRequest):
    store = get_model_store(model_id)
    model = store.get('model')
    encoders = store.get('encoders')
    
    if not model:
        raise HTTPException(status_code=400, detail="Model not trained yet")
    
    try:
        # Prepare input dataframe
        input_df = pd.DataFrame([request.input_data])
        
        # Apply preprocessing
        for col, le in encoders.items():
            if col in input_df.columns:
                val = str(input_df[col].iloc[0])
                # Handle unseen labels carefully
                if val in le.classes_:
                    input_df[col] = le.transform([val])
                else:
                    # Fallback for demo
                    input_df[col] = 0 
        
        # Ensure correct column order
        config = store.get('config')
        if config:
            features = config['features']
            # Fill missing cols with 0
            for f in features:
                if f not in input_df.columns:
                    input_df[f] = 0
            input_df = input_df[features]
            
        prediction = model.predict(input_df)[0]
        
        # Decode target if necessary
        target_encoder = store.get('target_encoder')
        if target_encoder:
            prediction = target_encoder.inverse_transform([int(prediction)])[0]
            
        # Convert numpy types to python native for JSON serialization
        if isinstance(prediction, (np.int64, np.int32)):
            prediction = int(prediction)
        elif isinstance(prediction, (np.float64, np.float32)):
            prediction = float(prediction)
            
        return {"prediction": prediction}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/models/{model_id}/chat")
def chat_with_context(model_id: str, request: ChatRequest):
    # Simple rule-based chat for the backend
    # The Frontend uses Gemini for advanced reasoning. 
    # This endpoint can be used for data-specific queries that need backend calculation.
    store = get_model_store(model_id)
    msg = request.message.lower()
    
    response = "I can help you with your model."
    
    if "rows" in msg or "size" in msg:
        df = store.get('data')
        rows = len(df) if df is not None else 0
        response = f"The dataset has {rows} rows."
    elif "accuracy" in msg or "score" in msg:
        score = store.get('best_score')
        response = f"The best model score is {score:.2f}" if score else "Model not trained yet."
    elif "status" in msg:
        response = f"Current status: {store.get('status', 'unknown')}"
    else:
        response = "I received your message. Ask me about rows, score, or status."

    return {"response": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
