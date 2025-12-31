import os
import logging
import io
import json
import uvicorn
import pandas as pd
import numpy as np
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

# --- CRITICAL: CORS MIDDLEWARE ---
# This allows your frontend to talk to this backend without security blocks.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global State ---
class SessionState:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.target_col: Optional[str] = None
        self.feature_cols: List[str] = []
        self.task_type: str = "unknown"
        self.best_model: Any = None
        self.best_model_code: str = ""
        self.best_score: float = -float("inf")
        self.encoders: Dict = {}
        self.imputer: Any = None

session = SessionState()

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

def clean_data(df, target, features):
    # Select columns
    X = df[features].copy()
    y = df[target].copy()

    # Drop missing target
    y = y.dropna()
    X = X.loc[y.index]

    # Initialize Preprocessors
    session.encoders = {}
    session.imputer = SimpleImputer(strategy='mean')

    # Identify types
    num_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols = X.select_dtypes(exclude=[np.number]).columns

    # 1. Impute Numerics
    if len(num_cols) > 0:
        X[num_cols] = session.imputer.fit_transform(X[num_cols])

    # 2. Encode Categoricals
    if len(cat_cols) > 0:
        for col in cat_cols:
            le = LabelEncoder()
            # Convert to string to handle mixed types safely
            X[col] = X[col].astype(str)
            X[col] = le.fit_transform(X[col])
            session.encoders[col] = le

    return X, y

# --- Endpoints ---

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "alive", 
        "data_loaded": session.df is not None,
        "best_score": session.best_score if session.best_score > -float('inf') else None
    }

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Chat with data context."""
    try:
        client = get_gemini_client()
        
        system_context = "You are an expert Data Science Assistant."
        if session.df is not None:
            system_context += f"\n\nDATA CONTEXT:\n- Columns: {list(session.df.columns)}\n- Rows: {len(session.df)}\n"
            if session.task_type:
                system_context += f"- Task: {session.task_type}\n- Target: {session.target_col}\n"
            if session.best_model:
                system_context += f"- Current Best Model Score: {session.best_score:.4f}\n"

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=f"{system_context}\n\nUser: {request.message}",
            config=types.GenerateContentConfig(temperature=0.7)
        )
        return {"response": response.text or "No response."}
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return {"response": f"I encountered an error: {str(e)}"}

@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files allowed.")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        session.df = df
        
        # Reset State
        session.best_model = None
        session.best_score = -float("inf")
        session.target_col = None
        
        return {
            "message": "Upload successful",
            "rows": len(df),
            "columns": list(df.columns),
            "preview": df.head(3).to_dict()
        }
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze", response_model=ConfigResponse)
async def analyze_data():
    if session.df is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded.")

    client = get_gemini_client()
    
    buffer = io.StringIO()
    session.df.info(buf=buffer)
    info_str = buffer.getvalue()
    head_str = session.df.head(5).to_markdown()
    
    prompt = f"""
    Analyze this dataset:
    {info_str}
    
    Sample:
    {head_str}
    
    Task: Identify the most likely target variable and feature variables.
    Determine if this is classification or regression.
    
    Return JSON:
    {{
      "suggested_target": "column_name",
      "suggested_features": ["col1", "col2"],
      "task_type": "classification"
    }}
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
        raise HTTPException(status_code=500, detail=f"AI Analysis failed: {e}")

@app.post("/set-config")
async def set_config(config: UserConfig):
    if session.df is None:
        raise HTTPException(status_code=400, detail="No dataset.")
    
    session.target_col = config.target
    session.feature_cols = config.features
    session.task_type = config.task_type
    
    return {"message": "Config saved."}

@app.post("/train", response_model=TrainResponse)
async def train_loop(iterations: int = 3):
    if not session.target_col or not session.feature_cols:
        raise HTTPException(status_code=400, detail="Config missing. Run analysis first.")

    try:
        X, y = clean_data(session.df, session.target_col, session.feature_cols)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Data Prep Error: {e}")

    client = get_gemini_client()
    history = []
    
    for i in range(iterations):
        prompt = f"""
        Write Python code to instantiate a scikit-learn model for {session.task_type}.
        Data Shape: {X.shape}
        Goal: Maximize {'Accuracy' if session.task_type == 'classification' else 'R2 Score'}.
        
        Previous attempts: {history}
        Current Best Score: {session.best_score}
        
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
            
            # Sandbox Scope
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
            
            if not model: raise ValueError("Model not defined in code")
            
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            if session.task_type == 'classification':
                score = accuracy_score(y_test, preds)
            else:
                score = r2_score(y_test, preds)
                
            history.append({"code": code, "score": score})
            
            if score > session.best_score:
                session.best_score = score
                session.best_model = model
                session.best_model_code = code
                
        except Exception as e:
            logger.error(f"Loop {i} failed: {e}")
            history.append({"error": str(e)})

    return TrainResponse(
        status="success",
        iterations_run=iterations,
        best_score=session.best_score,
        best_algorithm_code=session.best_model_code
    )

@app.post("/predict")
async def make_prediction(request: PredictionRequest):
    if not session.best_model:
        raise HTTPException(status_code=400, detail="No model trained.")
    
    try:
        input_df = pd.DataFrame([request.input_data])
        
        # Apply strict preprocessing alignment
        num_cols = input_df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0 and session.imputer:
            input_df[num_cols] = session.imputer.transform(input_df[num_cols])
            
        for col, le in session.encoders.items():
            if col in input_df.columns:
                # Handle unseen labels by assigning -1 or a safe default
                input_df[col] = input_df[col].astype(str).map(
                    lambda s: le.transform([s])[0] if s in le.classes_ else 0
                )

        # Reorder columns to match training
        input_df = input_df[session.feature_cols]
        
        pred = session.best_model.predict(input_df)
        return {"prediction": float(pred[0]), "model_used": session.best_model_code}
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
