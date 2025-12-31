import os
import logging
import io
import traceback
import uvicorn
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from google import genai
from google.genai import types

# ML Imports for the "Sandbox"
from sklearn.model_selection import train_test_split, cross_val_score
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

# --- Global In-Memory State (For Demo Purposes) ---
# In a real production app, use a database (Firestore/SQL) and Object Storage (S3).
class SessionState:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.target_col: Optional[str] = None
        self.feature_cols: List[str] = []
        self.task_type: str = "unknown" # 'classification' or 'regression'
        self.best_model: Any = None
        self.best_model_code: str = ""
        self.best_score: float = -float("inf")
        self.encoders: Dict = {}
        self.imputer: Any = None

session = SessionState()

# --- Pydantic Models ---
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
    message: str

# --- Helper Functions ---

def get_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY missing.")
    return genai.Client(api_key=api_key)

def clean_data(df, target, features):
    """Prepares data for ML: Handles missing values and encoding."""
    X = df[features].copy()
    y = df[target].copy()

    # 1. Handle Missing Values
    # Simple strategy: Fill numeric with mean, categorical with mode
    # For this demo, we drop rows with missing target
    y = y.dropna()
    X = X.loc[y.index]

    # Impute X
    num_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols = X.select_dtypes(exclude=[np.number]).columns
    
    # Store encoders for prediction later
    session.encoders = {}
    
    if len(cat_cols) > 0:
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = X[col].astype(str)
            X[col] = le.fit_transform(X[col])
            session.encoders[col] = le
    
    # Simple Mean Imputation for numbers
    if len(num_cols) > 0:
        imp = SimpleImputer(strategy='mean')
        X[num_cols] = imp.fit_transform(X[num_cols])
        session.imputer = imp

    return X, y

# --- Endpoints ---

@app.get("/")
async def root():
    return {
        "status": "ready", 
        "data_loaded": session.df is not None,
        "best_score": session.best_score
    }

@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Step 1: Upload and read the CSV."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files allowed.")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        session.df = df
        session.best_model = None # Reset previous model
        session.best_score = -float("inf")
        
        return {
            "message": "Dataset uploaded successfully",
            "rows": df.shape[0],
            "columns": list(df.columns),
            "preview": df.head(3).to_dict()
        }
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze", response_model=ConfigResponse)
async def analyze_data():
    """Step 2: AI analyzes data and suggests Target/Features."""
    if session.df is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded.")

    client = get_gemini_client()
    
    # Prepare metadata for Gemini
    buffer = io.StringIO()
    session.df.info(buf=buffer)
    info_str = buffer.getvalue()
    head_str = session.df.head(5).to_markdown()
    
    prompt = f"""
    Analyze this dataset structure and sample:
    
    {info_str}
    
    Sample Data:
    {head_str}
    
    1. Identify the most likely Dependent Variable (Target) column.
    2. Identify relevant Independent Variables (Features).
    3. Determine if this is a 'classification' or 'regression' task.
    
    Respond in strict JSON format:
    {{
      "suggested_target": "column_name",
      "suggested_features": ["col1", "col2", ...],
      "task_type": "classification" or "regression"
    }}
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2
            )
        )
        import json
        result = json.loads(response.text)
        return ConfigResponse(**result)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"AI Analysis failed: {str(e)}")

@app.post("/set-config")
async def set_config(config: UserConfig):
    """Step 3 & 4: User Validates and Confirms variables."""
    if session.df is None:
        raise HTTPException(status_code=400, detail="No dataset.")
    
    # Validate columns exist
    missing = [c for c in config.features + [config.target] if c not in session.df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Columns not found: {missing}")

    session.target_col = config.target
    session.feature_cols = config.features
    session.task_type = config.task_type
    
    return {"message": "Configuration saved. Ready to train."}

@app.post("/train", response_model=TrainResponse)
async def train_loop(iterations: int = 3):
    """Step 5: The Loop. AI proposes code, System runs it, Score improves."""
    if not session.target_col or not session.feature_cols:
        raise HTTPException(status_code=400, detail="Configuration not set.")

    # Prepare Data
    try:
        X, y = clean_data(session.df, session.target_col, session.feature_cols)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Data prep failed: {e}")

    client = get_gemini_client()
    history = [] # Keep track of what we tried
    
    for i in range(iterations):
        logger.info(f"--- AutoML Loop Iteration {i+1} ---")
        
        # 1. Ask Gemini for Model Code
        prompt = f"""
        Task: Write Python code to instantiate a scikit-learn model for {session.task_type}.
        Data Info: {X.shape[1]} features, {X.shape[0]} rows.
        Goal: Maximize {'Accuracy' if session.task_type == 'classification' else 'R2 Score'}.
        
        Previous attempts: {history}
        Current Best Score: {session.best_score}
        
        Return ONLY the python code to instantiate the model object named 'model'.
        Do not include imports. Use standard sklearn params.
        Example: model = RandomForestClassifier(n_estimators=100, max_depth=5)
        """
        
        resp = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.7)
        )
        
        code_snippet = resp.text.strip().replace("```python", "").replace("```", "").strip()
        
        # 2. "Sandbox" Execution (Restricted Environment)
        local_scope = {
            "RandomForestClassifier": RandomForestClassifier,
            "RandomForestRegressor": RandomForestRegressor,
            "GradientBoostingClassifier": GradientBoostingClassifier,
            "GradientBoostingRegressor": GradientBoostingRegressor,
            "LinearRegression": LinearRegression,
            "LogisticRegression": LogisticRegression,
            "SVC": SVC,
            "SVR": SVR
        }
        
        try:
            # DANGEROUS IN PROD: We use exec() here to simulate the agent writing code.
            # In a real app, strict sandboxing is required.
            exec(code_snippet, {}, local_scope)
            model = local_scope.get("model")
            
            if not model:
                raise ValueError("AI code did not define 'model'")
                
            # 3. Train & Evaluate
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            if session.task_type == 'classification':
                score = accuracy_score(y_test, preds)
            else:
                score = r2_score(y_test, preds)
            
            history.append({"code": code_snippet, "score": score})
            
            # 4. Update Best
            if score > session.best_score:
                session.best_score = score
                session.best_model = model
                session.best_model_code = code_snippet
                logger.info(f"New Record! Score: {score} | Code: {code_snippet}")
                
        except Exception as e:
            logger.error(f"Attempt failed: {e}")
            history.append({"code": code_snippet, "error": str(e)})
            
    return TrainResponse(
        status="success",
        iterations_run=iterations,
        best_score=session.best_score,
        best_algorithm_code=session.best_model_code,
        message=f"AutoML Complete. Best model achieved score: {session.best_score:.4f}"
    )

@app.post("/predict")
async def make_prediction(request: PredictionRequest):
    """Step 6: Make predictions using the winning model."""
    if not session.best_model:
        raise HTTPException(status_code=400, detail="No model trained yet.")
        
    try:
        # Convert input dict to DataFrame
        input_df = pd.DataFrame([request.input_data])
        
        # Apply same encoding/imputation
        if session.imputer:
            num_cols = input_df.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 0:
                input_df[num_cols] = session.imputer.transform(input_df[num_cols])
                
        for col, le in session.encoders.items():
            if col in input_df.columns:
                # Handle unknown categories safely
                input_df[col] = input_df[col].astype(str).map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)

        prediction = session.best_model.predict(input_df[session.feature_cols])
        return {"prediction": float(prediction[0]), "model_used": session.best_model_code}
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        # Return helpful error if features match failed
        raise HTTPException(status_code=500, detail=f"Prediction failed. Ensure inputs match features: {session.feature_cols}. Error: {e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
