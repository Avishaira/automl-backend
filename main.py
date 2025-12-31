import os
import shutil
import pandas as pd
import numpy as np
import joblib
import json
import logging
import datetime
import uuid
import traceback
import time
import io
import warnings
import requests # Added for REST fallback
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

import google.generativeai as genai
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error

# --- CONFIGURATION ---
UPLOAD_DIR = "uploads"
ARTIFACTS_DIR = "artifacts"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AutoML_Brain")

# Suppress Deprecation Warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")

# --- IN-MEMORY DATABASE ---
projects_db = []
latest_project_id = None

def add_log(model_id: str, message: str):
    entry = next((item for item in projects_db if item["id"] == model_id), None)
    if entry:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        entry["logs"].append(log_entry)
        logger.info(f"Project {model_id}: {message}")

# --- GEMINI BRAIN (With Robust Fallback) ---
class GeminiBrain:
    def __init__(self):
        # Checks for environment variable first, then falls back to hardcoded key
        env_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.api_key = env_key or "AIzaSyBgqQKWB_FKtTDhxlV36ZiWRwrAaPqZzZY"
        
        self.active = False
        
        if self.api_key:
            # DEBUG LOG: Print first 5 chars to verify key is loaded
            masked_key = self.api_key[:5] + "..." if self.api_key else "None"
            logger.info(f"GeminiBrain initializing with Key: {masked_key}")
            
            try:
                genai.configure(api_key=self.api_key)
                self.active = True
            except Exception as e:
                logger.error(f"Failed to configure Gemini Lib: {e}")
        else:
            logger.warning("Gemini API Key missing! Set GEMINI_API_KEY environment variable.")

    def _generate_via_rest(self, prompt, model="gemini-1.5-flash"):
        """
        Direct REST API fallback. Bypasses the python library entirely.
        """
        logger.info(f"Attempting Raw REST API Connection ({model})...")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
        headers = {'Content-Type': 'application/json'}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            if response.status_code == 200:
                data = response.json()
                # Handle cases where response might be blocked/empty
                if 'candidates' in data and data['candidates']:
                    return data['candidates'][0]['content']['parts'][0]['text']
                else:
                    return "No content returned (Safety filter or empty response)."
            else:
                logger.error(f"REST API Failed: {response.status_code} - {response.text}")
                raise Exception(f"REST API Error {response.status_code}")
        except Exception as e:
            raise Exception(f"REST Connection failed: {str(e)}")

    def test_connection(self):
        """Robust connectivity test that tries multiple methods"""
        if not self.active: 
            return False, "API Key not found in environment variables."
        
        # List of models to try in order of preference
        candidates = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']
        
        # 1. Try Python Library
        for model_name in candidates:
            try:
                model = genai.GenerativeModel(model_name)
                model.generate_content("Ping")
                return True, f"Connected via Library ({model_name})"
            except Exception:
                continue # Silently try next

        # 2. Try REST API (if library fails completely)
        try:
            self._generate_via_rest("Ping", model="gemini-1.5-flash")
            return True, "Connected via REST Fallback"
        except Exception:
            pass
            
        return False, "All connection attempts (Library & REST) failed."

    def chat(self, message: str):
        if not self.active: return "AI is unavailable (API Key missing)."
        
        # 1. Try Standard Library (Robust Loop)
        candidates = [
            'gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro', 'gemini-1.0-pro'
        ]
        
        for model_name in candidates:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(message)
                return response.text
            except Exception:
                continue # Silently try next
        
        # 2. If all library attempts fail, Try REST API
        try:
            return self._generate_via_rest(message)
        except Exception as e:
            return f"Error communicating with Gemini (Both Lib & REST failed). Details: {str(e)}"

    def analyze(self, df_head_csv, columns_list):
        # Simplified analyze that uses the chat function to leverage the same robust logic
        prompt = f"""
        Act as a Data Scientist. Analyze this dataset structure.
        Columns: {columns_list}
        Sample Data:
        {df_head_csv}
        
        Recommend:
        1. Target Column.
        2. Columns to DROP.
        3. Problem Type.
        
        Return JSON ONLY:
        {{ "target_suggestion": "col", "drop_suggestions": [], "type": "classification", "reasoning": "..." }}
        """
        try:
            res_text = self.chat(prompt)
            # clean markdown
            clean = res_text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean)
        except:
            return None

# --- APP LIFESPAN & STARTUP ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup Logic
    logger.info("🚀 Booting AutoML Brain...")
    brain = GeminiBrain()
    success, msg = brain.test_connection()
    if success:
        logger.info(f"✅ GEMINI AI: CONNECTED. ({msg})")
    else:
        logger.error(f"❌ GEMINI AI: FAILED. ({msg})")
    yield
    # Shutdown Logic (if any)

app = FastAPI(title="Gemini Expert AutoML", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ML ENGINEER ---
class MLEngineer:
    def __init__(self, model_id, filepath):
        self.model_id = model_id
        self.filepath = filepath
        self.brain = GeminiBrain()
        self.df = pd.read_csv(filepath)
        self.artifacts = {} 
        self.strategy = {}
        self.best_model = None
        self.best_score = -np.inf
        self.best_algo_name = ""
        self.target = None
        self.model_type = None
        self.le = None
        self.feature_metadata = [] 
        self.tournament_results = {}
        
    def log(self, msg):
        add_log(self.model_id, msg)

    # --- STEP 1: INITIAL ANALYSIS ---
    def analyze_data_structure(self):
        self.log("Step 1: Scanning dataset structure...")
        columns = list(self.df.columns)
        head_csv = self.df.head(5).to_csv(index=False)
        
        self.log("Consulting Gemini AI for recommendations...")
        ai_advice = self.brain.analyze(head_csv, str(columns))
        
        suggestion = {
            "all_columns": columns,
            "target": columns[-1],
            "drop_columns": [],
            "type": "regression",
            "reasoning": "Automatic heuristic"
        }

        if ai_advice:
            self.log("AI Recommendations Received.")
            suggestion["target"] = ai_advice.get("target_suggestion", columns[-1])
            suggestion["drop_columns"] = ai_advice.get("drop_suggestions", [])
            suggestion["type"] = ai_advice.get("type", "regression").lower()
            suggestion["reasoning"] = ai_advice.get("reasoning", "")
        else:
            self.log("AI unavailable. Using defaults.")
            
        entry = next((item for item in projects_db if item["id"] == self.model_id), None)
        if entry:
            entry["analysis_result"] = suggestion
            entry["status"] = "pending_approval"
            self.log("Waiting for user approval...")
        return suggestion

    # --- STEP 2: EXECUTION ---
    def execute_training(self, user_config):
        self.target = user_config.get("target")
        drops = user_config.get("drop_columns", [])
        
        self.log(f"Target: '{self.target}'")
        if drops:
            self.df = self.df.drop(columns=[c for c in drops if c in self.df.columns])
            
        y = self.df[self.target]
        if y.dtype == 'object' or y.nunique() < 20:
            self.model_type = "classification"
        else:
            self.model_type = "regression"
            
        X, y = self.step_2_cleaning()
        X_train, X_test, y_train, y_test = self.step_3_splitting(X, y)
        self.step_4_training(X_train, X_test, y_train, y_test)
        self.step_5_artifacts()
        
        return {
            "winner": self.best_algo_name,
            "test_accuracy": self.best_score,
            "tournament_results": self.tournament_results
        }

    def step_2_cleaning(self):
        orig_path = os.path.join(ARTIFACTS_DIR, f"{self.model_id}_original.csv")
        self.df.to_csv(orig_path, index=False)
        self.artifacts["original_data"] = orig_path
        
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]

        # Extract Metadata
        self.feature_metadata = []
        for col in X.columns:
            col_data = X[col]
            meta = {"name": col}
            unique_count = col_data.nunique()
            is_numeric = pd.api.types.is_numeric_dtype(col_data)
            
            if is_numeric and unique_count > 15:
                meta["type"] = "number"
                meta["min"] = float(col_data.min()) if not pd.isna(col_data.min()) else 0
                meta["max"] = float(col_data.max()) if not pd.isna(col_data.max()) else 100
                meta["default"] = float(round(col_data.mean(), 2)) if not pd.isna(col_data.mean()) else 0
            else:
                meta["type"] = "categorical"
                unique_vals = sorted(col_data.astype(str).unique().tolist())
                meta["options"] = unique_vals[:50]
                meta["default"] = unique_vals[0] if unique_vals else ""
            self.feature_metadata.append(meta)
        
        if self.model_type == "classification" and y.dtype == 'object':
            self.le = LabelEncoder()
            y = self.le.fit_transform(y)
            
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'bool']).columns
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), numeric_features),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ]), categorical_features)
            ], verbose_feature_names_out=False)
            
        X_processed = self.preprocessor.fit_transform(X)
        clean_df = pd.DataFrame(X_processed)
        clean_df['TARGET'] = y
        clean_path = os.path.join(ARTIFACTS_DIR, f"{self.model_id}_cleaned.csv")
        clean_df.to_csv(clean_path, index=False)
        self.artifacts["cleaned_data"] = clean_path
        return X, y

    def step_3_splitting(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def step_4_training(self, X_train, X_test, y_train, y_test):
        self.log("Running Model Tournament...")
        models = []
        if self.model_type == "classification":
            models = [
                ("Random Forest", RandomForestClassifier(n_estimators=100)),
                ("Gradient Boosting", GradientBoostingClassifier()),
                ("Logistic Regression", LogisticRegression(max_iter=1000)),
                ("KNN", KNeighborsClassifier())
            ]
            scoring = 'accuracy'
        else:
            models = [
                ("Random Forest Regressor", RandomForestRegressor(n_estimators=100)),
                ("Gradient Boosting Regressor", GradientBoostingRegressor()),
                ("Linear Regression", LinearRegression()),
                ("KNN Regressor", KNeighborsRegressor())
            ]
            scoring = 'r2'
            
        for name, model in models:
            pipeline = Pipeline(steps=[('preprocessor', self.preprocessor), ('model', model)])
            pipeline.fit(X_train, y_train)
            if self.model_type == "classification":
                score = accuracy_score(y_test, pipeline.predict(X_test))
            else:
                score = r2_score(y_test, pipeline.predict(X_test))
            
            self.log(f"--> {name} Score: {round(score, 4)}")
            self.tournament_results[name] = float(score)

            if score > self.best_score:
                self.best_score = score
                self.best_model = pipeline
                self.best_algo_name = name
                
        self.log(f"Winner: {self.best_algo_name}")

    def step_5_artifacts(self):
        model_path = os.path.join(ARTIFACTS_DIR, f"{self.model_id}_model.pkl")
        joblib.dump({"pipeline": self.best_model, "le": self.le}, model_path)
        self.artifacts["model_file"] = model_path

# --- API ENDPOINTS ---
@app.get("/")
def health(): return {"status": "online", "message": "Robust AutoML Backend Running"}

@app.get("/test_ai")
def test_ai_endpoint():
    brain = GeminiBrain()
    success, msg = brain.test_connection()
    return {"status": "connected" if success else "error", "message": msg}

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_with_gemini(request: ChatRequest):
    brain = GeminiBrain()
    response = brain.chat(request.message)
    return {"response": response}

@app.post("/analyze_strategy")
async def analyze_strategy_v2(file: UploadFile = File(...)):
    global latest_project_id
    fid = str(uuid.uuid4())
    filename = file.filename.replace(" ", "_")
    path = os.path.join(UPLOAD_DIR, f"{fid}_{filename}")
    with open(path, "wb") as f: shutil.copyfileobj(file.file, f)
    
    mid = str(uuid.uuid4())
    latest_project_id = mid
    entry = {
        "id": mid, "name": filename, "status": "analyzing", "logs": [], 
        "artifacts": {}, "file_path": path
    }
    projects_db.insert(0, entry)
    
    engineer = MLEngineer(mid, path)
    suggestion = engineer.analyze_data_structure()
    df = pd.read_csv(path)
    return {
        "columns": suggestion["all_columns"],
        "preview": df.head().to_dict(),
        "rows": len(df),
        "ai_strategy": json.dumps({
            "target_suggestion": suggestion["target"],
            "drop_suggestions": suggestion["drop_columns"],
            "reasoning": suggestion["reasoning"]
        })
    }

class TrainRequestV2(BaseModel):
    target: str
    drop_columns: List[str]

@app.post("/train")
async def train_v2(request: TrainRequestV2):
    global latest_project_id
    if not latest_project_id: raise HTTPException(400, "No active session.")
    entry = next((item for item in projects_db if item["id"] == latest_project_id), None)
    engineer = MLEngineer(latest_project_id, entry["file_path"])
    results = engineer.execute_training(request.dict())
    return {
        "status": "success",
        "tournament_results": results["tournament_results"],
        "winner": results["winner"],
        "test_accuracy": results["test_accuracy"]
    }

class PredictionRequest(BaseModel):
    features: dict

@app.post("/predict")
async def predict_v2(request: PredictionRequest):
    global latest_project_id
    if not latest_project_id: raise HTTPException(400, "No active model.")
    path = os.path.join(ARTIFACTS_DIR, f"{latest_project_id}_model.pkl")
    if not os.path.exists(path): raise HTTPException(404, "Model not ready")
    try:
        artifact = joblib.load(path)
        model = artifact["pipeline"]
        le = artifact["le"]
        safe_data = {k: (v if v != "" else np.nan) for k, v in request.features.items()}
        df = pd.DataFrame([safe_data])
        model_cols = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else df.columns
        for col in model_cols:
            if col not in df.columns: df[col] = 0
        pred = model.predict(df)[0]
        if le: pred = le.inverse_transform([int(pred)])[0]
        return {"prediction": str(pred)}
    except Exception as e:
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
