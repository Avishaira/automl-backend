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
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

import google.generativeai as genai
from sklearn.model_selection import train_test_split
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

app = FastAPI(title="Gemini Expert AutoML")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- IN-MEMORY DATABASE ---
projects_db = []

def add_log(model_id: str, message: str):
    entry = next((item for item in projects_db if item["id"] == model_id), None)
    if entry:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        entry["logs"].append(log_entry)
        logger.info(f"Project {model_id}: {message}")

# --- GEMINI BRAIN ---
class GeminiBrain:
    def __init__(self):
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            self.active = True
        else:
            self.active = False

    def analyze(self, df_head_csv):
        if not self.active: return None
        
        prompt = f"""
        Act as a Data Scientist. Analyze this CSV sample:
        {df_head_csv}
        
        Recommend:
        1. Target Column (prediction label).
        2. Columns to DROP (ID, Name, Date, Unstructured text).
        3. Problem Type (Classification/Regression).
        
        Return JSON ONLY:
        {{
            "target": "col_name",
            "drop_columns": ["col_1", "col_2"],
            "type": "classification" or "regression",
            "reasoning": "explanation"
        }}
        """
        try:
            res = self.model.generate_content(prompt)
            clean_text = res.text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)
        except:
            return None

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
        
    def log(self, msg):
        add_log(self.model_id, msg)

    # --- STEP 1: INITIAL ANALYSIS (PRE-APPROVAL) ---
    def analyze_data_structure(self):
        self.log("Step 1: Scanning dataset structure...")
        
        columns = list(self.df.columns)
        head_csv = self.df.head(5).to_csv(index=False)
        
        self.log("Consulting Gemini AI for recommendations...")
        ai_advice = self.brain.analyze(head_csv)
        
        suggestion = {
            "all_columns": columns,
            "target": columns[-1], # Default
            "drop_columns": [],
            "type": "regression",
            "reasoning": "Automatic heuristic"
        }

        if ai_advice:
            self.log("AI Recommendations Received.")
            suggestion["target"] = ai_advice.get("target", columns[-1])
            suggestion["drop_columns"] = ai_advice.get("drop_columns", [])
            suggestion["type"] = ai_advice.get("type", "regression").lower()
            suggestion["reasoning"] = ai_advice.get("reasoning", "")
        else:
            self.log("AI unavailable. Waiting for user input.")
            
        # Update DB with suggestions so Frontend can show them
        entry = next((item for item in projects_db if item["id"] == self.model_id), None)
        if entry:
            entry["analysis_result"] = suggestion
            entry["status"] = "pending_approval"
            self.log("Waiting for user approval on variables...")

    # --- STEP 2: EXECUTION (POST-APPROVAL) ---
    def execute_training(self, user_config):
        self.target = user_config.get("target")
        drops = user_config.get("drop_columns", [])
        
        self.log(f"User Confirmed Target: '{self.target}'")
        if drops:
            self.log(f"Dropping columns: {drops}")
            self.df = self.df.drop(columns=[c for c in drops if c in self.df.columns])
            
        # Re-eval type based on final target
        y = self.df[self.target]
        if y.dtype == 'object' or y.nunique() < 20:
            self.model_type = "classification"
        else:
            self.model_type = "regression"
            
        self.log(f"Task Type: {self.model_type.upper()}")
        
        # Chain remaining steps
        X, y = self.step_2_cleaning()
        X_train, X_test, y_train, y_test = self.step_3_splitting(X, y)
        self.step_4_training(X_train, X_test, y_train, y_test)
        self.step_5_artifacts()

    def step_2_cleaning(self):
        self.log("Step 2: Data Cleaning & Feature Engineering...")
        
        # Save Original (Cleaned of drops)
        orig_path = os.path.join(ARTIFACTS_DIR, f"{self.model_id}_original.csv")
        self.df.to_csv(orig_path, index=False)
        self.artifacts["original_data"] = orig_path
        
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]

        # Extract Metadata for UI Form
        self.feature_metadata = []
        for col in X.columns:
            col_data = X[col]
            meta = {"name": col}
            if pd.api.types.is_numeric_dtype(col_data):
                meta["type"] = "number"
                meta["min"] = float(col_data.min()) if not pd.isna(col_data.min()) else 0
                meta["max"] = float(col_data.max()) if not pd.isna(col_data.max()) else 100
                meta["default"] = float(round(col_data.mean(), 2)) if not pd.isna(col_data.mean()) else 0
            else:
                meta["type"] = "categorical"
                unique_vals = col_data.astype(str).unique().tolist()
                meta["options"] = unique_vals[:50]
                meta["default"] = unique_vals[0] if unique_vals else ""
            self.feature_metadata.append(meta)
        
        # Cleaning Logic
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
        self.log("Step 3: Splitting Train/Test Sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        train_path = os.path.join(ARTIFACTS_DIR, f"{self.model_id}_train.csv")
        test_path = os.path.join(ARTIFACTS_DIR, f"{self.model_id}_test.csv")
        
        # Save artifacts
        try:
            pd.concat([X_train, pd.Series(y_train, name='target', index=X_train.index)], axis=1).to_csv(train_path, index=False)
            pd.concat([X_test, pd.Series(y_test, name='target', index=X_test.index)], axis=1).to_csv(test_path, index=False)
            self.artifacts["train_data"] = train_path
            self.artifacts["test_data"] = test_path
        except: pass
        
        return X_train, X_test, y_train, y_test

    def step_4_training(self, X_train, X_test, y_train, y_test):
        self.log("Step 4: Running Model Tournament...")
        
        models = []
        if self.model_type == "classification":
            models = [
                ("Random Forest", RandomForestClassifier(n_estimators=100)),
                ("Gradient Boosting", GradientBoostingClassifier()),
                ("Logistic Regression", LogisticRegression(max_iter=1000)),
                ("KNN", KNeighborsClassifier())
            ]
        else:
            models = [
                ("Random Forest", RandomForestRegressor(n_estimators=100)),
                ("Gradient Boosting", GradientBoostingRegressor()),
                ("Linear Regression", LinearRegression()),
                ("KNN", KNeighborsRegressor())
            ]
            
        for name, model in models:
            self.log(f"Training {name}...")
            pipeline = Pipeline(steps=[('preprocessor', self.preprocessor), ('model', model)])
            pipeline.fit(X_train, y_train)
            
            if self.model_type == "classification":
                score = accuracy_score(y_test, pipeline.predict(X_test))
            else:
                score = r2_score(y_test, pipeline.predict(X_test))
                
            self.log(f"--> {name} Score: {round(score, 4)}")
            
            if score > self.best_score:
                self.best_score = score
                self.best_model = pipeline
                self.best_algo_name = name
                
        self.log(f"Step 5: Best Model Selected: {self.best_algo_name}")

    def step_5_artifacts(self):
        self.log("Step 6: Saving Model & Code...")
        
        model_path = os.path.join(ARTIFACTS_DIR, f"{self.model_id}_model.pkl")
        joblib.dump({"pipeline": self.best_model, "le": self.le}, model_path)
        self.artifacts["model_file"] = model_path
        
        code = f"""
import pandas as pd
import joblib
# Model: {self.best_algo_name}
# Target: {self.target}
"""
        code_path = os.path.join(ARTIFACTS_DIR, f"{self.model_id}_code.py")
        with open(code_path, "w") as f: f.write(code)
        self.artifacts["code_file"] = code_path

# --- TASKS ---
def analyze_task(model_id: str, file_path: str):
    engineer = MLEngineer(model_id, file_path)
    try:
        engineer.analyze_data_structure()
    except Exception as e:
        add_log(model_id, f"Analysis Error: {e}")
        # Mark as failed in DB
        entry = next((item for item in projects_db if item["id"] == model_id), None)
        if entry: entry["status"] = "failed"

def training_task(model_id: str, file_path: str, config: dict):
    engineer = MLEngineer(model_id, file_path)
    entry = next((item for item in projects_db if item["id"] == model_id), None)
    
    try:
        engineer.execute_training(config)
        if entry:
            entry["status"] = "completed"
            entry["accuracy"] = f"{round(engineer.best_score * 100, 2)}%"
            entry["artifacts"] = engineer.artifacts
            entry["feature_metadata"] = engineer.feature_metadata
            entry["target_col"] = engineer.target
    except Exception as e:
        engineer.log(f"Training Error: {e}")
        logger.error(traceback.format_exc())
        if entry: entry["status"] = "failed"

# --- API ENDPOINTS ---

@app.get("/")
def health(): return {"status": "online"}

@app.get("/models")
def get_models(): return {"models": projects_db}

@app.get("/models/{model_id}/logs")
def get_logs(model_id: str):
    entry = next((item for item in projects_db if item["id"] == model_id), None)
    if not entry: raise HTTPException(404, "Model not found")
    return {
        "logs": entry["logs"], 
        "status": entry["status"],
        "analysis_result": entry.get("analysis_result"), # NEW: Send suggestions
        "artifacts": entry.get("artifacts", {}),
        "feature_metadata": entry.get("feature_metadata", [])
    }

@app.get("/download/{model_id}/{file_type}")
def download_file(model_id: str, file_type: str):
    entry = next((item for item in projects_db if item["id"] == model_id), None)
    if not entry: raise HTTPException(404, "Not found")
    path = entry["artifacts"].get(file_type)
    if not path or not os.path.exists(path): raise HTTPException(404, "File not found")
    return FileResponse(path, filename=os.path.basename(path))

class PredictReq(BaseModel):
    data: Dict[str, Any]

@app.post("/predict/{model_id}")
def predict(model_id: str, req: PredictReq):
    path = os.path.join(ARTIFACTS_DIR, f"{model_id}_model.pkl")
    if not os.path.exists(path): raise HTTPException(404, "Model not ready")
    try:
        artifact = joblib.load(path)
        model = artifact["pipeline"]
        le = artifact["le"]
        df = pd.DataFrame([req.data])
        pred = model.predict(df)[0]
        if le: pred = le.inverse_transform([int(pred)])[0]
        return {"prediction": str(pred), "model_id": model_id}
    except Exception as e:
        raise HTTPException(500, str(e))

# --- STEP 1: UPLOAD ---
@app.post("/upload")
async def upload(bg_tasks: BackgroundTasks, file: UploadFile = File(...)):
    fid = str(uuid.uuid4())
    filename = file.filename.replace(" ", "_")
    path = os.path.join(UPLOAD_DIR, f"{fid}_{filename}")
    with open(path, "wb") as f: shutil.copyfileobj(file.file, f)
        
    mid = str(uuid.uuid4())
    entry = {
        "id": mid, "name": filename, "status": "analyzing", # Initial status
        "accuracy": None, "logs": [], "created_at": datetime.datetime.now().strftime("%H:%M"), 
        "artifacts": {}, "file_path": path # Save path for later
    }
    projects_db.insert(0, entry)
    bg_tasks.add_task(analyze_task, mid, path)
    return entry

# --- STEP 2: START TRAINING ---
class TrainConfig(BaseModel):
    target: str
    drop_columns: List[str]

@app.post("/models/{model_id}/train")
def start_training(model_id: str, config: TrainConfig, bg_tasks: BackgroundTasks):
    entry = next((item for item in projects_db if item["id"] == model_id), None)
    if not entry: raise HTTPException(404, "Model not found")
    
    entry["status"] = "training"
    bg_tasks.add_task(training_task, model_id, entry["file_path"], config.dict())
    return {"status": "started"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
