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
from typing import List, Optional, Dict, Any

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

# Suppress Deprecation Warnings from Google GenAI
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")

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

# Global reference for the "Demo" mode (compatibility with V2 Frontend)
latest_project_id = None

def add_log(model_id: str, message: str):
    entry = next((item for item in projects_db if item["id"] == model_id), None)
    if entry:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        entry["logs"].append(log_entry)
        logger.info(f"Project {model_id}: {message}")

# --- GEMINI BRAIN (Ultra-Robust Version) ---
class GeminiBrain:
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.active = False
        self.working_model_name = None

        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.active = True
            except Exception as e:
                logger.error(f"Failed to configure Gemini: {e}")
        else:
            logger.warning("Gemini API Key missing! AI features will be disabled.")

    def _generate_content_robust(self, prompt):
        """
        Tries multiple model names if 404s occur.
        """
        if not self.active:
            raise Exception("AI is not active.")

        # If we have a known working model, try only that first
        if self.working_model_name:
            try:
                model = genai.GenerativeModel(self.working_model_name)
                return model.generate_content(prompt)
            except Exception as e:
                logger.warning(f"Previously working model {self.working_model_name} failed: {e}")
                self.working_model_name = None # Reset and try list
        
        # Extended Candidate list - trying specific versions often helps with 404s
        candidates = [
            'gemini-1.5-flash',
            'gemini-1.5-flash-latest',
            'gemini-1.5-flash-001',
            'gemini-1.5-pro',
            'gemini-1.5-pro-latest',
            'gemini-1.5-pro-001',
            'gemini-pro',
            'gemini-1.0-pro'
        ]

        last_error = None
        for model_name in candidates:
            try:
                logger.info(f"Attempting to use model: {model_name}")
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                
                # If we get here, it worked
                self.working_model_name = model_name
                logger.info(f"SUCCESS: Connected to model: {model_name}")
                return response
            except Exception as e:
                error_str = str(e).lower()
                # Only continue if it's a "Not Found" or "404" error
                if "404" in error_str or "not found" in error_str:
                    logger.warning(f"Model {model_name} failed (404). Trying next...")
                    last_error = e
                    continue
                else:
                    # If it's another error (like auth or quota), raise immediately
                    raise e
        
        # If we exhausted the list
        raise Exception(f"All model candidates failed. Last error: {str(last_error)}")

    def chat(self, message: str):
        if not self.active: return "AI is unavailable (API Key missing)."
        try:
            response = self._generate_content_robust(message)
            return response.text
        except Exception as e:
            return f"Error communicating with Gemini: {str(e)}"

    def analyze(self, df_head_csv, columns_list):
        if not self.active: return None
        
        prompt = f"""
        Act as a Data Scientist. Analyze this dataset structure.
        Columns: {columns_list}
        Sample Data:
        {df_head_csv}
        
        Recommend:
        1. Target Column (prediction label).
        2. Columns to DROP (ID, Name, Date, Unstructured text, Leakage).
        3. Problem Type (Classification/Regression).
        
        Return JSON ONLY:
        {{
            "target_suggestion": "col_name",
            "drop_suggestions": ["col_1", "col_2"],
            "type": "classification",
            "reasoning": "explanation"
        }}
        """
        try:
            res = self._generate_content_robust(prompt)
            clean_text = res.text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)
        except Exception as e:
            logger.error(f"Gemini Analysis Error: {e}")
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
            self.log("AI unavailable or failed. Waiting for user input.")
            
        entry = next((item for item in projects_db if item["id"] == self.model_id), None)
        if entry:
            entry["analysis_result"] = suggestion
            entry["status"] = "pending_approval"
            self.log("Waiting for user approval on variables...")
        
        return suggestion

    # --- STEP 2: EXECUTION ---
    def execute_training(self, user_config):
        self.target = user_config.get("target")
        drops = user_config.get("drop_columns", [])
        
        self.log(f"User Confirmed Target: '{self.target}'")
        if drops:
            self.log(f"Dropping columns based on user selection: {len(drops)} columns")
            self.df = self.df.drop(columns=[c for c in drops if c in self.df.columns])
            
        y = self.df[self.target]
        if y.dtype == 'object' or y.nunique() < 20:
            self.model_type = "classification"
        else:
            self.model_type = "regression"
            
        self.log(f"Task Type: {self.model_type.upper()}")
        
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
        self.log("Step 2: Data Cleaning & Feature Engineering...")
        
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
        
        # Encoding Target
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
        ratio = self.strategy.get("split_ratio", 0.2)
        self.log(f"Step 3: Splitting Data ({int((1-ratio)*100)}% Train / {int(ratio*100)}% Test)...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=42)
        
        train_path = os.path.join(ARTIFACTS_DIR, f"{self.model_id}_train.csv")
        test_path = os.path.join(ARTIFACTS_DIR, f"{self.model_id}_test.csv")
        
        try:
            pd.concat([X_train, pd.Series(y_train, name='target', index=X_train.index)], axis=1).to_csv(train_path, index=False)
            pd.concat([X_test, pd.Series(y_test, name='target', index=X_test.index)], axis=1).to_csv(test_path, index=False)
            self.artifacts["train_data"] = train_path
            self.artifacts["test_data"] = test_path
        except: pass
        
        return X_train, X_test, y_train, y_test

    def step_4_training(self, X_train, X_test, y_train, y_test):
        self.log("Step 4: Running Professional Model Tournament (Cross-Validation)...")
        
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
            
            # Use Cross Validation for robust scoring
            if len(X_train) > 50:
                self.log(f"Validating {name} (5-Fold CV)...")
                try:
                    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring=scoring)
                    score = cv_scores.mean()
                    self.log(f"--> {name} Mean CV Score: {round(score, 4)}")
                except:
                    pipeline.fit(X_train, y_train)
                    score = pipeline.score(X_test, y_test)
                    self.log(f"--> {name} Test Score: {round(score, 4)}")
            else:
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
                
        self.log(f"Step 5: Winner is {self.best_algo_name}. Retraining on full set...")
        self.best_model.fit(X_train, y_train)
        
        final_test_score = 0
        if self.model_type == "classification":
            final_test_score = accuracy_score(y_test, self.best_model.predict(X_test))
        else:
            final_test_score = r2_score(y_test, self.best_model.predict(X_test))
            
        self.log(f"Final Test Set Score: {round(final_test_score * 100, 2)}%")

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
def health(): return {"status": "online", "message": "Robust AutoML Backend Running"}

# NEW: Chat Endpoint
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_with_gemini(request: ChatRequest):
    brain = GeminiBrain()
    response = brain.chat(request.message)
    return {"response": response}

# NEW: V2 Frontend Compatibility Layer (Wrappers)
@app.post("/analyze_strategy")
async def analyze_strategy_v2(file: UploadFile = File(...)):
    """Wrapper to make V2 Frontend work with V1 Robust Backend"""
    global latest_project_id
    
    # 1. Save file like V1 does
    fid = str(uuid.uuid4())
    filename = file.filename.replace(" ", "_")
    path = os.path.join(UPLOAD_DIR, f"{fid}_{filename}")
    with open(path, "wb") as f: shutil.copyfileobj(file.file, f)
    
    # 2. Create Project Entry
    mid = str(uuid.uuid4())
    latest_project_id = mid # Save for subsequent calls
    entry = {
        "id": mid, "name": filename, "status": "analyzing",
        "accuracy": None, "logs": [], "created_at": datetime.datetime.now().strftime("%H:%M"), 
        "artifacts": {}, "feature_metadata": [],
        "file_path": path
    }
    projects_db.insert(0, entry)
    
    # 3. Run Analysis Synchronously for V2 Frontend
    engineer = MLEngineer(mid, path)
    suggestion = engineer.analyze_data_structure()
    
    # 4. Return V2 Format
    df = pd.read_csv(path)
    preview = df.head().to_dict()
    
    return {
        "columns": suggestion["all_columns"],
        "preview": preview,
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
    """Wrapper for V2 Frontend Training"""
    global latest_project_id
    if not latest_project_id:
        raise HTTPException(400, "No active session. Upload file first.")
    
    entry = next((item for item in projects_db if item["id"] == latest_project_id), None)
    if not entry: raise HTTPException(404, "Session expired")
    
    engineer = MLEngineer(latest_project_id, entry["file_path"])
    
    # Run training synchronously for V2
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
    """Wrapper for V2 Frontend Prediction"""
    global latest_project_id
    if not latest_project_id: raise HTTPException(400, "No active model.")
    
    path = os.path.join(ARTIFACTS_DIR, f"{latest_project_id}_model.pkl")
    if not os.path.exists(path): raise HTTPException(404, "Model not ready")
    
    try:
        artifact = joblib.load(path)
        model = artifact["pipeline"]
        le = artifact["le"]
        
        # Safe dataframe creation
        safe_data = {k: (v if v != "" else np.nan) for k, v in request.features.items()}
        df = pd.DataFrame([safe_data])
        
        # Fill missing cols with default to prevent crash
        model_cols = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else df.columns
        for col in model_cols:
            if col not in df.columns:
                df[col] = 0
        
        pred = model.predict(df)[0]
        if le: 
            pred = le.inverse_transform([int(pred)])[0]
            
        return {"prediction": str(pred)}
    except Exception as e:
        raise HTTPException(500, str(e))

# --- ORIGINAL V1 ENDPOINTS (For backward compatibility) ---
@app.get("/models")
def get_models(): return {"models": projects_db}

@app.get("/models/{model_id}/logs")
def get_logs(model_id: str):
    entry = next((item for item in projects_db if item["id"] == model_id), None)
    if not entry: raise HTTPException(404, "Model not found")
    return {
        "logs": entry["logs"], 
        "status": entry["status"],
        "analysis_result": entry.get("analysis_result"),
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

@app.post("/upload")
async def upload(bg_tasks: BackgroundTasks, file: UploadFile = File(...)):
    fid = str(uuid.uuid4())
    filename = file.filename.replace(" ", "_")
    path = os.path.join(UPLOAD_DIR, f"{fid}_{filename}")
    with open(path, "wb") as f: shutil.copyfileobj(file.file, f)
        
    mid = str(uuid.uuid4())
    entry = {
        "id": mid, "name": filename, "status": "analyzing",
        "accuracy": None, "logs": [], "created_at": datetime.datetime.now().strftime("%H:%M"), 
        "artifacts": {}, "feature_metadata": [],
        "file_path": path
    }
    projects_db.insert(0, entry)
    bg_tasks.add_task(analyze_task, mid, path)
    return entry

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
