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

# --- CORS ---
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
        Act as a Senior Data Scientist. Analyze this dataset sample (CSV):
        {df_head_csv}
        
        Decide:
        1. Target Column (prediction label).
        2. Problem Type (Classification/Regression).
        3. Split Strategy (e.g., 80/20, 70/30).
        4. Cleaning Strategy.
        
        Return JSON ONLY:
        {{
            "target": "col_name",
            "type": "classification" or "regression",
            "split_ratio": 0.2,
            "cleaning_notes": "string explanation"
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
        self.feature_metadata = [] # NEW: Store form structure here
        
    def log(self, msg):
        add_log(self.model_id, msg)

    def step_1_analysis(self):
        self.log("Step 1: Reading dataset and scanning variables...")
        head_csv = self.df.head(5).to_csv(index=False)
        self.log("Sending data to AI for analysis...")
        
        ai_advice = self.brain.analyze(head_csv)
        
        if ai_advice:
            self.log("AI Analysis Complete.")
            self.strategy = ai_advice
            self.target = ai_advice.get('target')
            self.model_type = ai_advice.get('type').lower()
        else:
            self.log("AI unavailable, using heuristic backup.")
            self.target = self.df.columns[-1]
            self.model_type = "classification" if self.df[self.target].nunique() < 20 else "regression"
            self.strategy = {"split_ratio": 0.2}
            
        if self.target not in self.df.columns:
            self.target = self.df.columns[-1]
            self.log(f"Correction: Target set to '{self.target}'")

    def step_2_cleaning(self):
        self.log("Step 2: Performing Data Cleaning & Feature Extraction...")
        
        # Save Original
        orig_path = os.path.join(ARTIFACTS_DIR, f"{self.model_id}_original.csv")
        self.df.to_csv(orig_path, index=False)
        self.artifacts["original_data"] = orig_path
        
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]

        # --- NEW: Extract Metadata for UI Form ---
        self.log("Analyzing independent variables for UI generation...")
        for col in X.columns:
            col_data = X[col]
            meta = {"name": col}
            
            # Check if numeric
            if pd.api.types.is_numeric_dtype(col_data):
                meta["type"] = "number"
                # Use clean float values for JSON serialization
                meta["min"] = float(col_data.min()) if not pd.isna(col_data.min()) else 0
                meta["max"] = float(col_data.max()) if not pd.isna(col_data.max()) else 100
                mean_val = float(col_data.mean()) if not pd.isna(col_data.mean()) else 0
                meta["default"] = round(mean_val, 2)
            else:
                meta["type"] = "categorical"
                # Get unique values (limit to 50 to avoid massive lists)
                unique_vals = col_data.astype(str).unique().tolist()
                meta["options"] = unique_vals[:50]
                meta["default"] = unique_vals[0] if unique_vals else ""
            
            self.feature_metadata.append(meta)
        
        # --- Continue Cleaning ---
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
        
        # Save Cleaned Data
        clean_df = pd.DataFrame(X_processed)
        clean_df['TARGET'] = y
        clean_path = os.path.join(ARTIFACTS_DIR, f"{self.model_id}_cleaned.csv")
        clean_df.to_csv(clean_path, index=False)
        self.artifacts["cleaned_data"] = clean_path
        self.log("Saved 'cleaned_dataset.csv'.")
        
        return X, y

    def step_3_splitting(self, X, y):
        ratio = self.strategy.get("split_ratio", 0.2)
        self.log(f"Step 3: Splitting Dataset. Train: {int((1-ratio)*100)}% | Test: {int(ratio*100)}%")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=42)
        
        # Save Splits
        train_path = os.path.join(ARTIFACTS_DIR, f"{self.model_id}_train.csv")
        test_path = os.path.join(ARTIFACTS_DIR, f"{self.model_id}_test.csv")
        
        pd.concat([X_train, pd.Series(y_train, name='target', index=X_train.index)], axis=1).to_csv(train_path, index=False)
        pd.concat([X_test, pd.Series(y_test, name='target', index=X_test.index)], axis=1).to_csv(test_path, index=False)
        
        self.artifacts["train_data"] = train_path
        self.artifacts["test_data"] = test_path
        
        return X_train, X_test, y_train, y_test

    def step_4_training(self, X_train, X_test, y_train, y_test):
        self.log("Step 4: Starting Model Tournament...")
        
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
                metric_name = "Accuracy"
            else:
                score = r2_score(y_test, pipeline.predict(X_test))
                metric_name = "R2 Score"
                
            self.log(f"--> {name}: {metric_name} = {round(score, 4)}")
            
            if score > self.best_score:
                self.best_score = score
                self.best_model = pipeline
                self.best_algo_name = name
                
        self.log(f"Step 5: Best Model Selected: {self.best_algo_name} ({round(self.best_score * 100, 2)}%)")

    def step_5_artifacts(self):
        self.log("Step 6: Generating deployment artifacts...")
        
        model_path = os.path.join(ARTIFACTS_DIR, f"{self.model_id}_model.pkl")
        joblib.dump({"pipeline": self.best_model, "le": self.le}, model_path)
        self.artifacts["model_file"] = model_path
        
        code = f"""
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
# ... imports ...

# Model: {self.best_algo_name}
# Target: {self.target}

# 1. Load Data
df = pd.read_csv('dataset.csv')
X = df.drop(columns=['{self.target}'])
y = df['{self.target}']

# 2. Load Pipeline
model_data = joblib.load('{self.model_id}_model.pkl')
pipeline = model_data['pipeline']

# 3. Predict
# preds = pipeline.predict(X)
print("Model loaded.")
"""
        code_path = os.path.join(ARTIFACTS_DIR, f"{self.model_id}_code.py")
        with open(code_path, "w") as f: f.write(code)
        self.artifacts["code_file"] = code_path
        self.log("Code generated.")

def run_pipeline(model_id: str, file_path: str):
    engineer = MLEngineer(model_id, file_path)
    entry = next((item for item in projects_db if item["id"] == model_id), None)
    
    try:
        engineer.step_1_analysis()
        X, y = engineer.step_2_cleaning()
        X_train, X_test, y_train, y_test = engineer.step_3_splitting(X, y)
        engineer.step_4_training(X_train, X_test, y_train, y_test)
        engineer.step_5_artifacts()
        
        if entry:
            entry["status"] = "completed"
            entry["accuracy"] = f"{round(engineer.best_score * 100, 2)}%"
            entry["artifacts"] = engineer.artifacts
            entry["target_col"] = engineer.target
            # IMPORTANT: Save the extracted metadata so Frontend can build the form
            entry["feature_metadata"] = engineer.feature_metadata
            
    except Exception as e:
        engineer.log(f"ERROR: {str(e)}")
        logger.error(traceback.format_exc())
        if entry:
            entry["status"] = "failed"
            entry["error"] = str(e)

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
        "artifacts": entry.get("artifacts", {}),
        "feature_metadata": entry.get("feature_metadata", []) # Send metadata to frontend
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

@app.post("/upload")
async def upload(bg_tasks: BackgroundTasks, file: UploadFile = File(...)):
    fid = str(uuid.uuid4())
    filename = file.filename.replace(" ", "_")
    path = os.path.join(UPLOAD_DIR, f"{fid}_{filename}")
    with open(path, "wb") as f: shutil.copyfileobj(file.file, f)
        
    mid = str(uuid.uuid4())
    entry = {
        "id": mid, "name": filename, "status": "training", "accuracy": None, 
        "logs": [], "created_at": datetime.datetime.now().strftime("%H:%M"), 
        "artifacts": {}, "feature_metadata": []
    }
    projects_db.insert(0, entry)
    bg_tasks.add_task(run_pipeline, mid, path)
    return entry

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
