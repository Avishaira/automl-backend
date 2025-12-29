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
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# --- GENAI & ML IMPORTS ---
import google.generativeai as genai
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, r2_score

# --- CONFIGURATION ---
UPLOAD_DIR = "uploads"
MODELS_DIR = "models"
ARTIFACTS_DIR = "artifacts"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AutoML_Brain")

app = FastAPI(title="Gemini-Powered AutoML")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DB ---
projects_db = []

# --- GEMINI INTEGRATION ---
class GeminiBrain:
    def __init__(self):
        # Tries to get key from environment variable
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            self.active = True
        else:
            logger.warning("GOOGLE_API_KEY not found. Falling back to heuristic mode.")
            self.active = False

    def analyze_dataset(self, df_head_csv: str):
        """
        Sends the data preview to Gemini to decide strategy.
        """
        if not self.active:
            return None

        prompt = f"""
        You are a World-Class Machine Learning Engineer.
        Analyze this dataset sample (CSV format):
        
        {df_head_csv}
        
        Task:
        1. Identify the most likely Target Column (label) for prediction.
        2. Determine if this is a "Classification" or "Regression" problem.
        3. Suggest the best model architecture (e.g. Random Forest, Gradient Boosting).
        4. List key feature engineering steps needed.

        Return ONLY a raw JSON object (no markdown, no quotes around keys) with this structure:
        {{
            "target_column": "string",
            "problem_type": "classification" or "regression",
            "suggested_model": "string",
            "reasoning": "string"
        }}
        """
        try:
            response = self.model.generate_content(prompt)
            # Cleanup json format if Gemini adds markdown blocks
            text = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}")
            return None

# --- AUTOML SYSTEM ---
class AutoMLSystem:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = pd.read_csv(filepath)
        self.target_col = None
        self.model_type = None
        self.brain = GeminiBrain()
        
        self.pipeline = None
        self.label_encoder = None
        self.best_score = 0
        self.insights = {}

    def plan_and_execute(self):
        # 1. Ask Gemini for the Strategy
        analysis = self.brain.analyze_dataset(self.df.head(10).to_csv(index=False))
        
        if analysis:
            logger.info(f"Gemini Insights: {analysis}")
            self.target_col = analysis.get("target_column")
            self.model_type = analysis.get("problem_type").lower()
            self.insights = analysis
        else:
            # Fallback Heuristics (if Gemini fails or no key)
            logger.info("Using Heuristics fallback")
            potential = [c for c in self.df.columns if c.lower() in ['target', 'class', 'label', 'y', 'price', 'churn', 'survived']]
            self.target_col = potential[0] if potential else self.df.columns[-1]
            
            y = self.df[self.target_col]
            if y.dtype == 'object' or y.nunique() < 20:
                self.model_type = "classification"
            else:
                self.model_type = "regression"

        # Validate columns exist
        if self.target_col not in self.df.columns:
            # Fallback if Gemini hallucinated a column name
            self.target_col = self.df.columns[-1]

        # 2. Prepare Data
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]

        # Handle Categorical Target
        if self.model_type == "classification" and y.dtype == 'object':
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)

        # 3. Build Pipeline (Smart Preprocessing)
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'bool']).columns

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), numeric_features),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]), categorical_features)
            ])

        # 4. Train Models (The "Competition")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        models_to_try = []
        if self.model_type == "classification":
            models_to_try = [
                ("RandomForest", RandomForestClassifier(n_estimators=100)),
                ("GradientBoosting", GradientBoostingClassifier()),
                ("LogisticRegression", LogisticRegression(max_iter=1000))
            ]
        else:
            models_to_try = [
                ("RandomForest", RandomForestRegressor(n_estimators=100)),
                ("GradientBoosting", GradientBoostingRegressor()),
                ("LinearRegression", LinearRegression())
            ]

        best_model = None
        best_name = ""
        best_score = -np.inf

        for name, model_inst in models_to_try:
            pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', model_inst)])
            pipe.fit(X_train, y_train)
            
            score = 0
            if self.model_type == "classification":
                score = accuracy_score(y_test, pipe.predict(X_test))
            else:
                score = r2_score(y_test, pipe.predict(X_test))
            
            if score > best_score:
                best_score = score
                best_model = pipe
                best_name = name

        self.pipeline = best_model
        self.best_score = best_score
        
        return {
            "target": self.target_col,
            "type": self.model_type,
            "score": round(best_score * 100, 2),
            "algorithm": best_name,
            "reasoning": self.insights.get("reasoning", "Best statistical fit selected.")
        }

    def save_artifacts(self, model_id):
        # Save Pickle
        joblib.dump({
            "pipeline": self.pipeline,
            "label_encoder": self.label_encoder
        }, os.path.join(ARTIFACTS_DIR, f"{model_id}.pkl"))

        # Save Python Code (Generated dynamically)
        code = f"""
# Machine Learning Pipeline generated by AutoML
# Target Column: {self.target_col}
# Problem Type: {self.model_type}
# Insights: {self.insights.get('reasoning', 'N/A')}

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

# 1. Load Data
df = pd.read_csv('dataset.csv')
target = '{self.target_col}'

X = df.drop(columns=[target])
y = df[target]

# 2. Load Trained Pipeline
# The pipeline handles scaling, encoding and imputation automatically
model = joblib.load('{model_id}.pkl')

# 3. Predict
print("Running predictions...")
# predictions = model.predict(X)
"""
        with open(os.path.join(MODELS_DIR, f"{model_id}.py"), "w") as f:
            f.write(code)

# --- WORKER ---
def train_worker_real(model_id: str, file_path: str):
    logger.info(f"Engineer working on {model_id}")
    model_entry = next((item for item in projects_db if item["id"] == model_id), None)
    
    try:
        automl = AutoMLSystem(file_path)
        result = automl.plan_and_execute()
        automl.save_artifacts(model_id)
        
        if model_entry:
            model_entry["status"] = "completed"
            model_entry["accuracy"] = f"{result['score']}%"
            model_entry["target_col"] = result['target']
            model_entry["code_url"] = f"/models/{model_id}/code"
            model_entry["description"] = result['reasoning']
            
    except Exception as e:
        logger.error(f"Error: {traceback.format_exc()}")
        if model_entry:
            model_entry["status"] = "failed"
            model_entry["error"] = str(e)

# --- API ---
@app.get("/")
def health_check():
    return {"status": "running", "service": "Gemini AutoML"}

@app.get("/models")
def get_models():
    return {"models": projects_db}

@app.get("/models/{model_id}/code")
def get_code(model_id: str):
    return FileResponse(os.path.join(MODELS_DIR, f"{model_id}.py"))

@app.get("/files/{filename}")
def get_file(filename: str):
    return FileResponse(os.path.join(UPLOAD_DIR, filename))

class PredictReq(BaseModel):
    data: Dict[str, Any]

@app.post("/predict/{model_id}")
def predict(model_id: str, req: PredictReq):
    path = os.path.join(ARTIFACTS_DIR, f"{model_id}.pkl")
    if not os.path.exists(path):
        raise HTTPException(404, "Model not ready")
    
    try:
        artifact = joblib.load(path)
        model = artifact["pipeline"]
        le = artifact["label_encoder"]
        
        df = pd.DataFrame([req.data])
        pred = model.predict(df)[0]
        
        if le:
            pred = le.inverse_transform([int(pred)])[0]
            
        return {"prediction": str(pred), "model_id": model_id}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/upload")
async def upload(bg_tasks: BackgroundTasks, file: UploadFile = File(...)):
    fid = str(uuid.uuid4())
    clean = file.filename.replace(" ", "_")
    path = os.path.join(UPLOAD_DIR, clean)
    
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
        
    mid = str(uuid.uuid4())
    entry = {
        "id": mid,
        "name": clean,
        "status": "training",
        "accuracy": None,
        "created_at": datetime.datetime.now().strftime("%H:%M"),
        "file_url": f"/files/{clean}"
    }
    projects_db.insert(0, entry)
    
    bg_tasks.add_task(train_worker_real, mid, path)
    return entry

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
