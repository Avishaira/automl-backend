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

# ייבוא ספריות ה-AI וה-ML
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

# --- הגדרות ---
UPLOAD_DIR = "uploads"
ARTIFACTS_DIR = "artifacts"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# הגדרת לוגים
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AutoML_Brain")

app = FastAPI(title="Gemini Expert AutoML")

# אפשור גישה מכל דומיין (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- בסיס נתונים בזיכרון ---
# מבנה: { id: str, status: str, logs: list[str], artifacts: dict, ... }
projects_db = []

def add_log(model_id: str, message: str):
    """פונקציית עזר לכתיבת לוגים לפרויקט ספציפי"""
    entry = next((item for item in projects_db if item["id"] == model_id), None)
    if entry:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        entry["logs"].append(log_entry)
        logger.info(f"Project {model_id}: {message}")

# --- המוח של GEMINI ---
class GeminiBrain:
    def __init__(self):
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash') # מודל מהיר וחכם
            self.active = True
        else:
            self.active = False

    def analyze(self, df_head_csv):
        """שולח את דגימת הנתונים לניתוח אסטרטגי"""
        if not self.active: return None
        
        prompt = f"""
        Act as a Senior Data Scientist. Analyze this dataset sample (CSV):
        {df_head_csv}
        
        Decide:
        1. Target Column (prediction label).
        2. Problem Type (Classification/Regression).
        3. Split Strategy (e.g., 80/20, 70/30).
        4. Cleaning Strategy (Imputation methods).
        
        Return JSON ONLY:
        {{
            "target": "col_name",
            "type": "classification" or "regression",
            "split_ratio": 0.2,
            "cleaning_notes": "string explanation",
            "recommended_models": ["RandomForest", "GradientBoosting"]
        }}
        """
        try:
            res = self.model.generate_content(prompt)
            clean_text = res.text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)
        except:
            return None

# --- מהנדס ה-ML האוטומטי ---
class MLEngineer:
    def __init__(self, model_id, filepath):
        self.model_id = model_id
        self.filepath = filepath
        self.brain = GeminiBrain()
        self.df = pd.read_csv(filepath)
        self.artifacts = {} # נתיבים לקבצים שנשמרו
        self.strategy = {}
        self.best_model = None
        self.best_score = -np.inf
        self.best_algo_name = ""
        self.target = None
        self.model_type = None
        self.le = None # Label Encoder
        
    def log(self, msg):
        add_log(self.model_id, msg)

    def step_1_analysis(self):
        self.log("Step 1: Reading dataset and scanning variables...")
        self.log(f"Dataset shape: {self.df.shape[0]} rows, {self.df.shape[1]} columns.")
        
        head_csv = self.df.head(5).to_csv(index=False)
        self.log("Sending data sample to Gemini AI for strategic analysis...")
        
        ai_advice = self.brain.analyze(head_csv)
        
        if ai_advice:
            self.log("AI Analysis Complete.")
            self.log(f"Decided Target Variable: '{ai_advice.get('target')}'")
            self.log(f"Problem Type Identified: {ai_advice.get('type').upper()}")
            self.strategy = ai_advice
            self.target = ai_advice.get('target')
            self.model_type = ai_advice.get('type').lower()
        else:
            self.log("AI unavailable (Key missing?), using heuristic backup.")
            self.target = self.df.columns[-1]
            self.model_type = "classification" if self.df[self.target].nunique() < 20 else "regression"
            self.strategy = {"split_ratio": 0.2, "cleaning_notes": "Standard Auto-Imputation"}
            
        # ולידציה שהעמודה קיימת
        if self.target not in self.df.columns:
            self.target = self.df.columns[-1]
            self.log(f"Correction: Target set to last column '{self.target}'")

    def step_2_cleaning(self):
        self.log("Step 2: Performing Data Cleaning & Engineering...")
        self.log(f"AI Cleaning Strategy: {self.strategy.get('cleaning_notes', 'Auto')}")
        
        # שמירת הקובץ המקורי
        orig_path = os.path.join(ARTIFACTS_DIR, f"{self.model_id}_original.csv")
        self.df.to_csv(orig_path, index=False)
        self.artifacts["original_data"] = orig_path
        
        # הפרדת משתנים
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]
        
        # קידוד משתנה המטרה אם הוא טקסט (עבור סיווג)
        if self.model_type == "classification" and y.dtype == 'object':
            self.log("Encoding target variable labels to numbers...")
            self.le = LabelEncoder()
            y = self.le.fit_transform(y)
            
        # הגדרת הצינור (Pipeline) לעיבוד נתונים
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'bool']).columns
        
        self.log(f"Detected {len(numeric_features)} numeric and {len(categorical_features)} categorical features.")
        
        # יצירת טרנספורמרים
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')), # השלמת חסרים במספרים
                    ('scaler', StandardScaler()) # נרמול
                ]), numeric_features),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')), # השלמת חסרים בטקסט
                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # המרת טקסט לבינארי
                ]), categorical_features)
            ], verbose_feature_names_out=False)
            
        # ביצוע הניקוי בפועל
        self.log("Applying transformations...")
        X_processed = self.preprocessor.fit_transform(X)
        
        # שמירת הקובץ הנקי (משוחזר לדאטה-פריים)
        clean_df = pd.DataFrame(X_processed)
        # הוספת עמודת המטרה חזרה כדי שהקובץ יהיה שלם
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
        
        # שמירת קבצי אימון ובדיקה
        train_df = pd.concat([X_train, pd.Series(y_train, name=self.target, index=X_train.index)], axis=1)
        test_df = pd.concat([X_test, pd.Series(y_test, name=self.target, index=X_test.index)], axis=1)
        
        train_path = os.path.join(ARTIFACTS_DIR, f"{self.model_id}_train.csv")
        test_path = os.path.join(ARTIFACTS_DIR, f"{self.model_id}_test.csv")
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        self.artifacts["train_data"] = train_path
        self.artifacts["test_data"] = test_path
        self.log("Saved 'train.csv' and 'test.csv'.")
        
        return X_train, X_test, y_train, y_test

    def step_4_training(self, X_train, X_test, y_train, y_test):
        self.log("Step 4: Starting Model Tournament...")
        
        models = []
        # בחירת מודלים לפי סוג הבעיה
        if self.model_type == "classification":
            models = [
                ("Random Forest", RandomForestClassifier(n_estimators=100)),
                ("Gradient Boosting", GradientBoostingClassifier()),
                ("Logistic Regression", LogisticRegression(max_iter=1000)),
                ("K-Nearest Neighbors", KNeighborsClassifier())
            ]
        else:
            models = [
                ("Random Forest Regressor", RandomForestRegressor(n_estimators=100)),
                ("Gradient Boosting Regressor", GradientBoostingRegressor()),
                ("Linear Regression", LinearRegression()),
                ("KNN Regressor", KNeighborsRegressor())
            ]
            
        for name, model in models:
            self.log(f"Training {name} with default hyperparameters...")
            pipeline = Pipeline(steps=[('preprocessor', self.preprocessor), ('model', model)])
            
            pipeline.fit(X_train, y_train)
            
            if self.model_type == "classification":
                score = accuracy_score(y_test, pipeline.predict(X_test))
                metric_name = "Accuracy"
            else:
                score = r2_score(y_test, pipeline.predict(X_test))
                metric_name = "R2 Score"
                
            self.log(f"--> {name} Result: {metric_name} = {round(score, 4)}")
            
            if score > self.best_score:
                self.best_score = score
                self.best_model = pipeline
                self.best_algo_name = name
                
        self.log(f"Step 5: Tournament Winner Identified: {self.best_algo_name}")
        self.log(f"Final Accuracy/Score: {round(self.best_score * 100, 2)}%")

    def step_5_artifacts(self):
        self.log("Step 6: Generating deployment artifacts...")
        
        # שמירת המודל הבינארי (.pkl)
        model_path = os.path.join(ARTIFACTS_DIR, f"{self.model_id}_model.pkl")
        joblib.dump({"pipeline": self.best_model, "le": self.le}, model_path)
        self.artifacts["model_file"] = model_path
        
        # יצירת קוד Python מלא לשחזור
        code = f"""
# Reproducible Machine Learning Script
# Generated by AI Master Engineer
# ----------------------------------
# Problem Type: {self.model_type}
# Target Variable: {self.target}
# Best Model: {self.best_algo_name}
# Score: {round(self.best_score, 4)}

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, r2_score

# 1. Load Data
# Ensure you have 'dataset.csv' in the same folder
print("Loading dataset...")
df = pd.read_csv('dataset.csv')

X = df.drop(columns=['{self.target}'])
y = df['{self.target}']

# 2. Preprocessing
print("Preprocessing data...")
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

# 3. Model Definition ({self.best_algo_name})
model = {self.best_model.named_steps['model']}

pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

# 4. Train
print("Training model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={self.strategy.get('split_ratio', 0.2)}, random_state=42)
pipeline.fit(X_train, y_train)

# 5. Evaluate
print("Evaluating...")
score = pipeline.score(X_test, y_test)
print(f"Final Score: {{score}}")
"""
        code_path = os.path.join(ARTIFACTS_DIR, f"{self.model_id}_code.py")
        with open(code_path, "w") as f:
            f.write(code)
        self.artifacts["code_file"] = code_path
        self.log("Full Python code generated successfully.")

# --- ניהול הריצה ---
def run_pipeline(model_id: str, file_path: str):
    engineer = MLEngineer(model_id, file_path)
    
    # איתור הפרויקט בזיכרון
    entry = next((item for item in projects_db if item["id"] == model_id), None)
    
    try:
        # הרצת השלבים
        engineer.step_1_analysis()
        X, y = engineer.step_2_cleaning()
        X_train, X_test, y_train, y_test = engineer.step_3_splitting(X, y)
        engineer.step_4_training(X_train, X_test, y_train, y_test)
        engineer.step_5_artifacts()
        
        # עדכון סיום
        if entry:
            entry["status"] = "completed"
            entry["accuracy"] = f"{round(engineer.best_score * 100, 2)}%"
            entry["artifacts"] = engineer.artifacts # שמירת הנתיבים להורדה
            
    except Exception as e:
        engineer.log(f"CRITICAL ERROR: {str(e)}")
        logger.error(traceback.format_exc())
        if entry:
            entry["status"] = "failed"
            entry["error"] = str(e)

# --- API ENDPOINTS ---

@app.get("/")
def health(): return {"status": "online"}

@app.get("/models")
def get_models(): return {"models": projects_db}

# Endpoint שמחזיר לוגים בזמן אמת ל-Frontend
@app.get("/models/{model_id}/logs")
def get_logs(model_id: str):
    entry = next((item for item in projects_db if item["id"] == model_id), None)
    if not entry: raise HTTPException(404, "Model not found")
    return {"logs": entry["logs"], "status": entry["status"], "artifacts": entry.get("artifacts", {})}

# Endpoint להורדת קבצים
@app.get("/download/{model_id}/{file_type}")
def download_file(model_id: str, file_type: str):
    entry = next((item for item in projects_db if item["id"] == model_id), None)
    if not entry or "artifacts" not in entry:
        raise HTTPException(404, "Artifacts not ready")
        
    path = entry["artifacts"].get(file_type) # original_data, cleaned_data, train_data, test_data, code_file, model_file
    if not path or not os.path.exists(path):
        raise HTTPException(404, "File not found")
        
    return FileResponse(path, filename=os.path.basename(path))

@app.post("/upload")
async def upload(bg_tasks: BackgroundTasks, file: UploadFile = File(...)):
    fid = str(uuid.uuid4())
    filename = file.filename.replace(" ", "_")
    path = os.path.join(UPLOAD_DIR, f"{fid}_{filename}")
    
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
        
    mid = str(uuid.uuid4())
    entry = {
        "id": mid,
        "name": filename,
        "status": "training",
        "accuracy": None,
        "logs": [],
        "created_at": datetime.datetime.now().strftime("%H:%M"),
        "artifacts": {}
    }
    projects_db.insert(0, entry)
    
    # התחלת האימון ברקע
    bg_tasks.add_task(run_pipeline, mid, path)
    return entry

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
