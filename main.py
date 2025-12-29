import os
import shutil
import pandas as pd
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware  # <--- 1. MUST IMPORT THIS
from pydantic import BaseModel
import uuid
import logging
import datetime
# import ai_engine  # We assume this file exists next to main.py

# If you don't have ai_engine.py yet, use this mock class so the code doesn't crash
class MockAIEngine:
    def get_best_model_code(self, df_head, target, domain):
        return "# Mock generated code\nprint('Model Trained')"
try:
    import ai_engine
except ImportError:
    ai_engine = MockAIEngine()


# --- CONFIGURATION ---
UPLOAD_DIR = "uploads"
MODELS_DIR = "models"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AutoML_Backend")

app = FastAPI(title="AutoML Pocket Backend")

# --- 2. CORS CONFIGURATION (CRITICAL FIX) ---
# Without this, your React app CANNOT talk to this server
origins = ["*"]  # Allow all domains

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- IN-MEMORY DATABASE (For Prototype) ---
# Stores the list of models so the Frontend can fetch them
projects_db = []

# --- CORE LOGIC ---

def train_worker(model_id: str, file_path: str, target: str = "auto", domain: str = "tabular"):
    """
    Executes training in background and updates the in-memory DB.
    """
    logger.info(f"Starting training for Model {model_id}...")
    
    # Find the model in DB to update status
    model_entry = next((item for item in projects_db if item["id"] == model_id), None)
    
    try:
        # 1. Load data
        df = pd.read_csv(file_path)
        
        # 2. Simulate AI work
        # In a real app, 'target' would be selected by user. Here we default/guess.
        if target == "auto":
            target = df.columns[-1] # Simple guess: last column is target
            
        code = ai_engine.get_best_model_code(df.head(), target, domain)
        
        # 3. Save code
        code_path = os.path.join(MODELS_DIR, f"{model_id}.py")
        with open(code_path, "w") as f:
            f.write(code)
            
        # 4. Update DB on success
        if model_entry:
            model_entry["status"] = "completed"
            model_entry["accuracy"] = "92.4%" # Mock accuracy
            model_entry["progress"] = 100
            
        logger.info(f"Model {model_id} training complete.")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if model_entry:
            model_entry["status"] = "failed"
            model_entry["error"] = str(e)

# --- API ENDPOINTS ---

@app.get("/")
def health_check():
    return {"status": "running", "service": "AutoML Backend"}

# 3. ADDED: GET /models endpoint
# This is what the React app polls to check connection and get list
@app.get("/models")
def get_models():
    return {"models": projects_db}

@app.post("/upload")
async def upload_dataset(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Modified to match React App: Uploads AND starts 'auto' training immediately.
    """
    try:
        # 1. Save File
        file_id = str(uuid.uuid4())
        filename = file.filename
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{filename}")
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 2. Create Model Entry (so UI shows it immediately)
        model_id = str(uuid.uuid4())
        new_model = {
            "id": model_id,
            "name": filename.split('.')[0],
            "type": "Auto-Detected",
            "status": "training",
            "accuracy": None,
            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "file_id": file_id
        }
        projects_db.insert(0, new_model) # Add to start of list
        
        # 3. Start Background Training Immediately
        # We assume 'tabular' and auto-target for this "One-Click" flow
        background_tasks.add_task(train_worker, model_id, file_path, "auto", "tabular")
        
        # 4. Return the model object (React app expects this)
        return new_model

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail="Upload failed")

if __name__ == "__main__":
    import uvicorn
    # Use the PORT environment variable provided by Render, default to 8000
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
