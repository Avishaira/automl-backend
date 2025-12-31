import os
import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types

# Initialize Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AutoML_Brain")

app = FastAPI()

# Pydantic model for the incoming chat request
class ChatRequest(BaseModel):
    message: str

def get_gemini_client():
    """
    Initializes the Google GenAI client.
    Ensures the API key is present in environment variables.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not found in environment variables.")
        raise HTTPException(status_code=500, detail="Server configuration error: API Key missing.")
    
    # Initialize the client from the new SDK
    return genai.Client(api_key=api_key)

@app.get("/")
async def root():
    return {"status": "alive", "message": "AutoML Backend is running"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    logger.info(f"Received chat request: {request.message}")
    
    try:
        client = get_gemini_client()
        
        # Using the new SDK structure
        # 'gemini-2.5-flash' is the model identified in your logs
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=request.message,
            config=types.GenerateContentConfig(
                temperature=0.7,
            )
        )
        
        # Extract text from the new response object structure
        if response.text:
            return {"response": response.text}
        else:
            return {"response": "No response generated."}

    except Exception as e:
        logger.error(f"Error generating content: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Matches the port binding in your deployment logs
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
