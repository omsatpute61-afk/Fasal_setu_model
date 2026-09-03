import os
import secrets
from typing import Any
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.security import APIKeyHeader
from PIL import Image
import io

# We import the existing ONNX engine
from src.engine.decision_engine import DecisionEngine

# --- CONFIGURATION ---
# In production, store this in an environment variable, e.g., os.environ.get("PESTOPIA_API_KEY")
# For now, we define a static fallback key for testing if the env var isn't set.
VALID_API_KEY = os.environ.get("PESTOPIA_API_KEY", "fasal_setu_live_38f82a9b4c") 
API_KEY_NAME = "X-API-Key"

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

app = FastAPI(
    title="Pestopia Vision API",
    description="FastAPI endpoint for dual-vision agricultural diagnostics using ONNX.",
    version="2.0.0"
)

# Initialize engine on startup
engine = DecisionEngine()

# --- SECURITY DEPENDENCY ---
async def get_api_key(api_key_header: str = Depends(api_key_header)) -> str:
    if api_key_header == VALID_API_KEY:
        return api_key_header
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key",
    )

# --- ENDPOINTS ---
@app.get("/")
async def root():
    return {"status": "Pestopia API is running. Send POST requests to /predict with your X-API-Key header."}

@app.post("/predict")
async def predict_image(
    file: UploadFile = File(...), 
    api_key: str = Depends(get_api_key)
) -> dict[str, Any]:
    
    try:
        # Read the image bytes directly into memory
        image_bytes = await file.read()
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="File provided is not a valid image.")
            
        # Run inference using the exact same ONNX pipeline the Streamlit app uses
        result = engine.process_image(image)
        
        return {
            "success": True,
            "filename": file.filename,
            "predictions": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

# --- API KEY GENERATOR HELPER ---
if __name__ == "__main__":
    import uvicorn
    # Utility script block to quickly run the server locally
    print(f"Starting server... (Default API Key: {VALID_API_KEY})")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
