"""
FastAPI app with static file serving for frontend
"""
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from nlp_pipeline import analyze_transcript
import os

app = FastAPI(title="Physician Notetaker API")

class TranscriptRequest(BaseModel):
    transcript: str

@app.get("/")
async def root():
    """Serve the frontend HTML"""
    return FileResponse("static/index.html")

@app.post("/analyze")
async def analyze(req: TranscriptRequest):
    """Analyze medical transcript"""
    try:
        if not req.transcript.strip():
            raise HTTPException(status_code=400, detail="Transcript cannot be empty")
        
        result = analyze_transcript(req.transcript)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

# Mount static files directory
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
