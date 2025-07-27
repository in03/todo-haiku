#!/usr/bin/env python3
"""
Big Phoney API - FastAPI microservice for syllable counting
"""

import warnings
import os

# Suppress warnings for cleaner logs
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import big_phoney
import uvicorn

# Initialize the big-phoney library
phoney = big_phoney.BigPhoney()

app = FastAPI(
    title="Big Phoney API",
    description="A microservice for syllable counting using the big-phoney ML library",
    version="1.0.0"
)

class SyllableRequest(BaseModel):
    text: str

class SyllableResponse(BaseModel):
    syllables: int

class DetailedSyllableResponse(BaseModel):
    text: str
    syllables: int
    words: List[Dict[str, Any]]

class HaikuResponse(BaseModel):
    text: str
    lines: List[Dict[str, Any]]

@app.get("/")
async def root():
    """Root endpoint with basic status."""
    return {"message": "Big Phoney API is running", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "service": "big-phoney-api",
        "version": "1.0.0"
    }

@app.post("/syllables/simple", response_model=SyllableResponse)
async def count_syllables_simple(request: SyllableRequest):
    """Count syllables in text (simple endpoint)."""
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Count syllables using big-phoney
        syllables = phoney.count_syllables(request.text)
        return SyllableResponse(syllables=syllables)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error counting syllables: {str(e)}")

@app.post("/syllables", response_model=DetailedSyllableResponse)
async def count_syllables_detailed(request: SyllableRequest):
    """Count syllables in text with detailed word breakdown."""
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Split text into words
        words = request.text.split()
        
        # Count syllables for each word
        word_data = []
        total_syllables = 0
        
        for word in words:
            word_syllables = phoney.count_syllables(word)
            word_data.append({
                "word": word,
                "syllables": word_syllables
            })
            total_syllables += word_syllables
        
        return DetailedSyllableResponse(
            text=request.text,
            syllables=total_syllables,
            words=word_data
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error counting syllables: {str(e)}")

@app.post("/syllables/haiku", response_model=HaikuResponse)
async def count_syllables_haiku(request: SyllableRequest):
    """Count syllables in haiku text with line-by-line breakdown."""
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Split text into lines
        lines = [line.strip() for line in request.text.split('\n') if line.strip()]
        
        # Count syllables for each line
        line_data = []
        for line in lines:
            # Split line into words and count each word individually (for better dictionary lookup)
            words = line.split()
            line_syllables = sum(phoney.count_syllables(word) for word in words)
            line_data.append({
                "line": line,
                "syllables": line_syllables
            })
        
        return HaikuResponse(
            text=request.text,
            lines=line_data
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error counting syllables: {str(e)}")

@app.get("/docs")
async def get_docs():
    """Redirect to API documentation."""
    return {"docs_url": "/docs"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 