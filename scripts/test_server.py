#!/usr/bin/env python3
"""
Minimal test server for Ollama API
"""
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import json

app = FastAPI(title="Ollama Elite AI Platform", version="1.0.0")

@app.get("/health")
async def health():
    """Health check endpoint"""
    return JSONResponse({
        "status": "healthy",
        "service": "ollama-api",
        "version": "1.0.0"
    })

@app.get("/api/v1/health")
async def api_health():
    """API health check"""
    return JSONResponse({
        "status": "operational",
        "timestamp": "2026-01-13T19:00:00Z",
        "version": "1.0.0"
    })

@app.get("/")
async def root():
    """Root endpoint"""
    return JSONResponse({
        "name": "Ollama Elite AI Platform",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "api": "/api/v1",
            "docs": "/docs"
        }
    })

@app.post("/api/v1/generate")
async def generate(prompt: str = "Hello"):
    """Test generation endpoint"""
    return JSONResponse({
        "prompt": prompt,
        "response": "This is a test response from the Ollama Elite AI Platform",
        "model": "test-model",
        "tokens": 42
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
