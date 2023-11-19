from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from apps.web.routers import auths
from config import OLLAMA_WEBUI_VERSION, OLLAMA_WEBUI_AUTH

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(auths.router, prefix="/auths", tags=["auths"])


@app.get("/")
async def get_status():
    return {"status": True, "version": OLLAMA_WEBUI_VERSION, "auth": OLLAMA_WEBUI_AUTH}
