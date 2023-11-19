from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from apps.web.routers import auths, users
from config import WEBUI_VERSION, WEBUI_AUTH

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
app.include_router(users.router, prefix="/users", tags=["users"])


@app.get("/")
async def get_status():
    return {"status": True, "version": WEBUI_VERSION, "auth": WEBUI_AUTH}
