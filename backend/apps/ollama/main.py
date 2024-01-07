from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.concurrency import run_in_threadpool

import requests
import json
from pydantic import BaseModel

from apps.web.models.users import Users
from constants import ERROR_MESSAGES
from utils.utils import decode_token, get_current_user
from config import OLLAMA_API_BASE_URL, WEBUI_AUTH

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.OLLAMA_API_BASE_URL = OLLAMA_API_BASE_URL

# TARGET_SERVER_URL = OLLAMA_API_BASE_URL


@app.get("/url")
async def get_ollama_api_url(user=Depends(get_current_user)):
    if user and user.role == "admin":
        return {"OLLAMA_API_BASE_URL": app.state.OLLAMA_API_BASE_URL}
    else:
        raise HTTPException(status_code=401, detail=ERROR_MESSAGES.ACCESS_PROHIBITED)


class UrlUpdateForm(BaseModel):
    url: str


@app.post("/url/update")
async def update_ollama_api_url(
    form_data: UrlUpdateForm, user=Depends(get_current_user)
):
    if user and user.role == "admin":
        app.state.OLLAMA_API_BASE_URL = form_data.url
        return {"OLLAMA_API_BASE_URL": app.state.OLLAMA_API_BASE_URL}
    else:
        raise HTTPException(status_code=401, detail=ERROR_MESSAGES.ACCESS_PROHIBITED)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy(path: str, request: Request, user=Depends(get_current_user)):
    target_url = f"{app.state.OLLAMA_API_BASE_URL}/{path}"

    body = await request.body()
    headers = dict(request.headers)

    if user.role in ["user", "admin"]:
        if path in ["pull", "delete", "push", "copy", "create"]:
            if user.role != "admin":
                raise HTTPException(
                    status_code=401, detail=ERROR_MESSAGES.ACCESS_PROHIBITED
                )
    else:
        raise HTTPException(status_code=401, detail=ERROR_MESSAGES.ACCESS_PROHIBITED)

    headers.pop("Host", None)
    headers.pop("Authorization", None)
    headers.pop("Origin", None)
    headers.pop("Referer", None)

    r = None

    def get_request():
        nonlocal r
        try:
            r = requests.request(
                method=request.method,
                url=target_url,
                data=body,
                headers=headers,
                stream=True,
            )

            r.raise_for_status()

            return StreamingResponse(
                r.iter_content(chunk_size=8192),
                status_code=r.status_code,
                headers=dict(r.headers),
            )
        except Exception as e:
            raise e

    try:
        return await run_in_threadpool(get_request)
    except Exception as e:
        error_detail = "Ollama WebUI: Server Connection Error"
        if r is not None:
            try:
                res = r.json()
                if "error" in res:
                    error_detail = f"Ollama: {res['error']}"
            except:
                error_detail = f"Ollama: {e}"

        raise HTTPException(
            status_code=r.status_code if r else 500,
            detail=error_detail,
        )
