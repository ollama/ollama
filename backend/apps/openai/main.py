from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

import requests
import json
from pydantic import BaseModel

from apps.web.models.users import Users
from constants import ERROR_MESSAGES
from utils.utils import decode_token, get_current_user
from config import OPENAI_API_BASE_URL, OPENAI_API_KEY

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.OPENAI_API_BASE_URL = OPENAI_API_BASE_URL
app.state.OPENAI_API_KEY = OPENAI_API_KEY


class UrlUpdateForm(BaseModel):
    url: str


class KeyUpdateForm(BaseModel):
    key: str


@app.get("/url")
async def get_openai_url(user=Depends(get_current_user)):
    if user and user.role == "admin":
        return {"OPENAI_API_BASE_URL": app.state.OPENAI_API_BASE_URL}
    else:
        raise HTTPException(status_code=401, detail=ERROR_MESSAGES.ACCESS_PROHIBITED)


@app.post("/url/update")
async def update_openai_url(form_data: UrlUpdateForm, user=Depends(get_current_user)):
    if user and user.role == "admin":
        app.state.OPENAI_API_BASE_URL = form_data.url
        return {"OPENAI_API_BASE_URL": app.state.OPENAI_API_BASE_URL}
    else:
        raise HTTPException(status_code=401, detail=ERROR_MESSAGES.ACCESS_PROHIBITED)


@app.get("/key")
async def get_openai_key(user=Depends(get_current_user)):
    if user and user.role == "admin":
        return {"OPENAI_API_KEY": app.state.OPENAI_API_KEY}
    else:
        raise HTTPException(status_code=401, detail=ERROR_MESSAGES.ACCESS_PROHIBITED)


@app.post("/key/update")
async def update_openai_key(form_data: KeyUpdateForm, user=Depends(get_current_user)):
    if user and user.role == "admin":
        app.state.OPENAI_API_KEY = form_data.key
        return {"OPENAI_API_KEY": app.state.OPENAI_API_KEY}
    else:
        raise HTTPException(status_code=401, detail=ERROR_MESSAGES.ACCESS_PROHIBITED)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy(path: str, request: Request, user=Depends(get_current_user)):
    target_url = f"{app.state.OPENAI_API_BASE_URL}/{path}"
    print(target_url, app.state.OPENAI_API_KEY)

    if user.role not in ["user", "admin"]:
        raise HTTPException(status_code=401, detail=ERROR_MESSAGES.ACCESS_PROHIBITED)
    if app.state.OPENAI_API_KEY == "":
        raise HTTPException(status_code=401, detail=ERROR_MESSAGES.API_KEY_NOT_FOUND)

    body = await request.body()

    # TODO: Remove below after gpt-4-vision fix from Open AI
    # Try to decode the body of the request from bytes to a UTF-8 string (Require add max_token to fix gpt-4-vision)
    try:
        body = body.decode("utf-8")
        body = json.loads(body)

        # Check if the model is "gpt-4-vision-preview" and set "max_tokens" to 4000
        # This is a workaround until OpenAI fixes the issue with this model
        if body.get("model") == "gpt-4-vision-preview":
            if "max_tokens" not in body:
                body["max_tokens"] = 4000
            print("Modified body_dict:", body)

        # Convert the modified body back to JSON
        body = json.dumps(body)
    except json.JSONDecodeError as e:
        print("Error loading request body into a dictionary:", e)

    headers = {}
    headers["Authorization"] = f"Bearer {app.state.OPENAI_API_KEY}"
    headers["Content-Type"] = "application/json"

    try:
        r = requests.request(
            method=request.method,
            url=target_url,
            data=body,
            headers=headers,
            stream=True,
        )

        r.raise_for_status()

        # Check if response is SSE
        if "text/event-stream" in r.headers.get("Content-Type", ""):
            return StreamingResponse(
                r.iter_content(chunk_size=8192),
                status_code=r.status_code,
                headers=dict(r.headers),
            )
        else:
            # For non-SSE, read the response and return it
            # response_data = (
            #     r.json()
            #     if r.headers.get("Content-Type", "")
            #     == "application/json"
            #     else r.text
            # )

            response_data = r.json()

            print(type(response_data))

            if "openai" in app.state.OPENAI_API_BASE_URL and path == "models":
                response_data["data"] = list(
                    filter(lambda model: "gpt" in model["id"], response_data["data"])
                )

            return response_data
    except Exception as e:
        print(e)
        error_detail = "Ollama WebUI: Server Connection Error"
        if r is not None:
            try:
                res = r.json()
                if "error" in res:
                    error_detail = f"External: {res['error']}"
            except:
                error_detail = f"External: {e}"

        raise HTTPException(status_code=r.status_code, detail=error_detail)
