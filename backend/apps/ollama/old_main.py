from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import requests
import json
from pydantic import BaseModel

from apps.web.models.users import Users
from constants import ERROR_MESSAGES
from utils.utils import decode_token, get_current_user
from config import OLLAMA_API_BASE_URL, WEBUI_AUTH

import aiohttp

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


# async def fetch_sse(method, target_url, body, headers):
#     async with aiohttp.ClientSession() as session:
#         try:
#             async with session.request(
#                 method, target_url, data=body, headers=headers
#             ) as response:
#                 print(response.status)
#                 async for line in response.content:
#                     yield line
#         except Exception as e:
#             print(e)
#             error_detail = "Ollama WebUI: Server Connection Error"
#             yield json.dumps({"error": error_detail, "message": str(e)}).encode()


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy(path: str, request: Request, user=Depends(get_current_user)):
    target_url = f"{app.state.OLLAMA_API_BASE_URL}/{path}"
    print(target_url)

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

    session = aiohttp.ClientSession()
    response = None
    try:
        response = await session.request(
            request.method, target_url, data=body, headers=headers
        )

        print(response)
        if not response.ok:
            data = await response.json()
            print(data)
            response.raise_for_status()

        async def generate():
            async for line in response.content:
                print(line)
                yield line
            await session.close()

        return StreamingResponse(generate(), response.status)

    except Exception as e:
        print(e)
        error_detail = "Ollama WebUI: Server Connection Error"

        if response is not None:
            try:
                res = await response.json()
                if "error" in res:
                    error_detail = f"Ollama: {res['error']}"
            except:
                error_detail = f"Ollama: {e}"

        await session.close()
        raise HTTPException(
            status_code=response.status if response else 500,
            detail=error_detail,
        )
