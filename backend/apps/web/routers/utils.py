from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from fastapi import Depends, HTTPException, status
from starlette.responses import StreamingResponse

from pydantic import BaseModel

from utils.misc import calculate_sha256
import requests


import os
import asyncio
import json
from config import OLLAMA_API_BASE_URL


router = APIRouter()


class UploadBlobForm(BaseModel):
    filename: str


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    os.makedirs("./uploads", exist_ok=True)
    file_path = os.path.join("./uploads", file.filename)

    def file_write_stream():
        total = 0
        total_size = file.size
        chunk_size = 1024 * 1024

        done = False
        try:
            with open(file_path, "wb") as f:
                while True:
                    chunk = file.file.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    total += len(chunk)
                    done = total_size == total

                    res = {
                        "total": total_size,
                        "uploaded": total,
                    }

                    yield f"data: {json.dumps(res)}\n\n"

                if done:
                    with open(file_path, "rb") as f:
                        hashed = calculate_sha256(f)

                        f.seek(0)
                        file_data = f.read()

                        url = f"{OLLAMA_API_BASE_URL}/blobs/sha256:{hashed}"

                        response = requests.post(url, data=file_data)

                        if response.ok:
                            res = {
                                "done": done,
                                "blob": f"sha256:{hashed}",
                            }
                            os.remove(file_path)

                            yield f"data: {json.dumps(res)}\n\n"
                        else:
                            raise "Ollama: Could not create blob, Please try again."

        except Exception as e:
            res = {"error": str(e)}
            yield f"data: {json.dumps(res)}\n\n"

    return StreamingResponse(file_write_stream(), media_type="text/event-stream")
