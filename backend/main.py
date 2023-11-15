import time
import sys

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

from fastapi import HTTPException
from starlette.exceptions import HTTPException as StarletteHTTPException

from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.middleware.cors import CORSMiddleware

from apps.ollama.main import app as ollama_app


class SPAStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except (HTTPException, StarletteHTTPException) as ex:
            if ex.status_code == 404:
                return await super().get_response("index.html", scope)
            else:
                raise ex


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def check_url(request: Request, call_next):
    start_time = int(time.time())
    response = await call_next(request)
    process_time = int(time.time()) - start_time
    response.headers["X-Process-Time"] = str(process_time)

    return response


app.mount("/ollama/api", WSGIMiddleware(ollama_app))
app.mount("/", SPAStaticFiles(directory="../build", html=True), name="spa-static-files")
