from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.authentication import AuthenticationMiddleware
from apps.web.routers import auths, users, chats, modelfiles, utils
from config import WEBUI_VERSION, WEBUI_AUTH
from apps.web.middlewares.auth import BearerTokenAuthBackend, on_auth_error

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

app.add_middleware(AuthenticationMiddleware, backend=BearerTokenAuthBackend(), on_error=on_auth_error)

app.include_router(users.router, prefix="/users", tags=["users"])
app.include_router(chats.router, prefix="/chats", tags=["chats"])
app.include_router(modelfiles.router, prefix="/modelfiles", tags=["modelfiles"])
app.include_router(utils.router, prefix="/utils", tags=["utils"])


@app.get("/")
async def get_status():
    return {"status": True, "version": WEBUI_VERSION, "auth": WEBUI_AUTH}
