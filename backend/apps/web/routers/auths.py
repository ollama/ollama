from fastapi import Response
from fastapi import Depends, FastAPI, HTTPException, status
from datetime import datetime, timedelta
from typing import List, Union

from fastapi import APIRouter
from pydantic import BaseModel
import time
import uuid

from constants import ERROR_MESSAGES
from utils import (
    get_password_hash,
    bearer_scheme,
    create_token,
)

from apps.web.models.auths import (
    SigninForm,
    SignupForm,
    UserResponse,
    SigninResponse,
    Auths,
)
from apps.web.models.users import Users
import config

router = APIRouter()

DB = config.DB


############################
# GetSessionUser
############################


@router.get("/", response_model=UserResponse)
async def get_session_user(cred=Depends(bearer_scheme)):
    token = cred.credentials
    user = Users.get_user_by_token(token)
    if user:
        return {
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "role": user.role,
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
        )


############################
# SignIn
############################


@router.post("/signin", response_model=SigninResponse)
async def signin(form_data: SigninForm):
    user = Auths.authenticate_user(form_data.email.lower(), form_data.password)
    if user:
        token = create_token(data={"email": user.email})

        return {
            "token": token,
            "token_type": "Bearer",
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "role": user.role,
        }
    else:
        raise HTTPException(400, detail=ERROR_MESSAGES.DEFAULT())


############################
# SignUp
############################


@router.post("/signup", response_model=SigninResponse)
async def signup(form_data: SignupForm):
    if not Users.get_user_by_email(form_data.email.lower()):
        try:
            hashed = get_password_hash(form_data.password)
            user = Auths.insert_new_auth(form_data.email, hashed, form_data.name)

            if user:
                token = create_token(data={"email": user.email})
                # response.set_cookie(key='token', value=token, httponly=True)

                return {
                    "token": token,
                    "token_type": "Bearer",
                    "id": user.id,
                    "email": user.email,
                    "name": user.name,
                    "role": user.role,
                }
            else:
                raise HTTPException(500, detail=ERROR_MESSAGES.DEFAULT(err))
        except Exception as err:
            raise HTTPException(500, detail=ERROR_MESSAGES.DEFAULT(err))
    else:
        raise HTTPException(400, detail=ERROR_MESSAGES.DEFAULT())
