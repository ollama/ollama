from fastapi import Response
from fastapi import Depends, FastAPI, HTTPException, status
from datetime import datetime, timedelta
from typing import List, Union

from fastapi import APIRouter
from pydantic import BaseModel
import time
import uuid

from apps.web.models.auths import (
    SigninForm,
    SignupForm,
    UserResponse,
    SigninResponse,
    Auths,
)
from apps.web.models.users import Users


from utils.utils import (
    get_password_hash,
    bearer_scheme,
    create_token,
)
from utils.misc import get_gravatar_url
from constants import ERROR_MESSAGES


router = APIRouter()

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
            "profile_image_url": user.profile_image_url,
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.INVALID_TOKEN,
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
            "profile_image_url": user.profile_image_url,
        }
    else:
        raise HTTPException(400, detail=ERROR_MESSAGES.INVALID_CRED)


############################
# SignUp
############################


@router.post("/signup", response_model=SigninResponse)
async def signup(form_data: SignupForm):
    if not Users.get_user_by_email(form_data.email.lower()):
        try:
            role = "admin" if Users.get_num_users() == 0 else "pending"
            hashed = get_password_hash(form_data.password)
            user = Auths.insert_new_auth(form_data.email, hashed, form_data.name, role)

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
                    "profile_image_url": user.profile_image_url,
                }
            else:
                raise HTTPException(500, detail=ERROR_MESSAGES.DEFAULT(err))
        except Exception as err:
            raise HTTPException(500, detail=ERROR_MESSAGES.DEFAULT(err))
    else:
        raise HTTPException(400, detail=ERROR_MESSAGES.DEFAULT())
