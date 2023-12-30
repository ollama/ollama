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
    UpdatePasswordForm,
    UserResponse,
    SigninResponse,
    Auths,
)
from apps.web.models.users import Users


from utils.utils import (
    get_password_hash,
    get_current_user,
    create_token,
)
from utils.misc import get_gravatar_url
from constants import ERROR_MESSAGES


router = APIRouter()

############################
# GetSessionUser
############################


@router.get("/", response_model=UserResponse)
async def get_session_user(user=Depends(get_current_user)):
    return {
        "id": user.id,
        "email": user.email,
        "name": user.name,
        "role": user.role,
        "profile_image_url": user.profile_image_url,
    }


############################
# Update Password
############################


@router.post("/update/password", response_model=bool)
async def update_password(form_data: UpdatePasswordForm, cred=Depends(bearer_scheme)):
    token = cred.credentials
    session_user = Users.get_user_by_token(token)

    if session_user:
        user = Auths.authenticate_user(session_user.email, form_data.password)

        if user:
            hashed = get_password_hash(form_data.new_password)
            return Auths.update_user_password_by_id(user.id, hashed)
        else:
            raise HTTPException(400, detail=ERROR_MESSAGES.INVALID_PASSWORD)
    else:
        raise HTTPException(400, detail=ERROR_MESSAGES.INVALID_CRED)


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
            user = Auths.insert_new_auth(
                form_data.email.lower(), hashed, form_data.name, role
            )

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
                raise HTTPException(500, detail=ERROR_MESSAGES.CREATE_USER_ERROR)
        except Exception as err:
            raise HTTPException(500, detail=ERROR_MESSAGES.DEFAULT(err))
    else:
        raise HTTPException(400, detail=ERROR_MESSAGES.EMAIL_TAKEN)
