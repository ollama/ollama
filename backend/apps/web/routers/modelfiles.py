from fastapi import Response
from fastapi import Depends, FastAPI, HTTPException, status
from datetime import datetime, timedelta
from typing import List, Union, Optional

from fastapi import APIRouter
from pydantic import BaseModel
import json

from apps.web.models.users import Users
from apps.web.models.modelfiles import (
    Modelfiles,
    ModelfileForm,
    ModelfileTagNameForm,
    ModelfileUpdateForm,
    ModelfileResponse,
)

from utils.utils import (
    bearer_scheme,
)
from constants import ERROR_MESSAGES

router = APIRouter()

############################
# GetModelfiles
############################


@router.get("/", response_model=List[ModelfileResponse])
async def get_modelfiles(skip: int = 0, limit: int = 50, cred=Depends(bearer_scheme)):
    token = cred.credentials
    user = Users.get_user_by_token(token)

    if user:
        return Modelfiles.get_modelfiles(skip, limit)
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.INVALID_TOKEN,
        )


############################
# CreateNewModelfile
############################


@router.post("/create", response_model=Optional[ModelfileResponse])
async def create_new_modelfile(form_data: ModelfileForm, cred=Depends(bearer_scheme)):
    token = cred.credentials
    user = Users.get_user_by_token(token)

    if user:
        # Admin Only
        if user.role == "admin":
            modelfile = Modelfiles.insert_new_modelfile(user.id, form_data)
            return ModelfileResponse(
                **{
                    **modelfile.model_dump(),
                    "modelfile": json.loads(modelfile.modelfile),
                }
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.INVALID_TOKEN,
        )


############################
# GetModelfileByTagName
############################


@router.post("/", response_model=Optional[ModelfileResponse])
async def get_modelfile_by_tag_name(
    form_data: ModelfileTagNameForm, cred=Depends(bearer_scheme)
):
    token = cred.credentials
    user = Users.get_user_by_token(token)

    if user:
        modelfile = Modelfiles.get_modelfile_by_tag_name(form_data.tag_name)

        if modelfile:
            return ModelfileResponse(
                **{
                    **modelfile.model_dump(),
                    "modelfile": json.loads(modelfile.modelfile),
                }
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.INVALID_TOKEN,
        )


############################
# UpdateModelfileByTagName
############################


@router.post("/update", response_model=Optional[ModelfileResponse])
async def update_modelfile_by_tag_name(
    form_data: ModelfileUpdateForm, cred=Depends(bearer_scheme)
):
    token = cred.credentials
    user = Users.get_user_by_token(token)

    if user:
        if user.role == "admin":
            modelfile = Modelfiles.get_modelfile_by_tag_name(form_data.tag_name)
            if modelfile:
                updated_modelfile = {
                    **json.loads(modelfile.modelfile),
                    **form_data.modelfile,
                }

                modelfile = Modelfiles.update_modelfile_by_tag_name(
                    form_data.tag_name, updated_modelfile
                )

                return ModelfileResponse(
                    **{
                        **modelfile.model_dump(),
                        "modelfile": json.loads(modelfile.modelfile),
                    }
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.INVALID_TOKEN,
        )


############################
# DeleteModelfileByTagName
############################


@router.delete("/delete", response_model=bool)
async def delete_modelfile_by_tag_name(
    form_data: ModelfileTagNameForm, cred=Depends(bearer_scheme)
):
    token = cred.credentials
    user = Users.get_user_by_token(token)

    if user:
        if user.role == "admin":
            result = Modelfiles.delete_modelfile_by_tag_name(form_data.tag_name)
            return result
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.INVALID_TOKEN,
        )
