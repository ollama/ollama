from fastapi import Response, Request
from fastapi import Depends, FastAPI, HTTPException, status
from datetime import datetime, timedelta
from typing import List, Union

from fastapi import APIRouter
from pydantic import BaseModel
import time
import uuid

from apps.web.models.users import Users

from utils.utils import get_password_hash, get_current_user, create_token
from utils.misc import get_gravatar_url, validate_email_format
from constants import ERROR_MESSAGES

router = APIRouter()


class SetDefaultModelsForm(BaseModel):
    models: str


class PromptSuggestion(BaseModel):
    title: List[str]
    content: str


class SetDefaultSuggestionsForm(BaseModel):
    suggestions: List[PromptSuggestion]


############################
# SetDefaultModels
############################


@router.post("/default/models", response_model=str)
async def set_global_default_models(
    request: Request, form_data: SetDefaultModelsForm, user=Depends(get_current_user)
):
    if user.role == "admin":
        request.app.state.DEFAULT_MODELS = form_data.models
        return request.app.state.DEFAULT_MODELS
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )


@router.post("/default/suggestions", response_model=List[PromptSuggestion])
async def set_global_default_suggestions(
    request: Request,
    form_data: SetDefaultSuggestionsForm,
    user=Depends(get_current_user),
):
    if user.role == "admin":
        data = form_data.model_dump()
        request.app.state.DEFAULT_PROMPT_SUGGESTIONS = data["suggestions"]
        return request.app.state.DEFAULT_PROMPT_SUGGESTIONS
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )
