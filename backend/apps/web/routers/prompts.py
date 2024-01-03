from fastapi import Depends, FastAPI, HTTPException, status
from datetime import datetime, timedelta
from typing import List, Union, Optional

from fastapi import APIRouter
from pydantic import BaseModel
import json


from apps.web.models.prompts import Prompts, PromptForm, PromptModel

from utils.utils import get_current_user
from constants import ERROR_MESSAGES

router = APIRouter()

############################
# GetPrompts
############################


@router.get("/", response_model=List[PromptModel])
async def get_prompts(user=Depends(get_current_user)):
    return Prompts.get_prompts()


############################
# CreateNewPrompt
############################


@router.post("/create", response_model=Optional[PromptModel])
async def create_new_prompt(form_data: PromptForm, user=Depends(get_current_user)):
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    prompt = Prompts.insert_new_prompt(user.id, form_data)

    if prompt:
        return prompt
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.DEFAULT(),
        )


############################
# GetPromptByCommand
############################


@router.get("/{command}", response_model=Optional[PromptModel])
async def get_prompt_by_command(command: str, user=Depends(get_current_user)):
    prompt = Prompts.get_prompt_by_command(command)

    if prompt:
        return prompt
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )


############################
# UpdatePromptByCommand
############################


@router.post("/{command}/update", response_model=Optional[PromptModel])
async def update_prompt_by_command(
    command: str, form_data: PromptForm, user=Depends(get_current_user)
):
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    prompt = Prompts.update_prompt_by_command(command, form_data)
    if prompt:
        return prompt
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )


############################
# DeletePromptByCommand
############################


@router.delete("/{command}/delete", response_model=bool)
async def delete_prompt_by_command(command: str, user=Depends(get_current_user)):
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    result = Prompts.delete_prompt_by_command(command)
    return result
