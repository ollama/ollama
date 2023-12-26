from fastapi import Response
from fastapi import Depends, FastAPI, HTTPException, status
from datetime import datetime, timedelta
from typing import List, Union, Optional

from fastapi import APIRouter
from pydantic import BaseModel
import json

from apps.web.models.users import Users
from apps.web.models.chats import (
    ChatModel,
    ChatResponse,
    ChatTitleForm,
    ChatForm,
    ChatTitleIdResponse,
    Chats,
)

from utils.utils import (
    bearer_scheme,
)
from constants import ERROR_MESSAGES

router = APIRouter()

############################
# GetChats
############################


@router.get("/", response_model=List[ChatTitleIdResponse])
async def get_user_chats(skip: int = 0, limit: int = 50, cred=Depends(bearer_scheme)):
    token = cred.credentials
    user = Users.get_user_by_token(token)

    if user:
        return Chats.get_chat_lists_by_user_id(user.id, skip, limit)
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.INVALID_TOKEN,
        )


############################
# CreateNewChat
############################


@router.post("/new", response_model=Optional[ChatResponse])
async def create_new_chat(form_data: ChatForm, cred=Depends(bearer_scheme)):
    token = cred.credentials
    user = Users.get_user_by_token(token)

    if user:
        chat = Chats.insert_new_chat(user.id, form_data)
        return ChatResponse(**{**chat.model_dump(), "chat": json.loads(chat.chat)})
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.INVALID_TOKEN,
        )


############################
# GetChatById
############################


@router.get("/{id}", response_model=Optional[ChatResponse])
async def get_chat_by_id(id: str, cred=Depends(bearer_scheme)):
    token = cred.credentials
    user = Users.get_user_by_token(token)

    if user:
        chat = Chats.get_chat_by_id_and_user_id(id, user.id)

        if chat:
            return ChatResponse(**{**chat.model_dump(), "chat": json.loads(chat.chat)})
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
# UpdateChatById
############################


@router.post("/{id}", response_model=Optional[ChatResponse])
async def update_chat_by_id(id: str, form_data: ChatForm, cred=Depends(bearer_scheme)):
    token = cred.credentials
    user = Users.get_user_by_token(token)

    if user:
        chat = Chats.get_chat_by_id_and_user_id(id, user.id)
        if chat:
            updated_chat = {**json.loads(chat.chat), **form_data.chat}

            chat = Chats.update_chat_by_id(id, updated_chat)
            return ChatResponse(**{**chat.model_dump(), "chat": json.loads(chat.chat)})
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
# DeleteChatById
############################


@router.delete("/{id}", response_model=bool)
async def delete_chat_by_id(id: str, cred=Depends(bearer_scheme)):
    token = cred.credentials
    user = Users.get_user_by_token(token)

    if user:
        result = Chats.delete_chat_by_id_and_user_id(id, user.id)
        return result
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.INVALID_TOKEN,
        )
