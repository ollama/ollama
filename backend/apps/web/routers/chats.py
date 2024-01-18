from fastapi import Depends, Request, HTTPException, status
from datetime import datetime, timedelta
from typing import List, Union, Optional
from utils.utils import get_current_user
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


from apps.web.models.tags import (
    TagModel,
    ChatIdTagModel,
    ChatIdTagForm,
    ChatTagsResponse,
    Tags,
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
async def get_user_chats(
    user=Depends(get_current_user), skip: int = 0, limit: int = 50
):
    return Chats.get_chat_lists_by_user_id(user.id, skip, limit)


############################
# GetAllChats
############################


@router.get("/all", response_model=List[ChatResponse])
async def get_all_user_chats(user=Depends(get_current_user)):
    return [
        ChatResponse(**{**chat.model_dump(), "chat": json.loads(chat.chat)})
        for chat in Chats.get_all_chats_by_user_id(user.id)
    ]


############################
# CreateNewChat
############################


@router.post("/new", response_model=Optional[ChatResponse])
async def create_new_chat(form_data: ChatForm, user=Depends(get_current_user)):
    try:
        chat = Chats.insert_new_chat(user.id, form_data)
        return ChatResponse(**{**chat.model_dump(), "chat": json.loads(chat.chat)})
    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )


############################
# GetAllTags
############################


@router.get("/tags/all", response_model=List[TagModel])
async def get_all_tags(user=Depends(get_current_user)):
    try:
        tags = Tags.get_tags_by_user_id(user.id)
        return tags
    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )


############################
# GetChatsByTags
############################


@router.get("/tags/tag/{tag_name}", response_model=List[ChatTitleIdResponse])
async def get_user_chats_by_tag_name(
    tag_name: str, user=Depends(get_current_user), skip: int = 0, limit: int = 50
):
    chat_ids = [
        chat_id_tag.chat_id
        for chat_id_tag in Tags.get_chat_ids_by_tag_name_and_user_id(tag_name, user.id)
    ]

    print(chat_ids)

    return Chats.get_chat_lists_by_chat_ids(chat_ids, skip, limit)


############################
# GetChatById
############################


@router.get("/{id}", response_model=Optional[ChatResponse])
async def get_chat_by_id(id: str, user=Depends(get_current_user)):
    chat = Chats.get_chat_by_id_and_user_id(id, user.id)

    if chat:
        return ChatResponse(**{**chat.model_dump(), "chat": json.loads(chat.chat)})
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=ERROR_MESSAGES.NOT_FOUND
        )


############################
# UpdateChatById
############################


@router.post("/{id}", response_model=Optional[ChatResponse])
async def update_chat_by_id(
    id: str, form_data: ChatForm, user=Depends(get_current_user)
):
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


############################
# DeleteChatById
############################


@router.delete("/{id}", response_model=bool)
async def delete_chat_by_id(id: str, user=Depends(get_current_user)):
    result = Chats.delete_chat_by_id_and_user_id(id, user.id)
    return result


############################
# GetChatTagsById
############################


@router.get("/{id}/tags", response_model=List[TagModel])
async def get_chat_tags_by_id(id: str, user=Depends(get_current_user)):
    tags = Tags.get_tags_by_chat_id_and_user_id(id, user.id)

    if tags != None:
        return tags
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=ERROR_MESSAGES.NOT_FOUND
        )


############################
# AddChatTagById
############################


@router.post("/{id}/tags", response_model=Optional[ChatIdTagModel])
async def add_chat_tag_by_id(
    id: str, form_data: ChatIdTagForm, user=Depends(get_current_user)
):
    tags = Tags.get_tags_by_chat_id_and_user_id(id, user.id)

    if form_data.tag_name not in tags:
        tag = Tags.add_tag_to_chat(user.id, form_data)

        if tag:
            return tag
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=ERROR_MESSAGES.DEFAULT()
        )


############################
# DeleteChatTagById
############################


@router.delete("/{id}/tags", response_model=Optional[bool])
async def delete_chat_tag_by_id(
    id: str, form_data: ChatIdTagForm, user=Depends(get_current_user)
):
    result = Tags.delete_tag_by_tag_name_and_chat_id_and_user_id(
        form_data.tag_name, id, user.id
    )

    if result:
        return result
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=ERROR_MESSAGES.NOT_FOUND
        )


############################
# DeleteAllChatTagsById
############################


@router.delete("/{id}/tags/all", response_model=Optional[bool])
async def delete_all_chat_tags_by_id(id: str, user=Depends(get_current_user)):
    result = Tags.delete_tags_by_chat_id_and_user_id(id, user.id)

    if result:
        return result
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=ERROR_MESSAGES.NOT_FOUND
        )


############################
# DeleteAllChats
############################


@router.delete("/", response_model=bool)
async def delete_all_user_chats(user=Depends(get_current_user)):
    result = Chats.delete_chats_by_user_id(user.id)
    return result
