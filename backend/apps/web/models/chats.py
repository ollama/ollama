from pydantic import BaseModel
from typing import List, Union, Optional
from peewee import *
from playhouse.shortcuts import model_to_dict

import json
import uuid
import time

from apps.web.internal.db import DB

####################
# Chat DB Schema
####################


class Chat(Model):
    id = CharField(unique=True)
    user_id = CharField()
    title = CharField()
    chat = TextField()  # Save Chat JSON as Text
    timestamp = DateField()

    class Meta:
        database = DB


class ChatModel(BaseModel):
    id: str
    user_id: str
    title: str
    chat: str
    timestamp: int  # timestamp in epoch


####################
# Forms
####################


class ChatForm(BaseModel):
    chat: dict


class ChatTitleForm(BaseModel):
    title: str


class ChatResponse(BaseModel):
    id: str
    user_id: str
    title: str
    chat: dict
    timestamp: int  # timestamp in epoch


class ChatTitleIdResponse(BaseModel):
    id: str
    title: str


class ChatTable:
    def __init__(self, db):
        self.db = db
        db.create_tables([Chat])

    def insert_new_chat(self, user_id: str, form_data: ChatForm) -> Optional[ChatModel]:
        id = str(uuid.uuid4())
        chat = ChatModel(
            **{
                "id": id,
                "user_id": user_id,
                "title": form_data.chat["title"]
                if "title" in form_data.chat
                else "New Chat",
                "chat": json.dumps(form_data.chat),
                "timestamp": int(time.time()),
            }
        )

        result = Chat.create(**chat.model_dump())
        return chat if result else None

    def update_chat_by_id(self, id: str, chat: dict) -> Optional[ChatModel]:
        try:
            query = Chat.update(
                chat=json.dumps(chat),
                title=chat["title"] if "title" in chat else "New Chat",
                timestamp=int(time.time()),
            ).where(Chat.id == id)
            query.execute()

            chat = Chat.get(Chat.id == id)
            return ChatModel(**model_to_dict(chat))
        except:
            return None

    def update_chat_by_id(self, id: str, chat: dict) -> Optional[ChatModel]:
        try:
            query = Chat.update(
                chat=json.dumps(chat),
                title=chat["title"] if "title" in chat else "New Chat",
                timestamp=int(time.time()),
            ).where(Chat.id == id)
            query.execute()

            chat = Chat.get(Chat.id == id)
            return ChatModel(**model_to_dict(chat))
        except:
            return None

    def get_chat_lists_by_user_id(
        self, user_id: str, skip: int = 0, limit: int = 50
    ) -> List[ChatModel]:
        return [
            ChatModel(**model_to_dict(chat))
            for chat in Chat.select()
            .where(Chat.user_id == user_id)
            .order_by(Chat.timestamp.desc())
            # .limit(limit)
            # .offset(skip)
        ]

    def get_chat_lists_by_chat_ids(
        self, chat_ids: List[str], skip: int = 0, limit: int = 50
    ) -> List[ChatModel]:
        return [
            ChatModel(**model_to_dict(chat))
            for chat in Chat.select()
            .where(Chat.id.in_(chat_ids))
            .order_by(Chat.timestamp.desc())
        ]

    def get_all_chats_by_user_id(self, user_id: str) -> List[ChatModel]:
        return [
            ChatModel(**model_to_dict(chat))
            for chat in Chat.select()
            .where(Chat.user_id == user_id)
            .order_by(Chat.timestamp.desc())
        ]

    def get_chat_by_id_and_user_id(self, id: str, user_id: str) -> Optional[ChatModel]:
        try:
            chat = Chat.get(Chat.id == id, Chat.user_id == user_id)
            return ChatModel(**model_to_dict(chat))
        except:
            return None

    def get_chats(self, skip: int = 0, limit: int = 50) -> List[ChatModel]:
        return [
            ChatModel(**model_to_dict(chat))
            for chat in Chat.select().limit(limit).offset(skip)
        ]

    def delete_chat_by_id_and_user_id(self, id: str, user_id: str) -> bool:
        try:
            query = Chat.delete().where((Chat.id == id) & (Chat.user_id == user_id))
            query.execute()  # Remove the rows, return number of rows removed.

            return True
        except:
            return False

    def delete_chats_by_user_id(self, user_id: str) -> bool:
        try:
            query = Chat.delete().where(Chat.user_id == user_id)
            query.execute()  # Remove the rows, return number of rows removed.

            return True
        except:
            return False


Chats = ChatTable(DB)
