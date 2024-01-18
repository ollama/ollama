from pydantic import BaseModel
from typing import List, Union, Optional
from peewee import *
from playhouse.shortcuts import model_to_dict

import json
import uuid
import time

from apps.web.internal.db import DB

####################
# Tag DB Schema
####################


class Tag(Model):
    name = CharField(unique=True)
    user_id = CharField()
    data = TextField(null=True)

    class Meta:
        database = DB


class ChatIdTag(Model):
    tag_name = ForeignKeyField(Tag, backref="chat_id_tags")
    chat_id = CharField()
    user_id = CharField()
    timestamp = DateField()

    class Meta:
        database = DB


class TagModel(BaseModel):
    name: str
    user_id: str
    data: Optional[str] = None


class ChatIdTagModel(BaseModel):
    tag_name: str
    chat_id: str
    user_id: str
    timestamp: int


####################
# Forms
####################


class ChatIdTagForm(BaseModel):
    tag_name: str
    chat_id: str


class TagChatIdsResponse(BaseModel):
    chat_ids: List[str]


class ChatTagsResponse(BaseModel):
    tags: List[str]


class TagTable:
    def __init__(self, db):
        self.db = db
        db.create_tables([Tag, ChatIdTag])

    def insert_new_tag(self, name: str, user_id: str) -> Optional[TagModel]:
        tag = TagModel(**{"user_id": user_id, "name": name})
        try:
            result = Tag.create(**tag.model_dump())
            if result:
                return tag
            else:
                return None
        except:
            return None

    def get_tag_by_name_and_user_id(
        self, name: str, user_id: str
    ) -> Optional[TagModel]:
        try:
            tag = Tag.get(Tag.name == name, Tag.user_id == user_id)
            return TagModel(**model_to_dict(tag))
        except:
            return None

    def add_tag_to_chat(
        self, user_id: str, form_data: ChatIdTagForm
    ) -> Optional[ChatTagsResponse]:
        tag = self.get_tag_by_name_and_user_id(form_data.tag_name, user_id)
        if tag == None:
            tag = self.insert_new_tag(form_data.tag_name, user_id)

        chatIdTag = ChatIdTagModel(**{"user_id": user_id, "tag_name": tag.name})
        try:
            result = ChatIdTag.create(**chatIdTag.model_dump())
            if result:
                return chatIdTag
            else:
                return None
        except:
            return None

    def get_tags_by_chat_id_and_user_id(
        self, chat_id: str, user_id: str
    ) -> List[TagModel]:
        return [
            TagModel(**model_to_dict(tag))
            for tag in Tag.select().where(
                Tag.name
                in [
                    ChatIdTagModel(**model_to_dict(chat_id_tag)).tag_name
                    for chat_id_tag in ChatIdTag.select()
                    .where(
                        (ChatIdTag.user_id == user_id) & (ChatIdTag.chat_id == chat_id)
                    )
                    .order_by(ChatIdTag.timestamp.desc())
                ]
            )
        ]

    def get_chat_ids_by_tag_name_and_user_id(
        self, tag_name: str, user_id: str
    ) -> Optional[ChatIdTagModel]:
        return [
            ChatIdTagModel(**model_to_dict(chat_id_tag))
            for chat_id_tag in ChatIdTag.select()
            .where((ChatIdTag.user_id == user_id) & (ChatIdTag.tag_name == tag_name))
            .order_by(ChatIdTag.timestamp.desc())
        ]

    def count_chat_ids_by_tag_name_and_user_id(
        self, tag_name: str, user_id: str
    ) -> int:
        return (
            ChatIdTag.select()
            .where((ChatIdTag.tag_name == tag_name) & (ChatIdTag.user_id == user_id))
            .count()
        )

    def delete_tag_by_tag_name_and_chat_id_and_user_id(
        self, tag_name: str, chat_id: str, user_id: str
    ) -> bool:
        try:
            query = ChatIdTag.delete().where(
                (ChatIdTag.tag_name == tag_name)
                & (ChatIdTag.chat_id == chat_id)
                & (ChatIdTag.user_id == user_id)
            )
            query.execute()  # Remove the rows, return number of rows removed.

            tag_count = self.count_chat_ids_by_tag_name_and_user_id(tag_name, user_id)
            if tag_count == 0:
                # Remove tag item from Tag col as well
                query = Tag.delete().where(
                    (Tag.name == tag_name) & (Tag.user_id == user_id)
                )
                query.execute()  # Remove the rows, return number of rows removed.

            return True
        except:
            return False

    def delete_tags_by_chat_id_and_user_id(self, chat_id: str, user_id: str) -> bool:
        tags = self.get_tags_by_chat_id_and_user_id(chat_id, user_id)

        for tag in tags:
            self.delete_tag_by_tag_name_and_chat_id_and_user_id(
                tag.tag_name, chat_id, user_id
            )

        return True


Tags = TagTable(DB)
