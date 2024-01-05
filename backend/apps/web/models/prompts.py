from pydantic import BaseModel
from peewee import *
from playhouse.shortcuts import model_to_dict
from typing import List, Union, Optional
import time

from utils.utils import decode_token
from utils.misc import get_gravatar_url

from apps.web.internal.db import DB

import json

####################
# Prompts DB Schema
####################


class Prompt(Model):
    command = CharField(unique=True)
    user_id = CharField()
    title = CharField()
    content = TextField()
    timestamp = DateField()

    class Meta:
        database = DB


class PromptModel(BaseModel):
    command: str
    user_id: str
    title: str
    content: str
    timestamp: int  # timestamp in epoch


####################
# Forms
####################


class PromptForm(BaseModel):
    command: str
    title: str
    content: str


class PromptsTable:

    def __init__(self, db):
        self.db = db
        self.db.create_tables([Prompt])

    def insert_new_prompt(self, user_id: str,
                          form_data: PromptForm) -> Optional[PromptModel]:
        prompt = PromptModel(
            **{
                "user_id": user_id,
                "command": form_data.command,
                "title": form_data.title,
                "content": form_data.content,
                "timestamp": int(time.time()),
            })

        try:
            result = Prompt.create(**prompt.model_dump())
            if result:
                return prompt
            else:
                return None
        except:
            return None

    def get_prompt_by_command(self, command: str) -> Optional[PromptModel]:
        try:
            prompt = Prompt.get(Prompt.command == command)
            return PromptModel(**model_to_dict(prompt))
        except:
            return None

    def get_prompts(self) -> List[PromptModel]:
        return [
            PromptModel(**model_to_dict(prompt)) for prompt in Prompt.select()
            # .limit(limit).offset(skip)
        ]

    def update_prompt_by_command(
            self, command: str,
            form_data: PromptForm) -> Optional[PromptModel]:
        try:
            query = Prompt.update(
                title=form_data.title,
                content=form_data.content,
                timestamp=int(time.time()),
            ).where(Prompt.command == command)

            query.execute()

            prompt = Prompt.get(Prompt.command == command)
            return PromptModel(**model_to_dict(prompt))
        except:
            return None

    def delete_prompt_by_command(self, command: str) -> bool:
        try:
            query = Prompt.delete().where((Prompt.command == command))
            query.execute()  # Remove the rows, return number of rows removed.

            return True
        except:
            return False


Prompts = PromptsTable(DB)
