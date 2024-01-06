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
# Modelfile DB Schema
####################


class Modelfile(Model):
    tag_name = CharField(unique=True)
    user_id = CharField()
    modelfile = TextField()
    timestamp = DateField()

    class Meta:
        database = DB


class ModelfileModel(BaseModel):
    tag_name: str
    user_id: str
    modelfile: str
    timestamp: int  # timestamp in epoch


####################
# Forms
####################


class ModelfileForm(BaseModel):
    modelfile: dict


class ModelfileTagNameForm(BaseModel):
    tag_name: str


class ModelfileUpdateForm(ModelfileForm, ModelfileTagNameForm):
    pass


class ModelfileResponse(BaseModel):
    tag_name: str
    user_id: str
    modelfile: dict
    timestamp: int  # timestamp in epoch


class ModelfilesTable:

    def __init__(self, db):
        self.db = db
        self.db.create_tables([Modelfile])

    def insert_new_modelfile(
            self, user_id: str,
            form_data: ModelfileForm) -> Optional[ModelfileModel]:
        if "tagName" in form_data.modelfile:
            modelfile = ModelfileModel(
                **{
                    "user_id": user_id,
                    "tag_name": form_data.modelfile["tagName"],
                    "modelfile": json.dumps(form_data.modelfile),
                    "timestamp": int(time.time()),
                })

            try:
                result = Modelfile.create(**modelfile.model_dump())
                if result:
                    return modelfile
                else:
                    return None
            except:
                return None

        else:
            return None

    def get_modelfile_by_tag_name(self,
                                  tag_name: str) -> Optional[ModelfileModel]:
        try:
            modelfile = Modelfile.get(Modelfile.tag_name == tag_name)
            return ModelfileModel(**model_to_dict(modelfile))
        except:
            return None

    def get_modelfiles(self,
                       skip: int = 0,
                       limit: int = 50) -> List[ModelfileResponse]:
        return [
            ModelfileResponse(
                **{
                    **model_to_dict(modelfile),
                    "modelfile":
                    json.loads(modelfile.modelfile),
                }) for modelfile in Modelfile.select()
            # .limit(limit).offset(skip)
        ]

    def update_modelfile_by_tag_name(
            self, tag_name: str, modelfile: dict) -> Optional[ModelfileModel]:
        try:
            query = Modelfile.update(
                modelfile=json.dumps(modelfile),
                timestamp=int(time.time()),
            ).where(Modelfile.tag_name == tag_name)

            query.execute()

            modelfile = Modelfile.get(Modelfile.tag_name == tag_name)
            return ModelfileModel(**model_to_dict(modelfile))
        except:
            return None

    def delete_modelfile_by_tag_name(self, tag_name: str) -> bool:
        try:
            query = Modelfile.delete().where((Modelfile.tag_name == tag_name))
            query.execute()  # Remove the rows, return number of rows removed.

            return True
        except:
            return False


Modelfiles = ModelfilesTable(DB)
