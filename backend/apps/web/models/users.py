from pydantic import BaseModel
from typing import List, Union, Optional
from pymongo import ReturnDocument
import time

from utils.utils import decode_token
from utils.misc import get_gravatar_url

from config import DB

####################
# User DB Schema
####################


class UserModel(BaseModel):
    id: str
    name: str
    email: str
    role: str = "pending"
    profile_image_url: str = "/user.png"
    created_at: int  # timestamp in epoch


####################
# Forms
####################


class UserRoleUpdateForm(BaseModel):
    id: str
    role: str


class UsersTable:
    def __init__(self, db):
        self.db = db
        self.table = db.users

    def insert_new_user(
        self, id: str, name: str, email: str, role: str = "pending"
    ) -> Optional[UserModel]:
        user = UserModel(
            **{
                "id": id,
                "name": name,
                "email": email,
                "role": role,
                "profile_image_url": get_gravatar_url(email),
                "created_at": int(time.time()),
            }
        )
        result = self.table.insert_one(user.model_dump())

        if result:
            return user
        else:
            return None

    def get_user_by_email(self, email: str) -> Optional[UserModel]:
        user = self.table.find_one({"email": email}, {"_id": False})

        if user:
            return UserModel(**user)
        else:
            return None

    def get_user_by_token(self, token: str) -> Optional[UserModel]:
        data = decode_token(token)

        if data != None and "email" in data:
            return self.get_user_by_email(data["email"])
        else:
            return None

    def get_users(self, skip: int = 0, limit: int = 50) -> List[UserModel]:
        return [
            UserModel(**user)
            for user in list(
                self.table.find({}, {"_id": False}).skip(skip).limit(limit)
            )
        ]

    def get_num_users(self) -> Optional[int]:
        return self.table.count_documents({})

    def update_user_by_id(self, id: str, updated: dict) -> Optional[UserModel]:
        user = self.table.find_one_and_update(
            {"id": id}, {"$set": updated}, return_document=ReturnDocument.AFTER
        )
        return UserModel(**user)

    def update_user_role_by_id(self, id: str, role: str) -> Optional[UserModel]:
        return self.update_user_by_id(id, {"role": role})


Users = UsersTable(DB)
