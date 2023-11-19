from pydantic import BaseModel
from typing import List, Union, Optional
import time
import uuid


from apps.web.models.users import UserModel, Users
from utils.utils import (
    verify_password,
    get_password_hash,
    bearer_scheme,
    create_token,
)

import config

DB = config.DB

####################
# DB MODEL
####################


class AuthModel(BaseModel):
    id: str
    email: str
    password: str
    active: bool = True


####################
# Forms
####################


class Token(BaseModel):
    token: str
    token_type: str


class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    role: str
    profile_image_url: str


class SigninResponse(Token, UserResponse):
    pass


class SigninForm(BaseModel):
    email: str
    password: str


class SignupForm(BaseModel):
    name: str
    email: str
    password: str


class AuthsTable:
    def __init__(self, db):
        self.db = db
        self.table = db.auths

    def insert_new_auth(
        self, email: str, password: str, name: str, role: str = "pending"
    ) -> Optional[UserModel]:
        print("insert_new_auth")

        id = str(uuid.uuid4())

        auth = AuthModel(
            **{"id": id, "email": email, "password": password, "active": True}
        )
        result = self.table.insert_one(auth.model_dump())
        user = Users.insert_new_user(id, name, email, role)

        print(result, user)
        if result and user:
            return user
        else:
            return None

    def authenticate_user(self, email: str, password: str) -> Optional[UserModel]:
        print("authenticate_user")

        auth = self.table.find_one({"email": email, "active": True})

        if auth:
            if verify_password(password, auth["password"]):
                user = self.db.users.find_one({"id": auth["id"]})
                return UserModel(**user)
            else:
                return None
        else:
            return None


Auths = AuthsTable(DB)
