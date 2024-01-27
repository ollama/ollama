from pydantic import BaseModel
from typing import List, Union, Optional
import time
import uuid
from peewee import *

from apps.web.models.users import UserModel, Users
from utils.utils import (
    verify_password,
    get_password_hash,
    bearer_scheme,
    create_token,
)

from apps.web.internal.db import DB

####################
# DB MODEL
####################


class Auth(Model):
    id = CharField(unique=True)
    email = CharField()
    password = CharField()
    active = BooleanField()

    class Meta:
        database = DB


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


class ProfileImageUrlForm(BaseModel):
    profile_image_url: str


class UpdateProfileForm(BaseModel):
    profile_image_url: str
    name: str


class UpdatePasswordForm(BaseModel):
    password: str
    new_password: str


class SignupForm(BaseModel):
    name: str
    email: str
    password: str


class AuthsTable:
    def __init__(self, db):
        self.db = db
        self.db.create_tables([Auth])

    def insert_new_auth(
        self, email: str, password: str, name: str, role: str = "pending"
    ) -> Optional[UserModel]:
        print("insert_new_auth")

        id = str(uuid.uuid4())

        auth = AuthModel(
            **{"id": id, "email": email, "password": password, "active": True}
        )
        result = Auth.create(**auth.model_dump())

        user = Users.insert_new_user(id, name, email, role)

        if result and user:
            return user
        else:
            return None

    def authenticate_user(self, email: str, password: str) -> Optional[UserModel]:
        print("authenticate_user", email)
        try:
            auth = Auth.get(Auth.email == email, Auth.active == True)
            if auth:
                if verify_password(password, auth.password):
                    user = Users.get_user_by_id(auth.id)
                    return user
                else:
                    return None
            else:
                return None
        except:
            return None

    def update_user_password_by_id(self, id: str, new_password: str) -> bool:
        try:
            query = Auth.update(password=new_password).where(Auth.id == id)
            result = query.execute()

            return True if result == 1 else False
        except:
            return False

    def update_email_by_id(self, id: str, email: str) -> bool:
        try:
            query = Auth.update(email=email).where(Auth.id == id)
            result = query.execute()

            return True if result == 1 else False
        except:
            return False

    def delete_auth_by_id(self, id: str) -> bool:
        try:
            # Delete User
            result = Users.delete_user_by_id(id)

            if result:
                # Delete Auth
                query = Auth.delete().where(Auth.id == id)
                query.execute()  # Remove the rows, return number of rows removed.

                return True
            else:
                return False
        except:
            return False


Auths = AuthsTable(DB)
