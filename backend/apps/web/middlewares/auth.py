from apps.web.models.users import Users
from fastapi import Request, status
from starlette.authentication import (
    AuthCredentials, AuthenticationBackend, AuthenticationError, 
)
from starlette.requests import HTTPConnection
from utils.utils import verify_token
from starlette.responses import JSONResponse
from constants import ERROR_MESSAGES

class BearerTokenAuthBackend(AuthenticationBackend):

    async def authenticate(self, conn: HTTPConnection):
        if "Authorization" not in conn.headers:
            return
        data = verify_token(conn)
        if data != None and 'email' in data:
            user = Users.get_user_by_email(data['email'])
            if user is None:
                raise AuthenticationError('Invalid credentials') 
            return AuthCredentials([user.role]), user
        else:
            raise AuthenticationError('Invalid credentials') 

def on_auth_error(request: Request, exc: Exception):
    print('Authentication failed: ', exc)
    return JSONResponse({"detail": ERROR_MESSAGES.INVALID_TOKEN}, status_code=status.HTTP_401_UNAUTHORIZED)