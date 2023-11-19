from enum import Enum


class MESSAGES(str, Enum):
    DEFAULT = lambda msg="": f"{msg if msg else ''}"


class ERROR_MESSAGES(str, Enum):
    DEFAULT = lambda err="": f"Something went wrong :/\n{err if err else ''}"
    UNAUTHORIZED = "401 Unauthorized"
    NOT_FOUND = "We could not find what you're looking for :/"
    USER_NOT_FOUND = "We could not find what you're looking for :/"
    MALICIOUS = "Unusual activities detected, please try again in a few minutes."
