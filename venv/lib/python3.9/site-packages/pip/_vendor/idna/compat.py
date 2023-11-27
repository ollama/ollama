from .core import *
from .codec import *
from typing import Any, Union

def ToASCII(label):
    # type: (str) -> bytes
    return encode(label)

def ToUnicode(label):
    # type: (Union[bytes, bytearray]) -> str
    return decode(label)

def nameprep(s):
    # type: (Any) -> None
    raise NotImplementedError('IDNA 2008 does not utilise nameprep protocol')

