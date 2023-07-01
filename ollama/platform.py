import os
import sys
import importlib
from subprocess import call
from contextlib import contextmanager


@contextmanager
def preserve_environ():
    env = os.environ.copy()
    yield
    os.environ = env


if not importlib.util.find_spec('llama_cpp'):
    with preserve_environ():
        os.environ['FORCE_CMAKE'] = '1'
        if os.uname().sysname.lower() == 'darwin':
            os.environ['CMAKE_ARGS'] = '-DLLAMA_METAL=on'

        call([sys.executable, '-m', 'pip', 'install', 'llama-cpp-python==0.1.67'])


from llama_cpp import Llama  # noqa: E402
