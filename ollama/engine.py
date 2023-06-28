import os
import sys
from os import path
from contextlib import contextmanager
from llama_cpp import Llama as LLM

import ollama.prompt
from ollama.model import models_home


@contextmanager
def suppress_stderr():
    stderr = os.dup(sys.stderr.fileno())
    with open(os.devnull, "w") as devnull:
        os.dup2(devnull.fileno(), sys.stderr.fileno())
        yield

    os.dup2(stderr, sys.stderr.fileno())


def generate(model_name, prompt, models={}, *args, **kwargs):
    if "max_tokens" not in kwargs:
        kwargs.update({"max_tokens": 16384})

    if "stop" not in kwargs:
        kwargs.update({"stop": ["Q:"]})

    if "stream" not in kwargs:
        kwargs.update({"stream": True})

    prompt = ollama.prompt.template(model_name, prompt)

    model = load(model_name, models=models)
    for output in model.create_completion(prompt, *args, **kwargs):
        yield output


def load(model_name, models={}):
    model = models.get(model_name, None)
    if not model:
        model_path = path.expanduser(model_name)
        if not path.exists(model_path):
            model_path = path.join(models_home, model_name + ".bin")

        try:
            # suppress LLM's output
            with suppress_stderr():
                model = LLM(model_path, verbose=False)
                models.update({model_name: model})
        except Exception:
            # e is sent to devnull, so create a generic exception
            raise Exception(f"Failed to load model: {model}")

    return model


def unload(model_name, models={}):
    if model_name in models:
        models.pop(model_name)
