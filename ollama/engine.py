import os
import json
import sys
from contextlib import contextmanager
from llama_cpp import Llama as LLM
from template import template

import ollama.model


@contextmanager
def suppress_stderr():
    stderr = os.dup(sys.stderr.fileno())
    with open(os.devnull, "w") as devnull:
        os.dup2(devnull.fileno(), sys.stderr.fileno())
        yield

    os.dup2(stderr, sys.stderr.fileno())


def generate(model, prompt, models_home=".", llms={}, *args, **kwargs):
    llm = load(model, models_home=models_home, llms=llms)

    prompt = template(model, prompt)

    if "max_tokens" not in kwargs:
        kwargs.update({"max_tokens": 16384})

    if "stop" not in kwargs:
        kwargs.update({"stop": ["Q:", "\n"]})

    if "stream" not in kwargs:
        kwargs.update({"stream": True})

    for output in llm(prompt, *args, **kwargs):
        yield json.dumps(output)


def load(model, models_home=".", llms={}):
    llm = llms.get(model, None)
    if not llm:
        model_path = {
            name: path for name, path in ollama.model.models(models_home)
        }.get(model, None)

        if model_path is None:
            # try loading this as a path to a model, rather than a model name
            if os.path.isfile(model):
                model_path = model
            else:
                raise ValueError("Model not found")

        # suppress LLM's output
        with suppress_stderr():
            llm = LLM(model_path, verbose=False)
            llms.update({model: llm})

    return llm


def unload(model, llms={}):
    if model in llms:
        llms.pop(model)
