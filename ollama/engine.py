import os
import sys
from os import path
from contextlib import contextmanager
from llama_cpp import Llama as LLM

import ollama.model
import ollama.prompt


@contextmanager
def suppress_stderr():
    stderr = os.dup(sys.stderr.fileno())
    with open(os.devnull, "w") as devnull:
        os.dup2(devnull.fileno(), sys.stderr.fileno())
        yield

    os.dup2(stderr, sys.stderr.fileno())


def generate(model, prompt, llms={}, *args, **kwargs):
    llm = load(model, llms=llms)

    prompt = ollama.prompt.template(model, prompt)
    if "max_tokens" not in kwargs:
        kwargs.update({"max_tokens": 16384})

    if "stop" not in kwargs:
        kwargs.update({"stop": ["Q:"]})

    if "stream" not in kwargs:
        kwargs.update({"stream": True})

    for output in llm(prompt, *args, **kwargs):
        yield output


def load(model, llms={}):
    llm = llms.get(model, None)
    if not llm:
        stored_model_path = path.join(ollama.model.models_home, model) + ".bin"
        if path.exists(stored_model_path):
            model_path = stored_model_path
        else:
            # try loading this as a path to a model, rather than a model name
            model_path = path.abspath(model)

        if not path.exists(model_path):
            raise Exception(f"Model not found: {model}")

        try:
            # suppress LLM's output
            with suppress_stderr():
                llm = LLM(model_path, verbose=False)
                llms.update({model: llm})
        except Exception as e:
            # e is sent to devnull, so create a generic exception
            raise Exception(f"Failed to load model: {model}")
    return llm


def unload(model, llms={}):
    if model in llms:
        llms.pop(model)
