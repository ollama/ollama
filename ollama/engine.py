import os
import sys
from os import path
from pathlib import Path
from contextlib import contextmanager
from fuzzywuzzy import process
from llama_cpp import Llama
from ctransformers import AutoModelForCausalLM

import ollama.prompt
from ollama.model import MODELS_CACHE_PATH


@contextmanager
def suppress(file):
    original = os.dup(file.fileno())
    with open(os.devnull, "w") as devnull:
        os.dup2(devnull.fileno(), file.fileno())
        yield

    os.dup2(original, file.fileno())


def generate(model_name, prompt, models={}, *args, **kwargs):
    model = load(model_name, models=models)
    inputs = ollama.prompt.template(model_name, prompt)
    return model.generate(inputs, *args, **kwargs)


def load(model_name, models={}):
    if not models.get(model_name, None):
        model_path = path.expanduser(model_name)
        if not path.exists(model_path):
            model_path = str(MODELS_CACHE_PATH / (model_name + ".bin"))

        runners = {
            model_type: cls
            for cls in [LlamaCppRunner, CtransformerRunner]
            for model_type in cls.model_types()
        }

        best_match, _ = process.extractOne(model_path, runners.keys())
        model = runners.get(best_match, LlamaCppRunner)

        models.update({model_name: model(model_path, best_match)})

    return models.get(model_name)


def unload(model_name, models={}):
    if model_name in models:
        models.pop(model_name)


class LlamaCppRunner:
    def __init__(self, model_path, model_type):
        try:
            with suppress(sys.stderr), suppress(sys.stdout):
                self.model = Llama(model_path, verbose=False, n_gpu_layers=1, seed=-1)
        except Exception:
            raise Exception("Failed to load model", model_path, model_type)

    @staticmethod
    def model_types():
        return [
            'llama',
            'orca',
            'vicuna',
            'ultralm',
        ]

    def generate(self, prompt, *args, **kwargs):
        if "max_tokens" not in kwargs:
            kwargs.update({"max_tokens": 512})

        if "stop" not in kwargs:
            kwargs.update({"stop": ["Q:"]})

        if "stream" not in kwargs:
            kwargs.update({"stream": True})

        with suppress(sys.stderr):
            for output in self.model(prompt, *args, **kwargs):
                yield output


class CtransformerRunner:
    def __init__(self, model_path, model_type):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, model_type=model_type, local_files_only=True
        )

    @staticmethod
    def model_types():
        return [
            'falcon',
            'mpt',
            'starcoder',
        ]

    def generate(self, prompt, *args, **kwargs):
        if "max_new_tokens" not in kwargs:
            kwargs.update({"max_new_tokens": 512})

        if "stop" not in kwargs:
            kwargs.update({"stop": ["User"]})

        if "stream" not in kwargs:
            kwargs.update({"stream": True})

        for output in self.model(prompt, *args, **kwargs):
            yield {
                'choices': [
                    {
                        'text': output,
                    },
                ],
            }
