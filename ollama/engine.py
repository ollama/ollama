import os
import sys
from os import path
from contextlib import contextmanager
from thefuzz import process
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
    model = load(model_name, models=models, *args, **kwargs)
    inputs = ollama.prompt.template(model_name, prompt)
    return model.generate(inputs, *args, **kwargs)


def load(model_name, models={}, *args, **kwargs):
    if not models.get(model_name, None):
        model_path = path.expanduser(model_name)
        if not path.exists(model_path):
            model_path = str(MODELS_CACHE_PATH / (model_name + ".bin"))

        runners = {
            model_type: cls
            for cls in [LlamaCppRunner, CtransformerRunner]
            for model_type in cls.model_types()
        }

        for match, _ in process.extract(model_path, runners.keys(), limit=len(runners)):
            try:
                model = runners.get(match)
                runner = model(model_path, match, *args, **kwargs)
                models.update({model_name: runner})
                return runner
            except Exception:
                pass

        raise Exception("failed to load model", model_path, model_name)


def unload(model_name, models={}):
    if model_name in models:
        models.pop(model_name)


class LlamaCppRunner:

    def __init__(self, model_path, model_type, *args, **kwargs):
        try:
            with suppress(sys.stderr), suppress(sys.stdout):
                self.model = Llama(model_path, verbose=False, n_gpu_layers=1, seed=kwargs.get('seed', -1))
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
        stop = [s for s in kwargs.get('stop', []) if s]
        if not stop:
            stop = ['Q: ']

        with suppress(sys.stderr):
            for output in self.model(
                    prompt,
                    max_tokens=kwargs.get('max_tokens', 512),
                    temperature=kwargs.get('temperature', 0.8),
                    top_p=kwargs.get('top_p', 0.95),
                    top_k=kwargs.get('top_k', 40),
                    repeat_penalty=kwargs.get('repeat_penalty', 1.1),
                    stream=kwargs.get('stream', True),
                    stop=stop):
                yield output


class CtransformerRunner:

    def __init__(self, model_path, model_type, *args, **kwargs):
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
        stop = [s for s in kwargs.get('stop', []) if s]
        if not stop:
            stop = ['User ']

        for output in self.model(
                prompt,
                max_new_tokens=kwargs.get('max_tokens', 512),
                temperature=kwargs.get('temperature', 0.8),
                top_p=kwargs.get('top_p', 0.95),
                top_k=kwargs.get('top_k', 40),
                repetition_penalty=kwargs.get('repeat_penalty', 1.1),
                stream=kwargs.get('stream', True),
                seed=kwargs.get('seed', -1),
                stop=stop):
            yield {
                'choices': [
                    {
                        'text': output,
                    },
                ],
            }
