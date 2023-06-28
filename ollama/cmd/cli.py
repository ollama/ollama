import os
import sys
import json
from pathlib import Path
from argparse import ArgumentParser

from ollama import model, engine
from ollama.cmd import server


def main():
    parser = ArgumentParser()
    parser.add_argument("--models-home", default=Path.home() / ".ollama" / "models")

    subparsers = parser.add_subparsers()

    server.set_parser(subparsers.add_parser("serve"))

    list_parser = subparsers.add_parser("list")
    list_parser.set_defaults(fn=list_models)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("model")
    generate_parser.add_argument("prompt", nargs="?")
    generate_parser.set_defaults(fn=generate)

    add_parser = subparsers.add_parser("add")
    add_parser.add_argument("model")
    add_parser.set_defaults(fn=add)

    pull_parser = subparsers.add_parser("pull")
    pull_parser.add_argument("remote")
    pull_parser.set_defaults(fn=pull)

    args = parser.parse_args()
    args = vars(args)

    try:
        fn = args.pop("fn")
        fn(**args)
    except KeyboardInterrupt:
        pass
    except KeyError:
        parser.print_help()
    except Exception as e:
        print(e)


def list_models(*args, **kwargs):
    for m in model.models(*args, **kwargs):
        print(m)


def generate(*args, **kwargs):
    if prompt := kwargs.get('prompt'):
        print('>>>', prompt, flush=True)
        print(flush=True)
        generate_oneshot(*args, **kwargs)
        print(flush=True)
        return

    return generate_interactive(*args, **kwargs)


def generate_oneshot(*args, **kwargs):
    for output in engine.generate(*args, **kwargs):
        output = json.loads(output)
        choices = output.get("choices", [])
        if len(choices) > 0:
            print(choices[0].get("text", ""), end="", flush=True)

    # end with a new line
    print()


def generate_interactive(*args, **kwargs):
    print('>>> ', end='', flush=True)
    for line in sys.stdin:
        if not sys.stdin.isatty():
            print(line, end='')

        print(flush=True)
        kwargs.update({'prompt': line})
        generate_oneshot(*args, **kwargs)
        print(flush=True)
        print('>>> ', end='', flush=True)


def add(model, models_home):
    os.rename(model, Path(models_home) / Path(model).name)


def pull(*args, **kwargs):
    model.pull(*args, **kwargs)
