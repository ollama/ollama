import os
import sys
from pathlib import Path
from argparse import ArgumentParser

from ollama import model, engine
from ollama.cmd import server


def main():
    parser = ArgumentParser()
    parser.add_argument("--models-home", default=Path.home() / ".ollama" / "models")

    # create models home if it doesn't exist
    models_home = parser.parse_known_args()[0].models_home
    if not models_home.exists():
        os.makedirs(models_home)

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
    pull_parser.add_argument("model")
    pull_parser.set_defaults(fn=pull)

    pull_parser = subparsers.add_parser("run")
    pull_parser.add_argument("model")
    pull_parser.add_argument("prompt", nargs="?")
    pull_parser.set_defaults(fn=run)

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
    if prompt := kwargs.get("prompt"):
        print(">>>", prompt, flush=True)
        generate_oneshot(*args, **kwargs)
        return

    if sys.stdin.isatty():
        return generate_interactive(*args, **kwargs)

    return generate_batch(*args, **kwargs)


def generate_oneshot(*args, **kwargs):
    print(flush=True)

    for output in engine.generate(*args, **kwargs):
        choices = output.get("choices", [])
        if len(choices) > 0:
            print(choices[0].get("text", ""), end="", flush=True)

    # end with a new line
    print(flush=True)
    print(flush=True)


def generate_interactive(*args, **kwargs):
    while True:
        print(">>> ", end="", flush=True)
        line = next(sys.stdin)
        if not line:
            return

        kwargs.update({"prompt": line})
        generate_oneshot(*args, **kwargs)


def generate_batch(*args, **kwargs):
    for line in sys.stdin:
        print(">>> ", line, end="", flush=True)
        kwargs.update({"prompt": line})
        generate_oneshot(*args, **kwargs)


def add(model, models_home):
    os.rename(model, Path(models_home) / Path(model).name)


def pull(*args, **kwargs):
    model.pull(*args, **kwargs)


def run(*args, **kwargs):
    name = model.pull(*args, **kwargs)
    kwargs.update({"model": name})
    print(f"Running {name}...")
    generate(*args, **kwargs)
