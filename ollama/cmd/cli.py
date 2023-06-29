import os
import sys
from pathlib import Path
from argparse import ArgumentParser
from yaspin import yaspin

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

    pull_parser = subparsers.add_parser("pull")
    pull_parser.add_argument("model")
    pull_parser.set_defaults(fn=pull)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("model")
    run_parser.add_argument("prompt", nargs="?")
    run_parser.set_defaults(fn=run)

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

    spinner = yaspin()
    spinner.start()
    spinner_running = True
    try:
        for output in engine.generate(*args, **kwargs):
            choices = output.get("choices", [])
            if len(choices) > 0:
                if spinner_running:
                    spinner.stop()
                    spinner_running = False
                    print("\r", end="")  # move cursor back to beginning of line again
                print(choices[0].get("text", ""), end="", flush=True)
    except Exception:
        spinner.stop()
        raise

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


def pull(*args, **kwargs):
    model.pull(*args, **kwargs)


def run(*args, **kwargs):
    name = model.pull(*args, **kwargs)
    kwargs.update({"model": name})
    print(f"Running {name}...")
    generate(*args, **kwargs)
