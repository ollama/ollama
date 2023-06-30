import os
import sys
from argparse import ArgumentParser, HelpFormatter, PARSER
from yaspin import yaspin

from ollama import model, engine
from ollama.cmd import server


class CustomHelpFormatter(HelpFormatter):
    """
    This class is used to customize the way the argparse help text is displayed.
    We specifically override the _format_action method to exclude the line that
    shows all the subparser command options in the help text. This line is typically
    in the form "{serve,models,pull,run}".
    """

    def _format_action(self, action):
        # get the original help text
        parts = super()._format_action(action)
        if action.nargs == PARSER:
            # remove the unwanted first line
            parts = "\n".join(parts.split("\n")[1:])
        return parts


def main():
    parser = ArgumentParser(
        description='Ollama: Run any large language model on any machine.',
        formatter_class=CustomHelpFormatter,
    )

    # create models home if it doesn't exist
    os.makedirs(model.MODELS_CACHE_PATH, exist_ok=True)

    subparsers = parser.add_subparsers(
        title='commands',
    )

    server.set_parser(
        subparsers.add_parser(
            "serve",
            description="Start a persistent server to interact with models via the API.",
            help="Start a persistent server to interact with models via the API.",
        )
    )

    list_parser = subparsers.add_parser(
        "models",
        description="List all available models stored locally.",
        help="List all available models stored locally.",
    )
    list_parser.set_defaults(fn=list_models)

    pull_parser = subparsers.add_parser(
        "pull",
        description="Download a specified model from a remote source.",
        help="Download a specified model from a remote source. Usage: pull [model]",
    )
    pull_parser.add_argument("model", help="Name of the model to download.")
    pull_parser.set_defaults(fn=pull)

    run_parser = subparsers.add_parser(
        "run",
        description="Run a model and submit prompts.",
        help="Run a model and submit prompts. Usage: run [model] [prompt]",
    )
    run_parser.add_argument("model", help="Name of the model to run.")
    run_parser.add_argument(
        "prompt",
        nargs="?",
        help="Optional prompt for the model, interactive mode enabled when not specified.",
    )
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
        for output in engine.generate(model_name=kwargs.pop('model'), *args, **kwargs):
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
    model.pull(model_name=kwargs.pop('model'), *args, **kwargs)


def run(*args, **kwargs):
    name = model.pull(*args, **kwargs)
    kwargs.update({"model": name})
    print(f"Running {name}...")
    generate(*args, **kwargs)
