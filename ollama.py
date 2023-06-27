import json
import os
import threading
import click
from tqdm import tqdm
from pathlib import Path
from llama_cpp import Llama
from flask import Flask, Response, stream_with_context, request
from flask_cors import CORS
from template import template

app = Flask(__name__)
CORS(app)  # enable CORS for all routes

# llms tracks which models are loaded
llms = {}
lock = threading.Lock()


def models_directory():
    home_dir = Path.home()
    models_dir = home_dir / ".ollama/models"

    if not models_dir.exists():
        models_dir.mkdir(parents=True)

    return models_dir


def load(model):
    """
    Load a model.

    Args:
    model (str): The name or path of the model to load.

    Returns:
    str or None: The name of the model
    dict or None: If the model cannot be loaded, a dictionary with an 'error' key is returned.
                  If the model is successfully loaded, None is returned.
    """

    with lock:
        load_from = ""
        if os.path.exists(model) and model.endswith(".bin"):
            # model is being referenced by path rather than name directly
            path = os.path.abspath(model)
            base = os.path.basename(path)

            load_from = path
            name = os.path.splitext(base)[0]  # Split the filename and extension
        else:
            # model is being loaded from the ollama models directory
            dir = models_directory()

            # TODO: download model from a repository if it does not exist
            load_from = str(dir / f"{model}.bin")
            name = model

        if load_from == "":
            return None, {"error": "Model not found."}

        if not os.path.exists(load_from):
            return None, {"error": f"The model {load_from} does not exist."}

        if name not in llms:
            llms[name] = Llama(model_path=load_from)

    return name, None


def unload(model):
    """
    Unload a model.

    Remove a model from the list of loaded models. If the model is not loaded, this is a no-op.

    Args:
    model (str): The name of the model to unload.
    """
    llms.pop(model, None)


def generate(model, prompt):
    # auto load
    name, error = load(model)
    if error is not None:
        return error
    generated = llms[name](
        str(prompt),  # TODO: optimize prompt based on model
        max_tokens=4096,
        stop=["Q:", "\n"],
        stream=True,
    )
    for output in generated:
        yield json.dumps(output)


def models():
    dir = models_directory()
    all_files = os.listdir(dir)
    bin_files = [
        file.replace(".bin", "") for file in all_files if file.endswith(".bin")
    ]
    return bin_files


@app.route("/load", methods=["POST"])
def load_route_handler():
    data = request.get_json()
    model = data.get("model")
    if not model:
        return Response("Model is required", status=400)
    error = load(model)
    if error is not None:
        return error
    return Response(status=204)


@app.route("/unload", methods=["POST"])
def unload_route_handler():
    data = request.get_json()
    model = data.get("model")
    if not model:
        return Response("Model is required", status=400)
    unload(model)
    return Response(status=204)


@app.route("/generate", methods=["POST"])
def generate_route_handler():
    data = request.get_json()
    model = data.get("model")
    prompt = data.get("prompt")
    prompt = template(model, prompt)
    if not model:
        return Response("Model is required", status=400)
    if not prompt:
        return Response("Prompt is required", status=400)
    if not os.path.exists(f"{model}"):
        return {"error": "The model does not exist."}, 400
    return Response(
        stream_with_context(generate(model, prompt)), mimetype="text/event-stream"
    )


@app.route("/models", methods=["GET"])
def models_route_handler():
    bin_files = models()
    return Response(json.dumps(bin_files), mimetype="application/json")


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    # allows the script to respond to command line input when executed directly
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command()
@click.option("--port", default=7734, help="Port to run the server on")
@click.option("--debug", default=False, help="Enable debug mode")
def serve(port, debug):
    print("Serving on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)


@cli.command(name="load")
@click.argument("model")
@click.option("--file", default=False, help="Indicates that a file path is provided")
def load_cli(model, file):
    if file:
        error = load(path=model)
    else:
        error = load(model)
    if error is not None:
        print(error)
        return
    print("Model loaded")


@cli.command(name="generate")
@click.argument("model")
@click.option("--prompt", default="", help="The prompt for the model")
def generate_cli(model, prompt):
    if prompt == "":
        prompt = input("Prompt: ")
    output = ""
    prompt = template(model, prompt)
    for generated in generate(model, prompt):
        generated_json = json.loads(generated)
        text = generated_json["choices"][0]["text"]
        output += text
        print(f"\r{output}", end="", flush=True)


@cli.command(name="models")
def models_cli():
    print(models())


@cli.command(name="pull")
@click.argument("model")
def pull_cli(model):
    print("not implemented")


@cli.command(name="import")
@click.argument("model")
def import_cli(model):
    print("not implemented")


if __name__ == "__main__":
    cli()
