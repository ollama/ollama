import json
import os
from llama_cpp import Llama
from flask import Flask, Response, stream_with_context, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # enable CORS for all routes

# llms tracks which models are loaded
llms = {}


@app.route("/load", methods=["POST"])
def load():
    data = request.get_json()
    model = data.get("model")

    if not model:
        return Response("Model is required", status=400)
    if not os.path.exists(f"./models/{model}.bin"):
        return {"error": "The model does not exist."}, 400

    if model not in llms:
        llms[model] = Llama(model_path=f"./models/{model}.bin")

    return Response(status=204)


@app.route("/unload", methods=["POST"])
def unload():
    data = request.get_json()
    model = data.get("model")

    if not model:
        return Response("Model is required", status=400)
    if not os.path.exists(f"./models/{model}.bin"):
        return {"error": "The model does not exist."}, 400

    llms.pop(model, None)

    return Response(status=204)


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    model = data.get("model")
    prompt = data.get("prompt")

    if not model:
        return Response("Model is required", status=400)
    if not prompt:
        return Response("Prompt is required", status=400)
    if not os.path.exists(f"./models/{model}.bin"):
        return {"error": "The model does not exist."}, 400

    if model not in llms:
        # auto load
        llms[model] = Llama(model_path=f"./models/{model}.bin")

    def stream_response():
        stream = llms[model](
            str(prompt),  # TODO: optimize prompt based on model
            max_tokens=4096,
            stop=["Q:", "\n"],
            echo=True,
            stream=True,
        )
        for output in stream:
            yield json.dumps(output)

    return Response(
        stream_with_context(stream_response()), mimetype="text/event-stream"
    )

@app.route("/models", methods=["GET"])
def models():
    all_files = os.listdir("./models")
    bin_files = [file.replace(".bin", "") for file in all_files if file.endswith(".bin")]
    return Response(json.dumps(bin_files), mimetype="application/json")

if __name__ == "__main__":
    app.run(debug=True, threaded=True, port=5001)
    app.run()
