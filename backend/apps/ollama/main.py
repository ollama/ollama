from flask import Flask, request, Response
from flask_cors import CORS


import requests
import json


from config import OLLAMA_API_BASE_URL

app = Flask(__name__)
CORS(
    app
)  # Enable Cross-Origin Resource Sharing (CORS) to allow requests from different domains

# Define the target server URL
TARGET_SERVER_URL = OLLAMA_API_BASE_URL


@app.route("/", defaults={"path": ""}, methods=["GET", "POST", "PUT", "DELETE"])
@app.route("/<path:path>", methods=["GET", "POST", "PUT", "DELETE"])
def proxy(path):
    # Combine the base URL of the target server with the requested path
    target_url = f"{TARGET_SERVER_URL}/{path}"
    print(target_url)

    # Get data from the original request
    data = request.get_data()
    headers = dict(request.headers)

    # Make a request to the target server
    target_response = requests.request(
        method=request.method,
        url=target_url,
        data=data,
        headers=headers,
        stream=True,  # Enable streaming for server-sent events
    )

    # Proxy the target server's response to the client
    def generate():
        for chunk in target_response.iter_content(chunk_size=8192):
            yield chunk

    response = Response(generate(), status=target_response.status_code)

    # Copy headers from the target server's response to the client's response
    for key, value in target_response.headers.items():
        response.headers[key] = value

    return response


if __name__ == "__main__":
    app.run(debug=True)
