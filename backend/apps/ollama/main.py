from flask import Flask, request, Response, jsonify
from flask_cors import CORS


import requests
import json


from apps.web.models.users import Users
from constants import ERROR_MESSAGES
from utils.utils import extract_token_from_auth_header
from config import OLLAMA_API_BASE_URL, WEBUI_AUTH

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
    print(path)

    # Get data from the original request
    data = request.get_data()
    headers = dict(request.headers)

    # Basic RBAC support
    if WEBUI_AUTH:
        if "Authorization" in headers:
            token = extract_token_from_auth_header(headers["Authorization"])
            user = Users.get_user_by_token(token)
            if user:
                # Only user and admin roles can access
                if user.role in ["user", "admin"]:
                    if path in ["pull", "delete", "push", "copy", "create"]:
                        # Only admin role can perform actions above
                        if user.role == "admin":
                            pass
                        else:
                            return (
                                jsonify({"detail": ERROR_MESSAGES.ACCESS_PROHIBITED}),
                                401,
                            )
                    else:
                        pass
                else:
                    return jsonify({"detail": ERROR_MESSAGES.ACCESS_PROHIBITED}), 401
            else:
                return jsonify({"detail": ERROR_MESSAGES.UNAUTHORIZED}), 401
        else:
            return jsonify({"detail": ERROR_MESSAGES.UNAUTHORIZED}), 401
    else:
        pass

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
