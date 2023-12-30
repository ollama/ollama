# Ollama Web UI Troubleshooting Guide

## Understanding the Ollama WebUI Architecture

The Ollama WebUI system is designed to streamline interactions between the client (your browser) and the Ollama API. At the heart of this design is a backend reverse proxy, enhancing security and resolving CORS issues.

- **How it Works**: When you make a request (like `/ollama/api/tags`) from the Ollama WebUI, it doesn’t go directly to the Ollama API. Instead, it first reaches the Ollama WebUI backend. The backend then forwards this request to the Ollama API via the route you define in the `OLLAMA_API_BASE_URL` environment variable. For instance, a request to `/ollama/api/tags` in the WebUI is equivalent to `OLLAMA_API_BASE_URL/tags` in the backend.

- **Security Benefits**: This design prevents direct exposure of the Ollama API to the frontend, safeguarding against potential CORS (Cross-Origin Resource Sharing) issues and unauthorized access. Requiring authentication to access the Ollama API further enhances this security layer.

## Ollama WebUI: Server Connection Error

If you're experiencing connection issues, it’s often due to the WebUI docker container not being able to reach the Ollama server at 127.0.0.1:11434 (host.docker.internal:11434) inside the container . Use the `--network=host` flag in your docker command to resolve this. Note that the port changes from 3000 to 8080, resulting in the link: `http://localhost:8080`.

**Example Docker Command**:

```bash
docker run -d --network=host -v ollama-webui:/app/backend/data -e OLLAMA_API_BASE_URL=http://127.0.0.1:11434/api --name ollama-webui --restart always ghcr.io/ollama-webui/ollama-webui:main
```

### General Connection Errors

**Ensure Ollama Version is Up-to-Date**: Always start by checking that you have the latest version of Ollama. Visit [Ollama's official site](https://ollama.ai/) for the latest updates.

**Troubleshooting Steps**:

1. **Verify Ollama URL Format**:
   - When running the Web UI container, ensure the `OLLAMA_API_BASE_URL` is correctly set, including the `/api` suffix. (e.g., `http://192.168.1.1:11434/api` for different host setups).
   - In the Ollama WebUI, navigate to "Settings" > "General".
   - Confirm that the Ollama Server URL is correctly set to `/ollama/api`, including the `/api` suffix.

By following these enhanced troubleshooting steps, connection issues should be effectively resolved. For further assistance or queries, feel free to reach out to us on our community Discord.
