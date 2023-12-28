# Ollama Web UI Troubleshooting Guide

## Ollama WebUI: Server Connection Error

If you're running ollama-webui and have chosen to install webui and ollama separately, you might encounter connection issues. This is often due to the docker container being unable to reach the Ollama server at 127.0.0.1:11434(host.docker.internal:11434). To resolve this, you can use the `--network=host` flag in the docker command. When done so port would be changed from 3000 to 8080, and the link would be: http://localhost:8080.

Here's an example of the command you should run:

```bash
docker run -d --network=host -v ollama-webui:/app/backend/data -e OLLAMA_API_BASE_URL=http://127.0.0.1:11434/api --name ollama-webui --restart always ghcr.io/ollama-webui/ollama-webui:main
```

## Connection Errors

Make sure you have the **latest version of Ollama** installed before proceeding with the installation. You can find the latest version of Ollama at [https://ollama.ai/](https://ollama.ai/).

If you encounter difficulties connecting to the Ollama server, please follow these steps to diagnose and resolve the issue:

**1. Check Ollama URL Format**

Ensure that the Ollama URL is correctly formatted in the application settings. Follow these steps:

- If your Ollama runs in a different host than Web UI make sure Ollama host address is provided when running Web UI container via `OLLAMA_API_BASE_URL` environment variable. [(e.g. OLLAMA_API_BASE_URL=http://192.168.1.1:11434/api)](https://github.com/ollama-webui/ollama-webui#accessing-external-ollama-on-a-different-server)
- Go to "Settings" within the Ollama WebUI.
- Navigate to the "General" section.
- Verify that the Ollama Server URL is set to: `/ollama/api`.

It is crucial to include the `/api` at the end of the URL to ensure that the Ollama Web UI can communicate with the server.

By following these troubleshooting steps, you should be able to identify and resolve connection issues with your Ollama server configuration. If you require further assistance or have additional questions, please don't hesitate to reach out or refer to our documentation for comprehensive guidance.
