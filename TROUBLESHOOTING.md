# Ollama Web UI Troubleshooting Guide

## Connection Errors

If you encounter difficulties connecting to the Ollama server, please follow these steps to diagnose and resolve the issue:

**1. Verify Ollama Server Configuration**

Ensure that the Ollama server is properly configured to accept incoming connections from all origins. To do this, make sure the server is launched with the `OLLAMA_ORIGINS=*` environment variable, as shown in the following command:

```bash
OLLAMA_HOST=0.0.0.0 OLLAMA_ORIGINS=* ollama serve
```

This configuration allows Ollama to accept connections from any source.

**2. Check Ollama URL Format**

Ensure that the Ollama URL is correctly formatted in the application settings. Follow these steps:

- Go to "Settings" within the Ollama WebUI.
- Navigate to the "General" section.
- Verify that the Ollama URL is in the following format: `http://localhost:11434/api`.

It is crucial to include the `/api` at the end of the URL to ensure that the Ollama Web UI can communicate with the server.

By following these troubleshooting steps, you should be able to identify and resolve connection issues with your Ollama server configuration. If you require further assistance or have additional questions, please don't hesitate to reach out or refer to our documentation for comprehensive guidance.

## Running ollama-webui as a container on Apple Silicon Mac

If you are running Docker on a M{1..3} based Mac and have taken the steps to run an x86 container, add "--platform linux/amd64" to the docker run command to prevent a warning.

Example:

```bash
docker run -d -p 3000:8080 -e OLLAMA_API_BASE_URL=http://example.com:11434/api --name ollama-webui --restart always ghcr.io/ollama-webui/ollama-webui:main
```

Becomes

```
docker run --platform linux/amd64 -d -p 3000:8080 -e OLLAMA_API_BASE_URL=http://example.com:11434/api --name ollama-webui --restart always ghcr.io/ollama-webui/ollama-webui:main
```

## References
[Change Docker Desktop Settings on Mac](https://docs.docker.com/desktop/settings/mac/) Search for "x86" in that page.
[Run x86 (Intel) and ARM based images on Apple Silicon (M1) Macs?](https://forums.docker.com/t/run-x86-intel-and-arm-based-images-on-apple-silicon-m1-macs/117123)
