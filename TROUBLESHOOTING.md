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
