# Ollama Gateway — Quickstart

Run Ollama behind a small API gateway with a web chat UI. Ollama stays private on the Docker network; only nginx is exposed on the host.

## What you get

| Service | Role |
|---------|------|
| **ollama** | Local LLM inference |
| **redis** | Session and chat history storage |
| **api** | FastAPI gateway (API key auth, sessions) |
| **frontend** | React chat UI |
| **nginx** | Public entrypoint at `http://localhost:8080` |

## Prerequisites

- **Docker Desktop** (Windows/macOS) or **Docker Engine + Compose** (Linux)
- **8 GB+ RAM** recommended for small models on CPU
- **NVIDIA GPU** (optional) — drivers + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

On Windows, use Docker Desktop with **WSL 2** backend.

---

## 1. Configure environment

From the repository root:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and set at least one API key:

```dotenv
HTTP_PORT=8080
OLLAMA_IMAGE=ollama/ollama:latest
OLLAMA_KEEP_ALIVE=5m
OLLAMA_NUM_PARALLEL=1
OLLAMA_MAX_LOADED_MODELS=1
SESSION_TTL_SECONDS=604800

# Required — comma-separated keys for users
API_KEYS=your-secret-key-here
```

You can add multiple keys (one per user or team):

```dotenv
API_KEYS=alice-key,bob-key,team-key
```

Each key maps to a stable `user_id`; chat sessions are isolated per key.

---

## 2. Start the stack

```powershell
docker compose up -d --build
```

First run builds the API gateway and frontend images. Wait until all containers are up:

```powershell
docker compose ps
```

Check health:

```powershell
curl.exe http://localhost:8080/api/health
```

Expected response includes `"status": "ok"` and `"redis_connected": true`.

---

## 3. Pull a model

Pull a small model suitable for CPU (recommended on low-end machines):

```powershell
docker exec -it ollama ollama pull llama3.2:1b
```

Other small options: `phi3:mini`, `qwen2.5:0.5b`.

---

## 4. Use the web UI

1. Open **http://localhost:8080**
2. Enter an API key from `.env` (e.g. `your-secret-key-here`)
3. Click **Sign In**
4. Click **+ New Chat**
5. Type a message and press **Send**

The UI stores your API key in the browser and loads session history from the gateway.

---

## 5. Optional — NVIDIA GPU

Verify Docker sees the GPU:

```powershell
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi
```

Start with the GPU override:

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

---

## API usage (curl / PowerShell)

All requests require:

```
Authorization: Bearer <your-api-key>
```

### List models

```powershell
curl.exe http://localhost:8080/api/v1/models `
  -H "Authorization: Bearer your-secret-key-here"
```

### Create a session

```powershell
$headers = @{
  Authorization  = "Bearer your-secret-key-here"
  "Content-Type" = "application/json"
}

$body = @{ title = "My chat"; model = "llama3.2:1b" } | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://localhost:8080/api/sessions" `
  -Method Post `
  -Headers $headers `
  -Body $body
```

### Send a message (server keeps context)

```powershell
$sessionId = "<session_id from create response>"

$body = @{ message = "What is Docker?" } | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://localhost:8080/api/sessions/$sessionId/chat" `
  -Method Post `
  -Headers $headers `
  -Body $body
```

Interactive API docs (when the gateway is running): **http://localhost:8080/api/docs**

---

## Useful commands

| Task | Command |
|------|---------|
| View logs | `docker compose logs -f` |
| API logs only | `docker compose logs -f api` |
| Restart after `.env` change | `docker compose up -d --build` |
| Stop stack | `docker compose down` |
| Stop and remove volumes (models + Redis) | `docker compose down -v` |

---

## Troubleshooting

### `Set API_KEYS in your .env file`

`API_KEYS` is required. Add it to `.env` and restart:

```powershell
docker compose up -d --build
```

### 502 Bad Gateway on login or `/api/health`

The API container may not be running:

```powershell
docker compose ps
docker compose logs api --tail 50
```

Restart the API:

```powershell
docker compose restart api
```

### Invalid API key

- Key must match one entry in `API_KEYS` in `.env` (no extra spaces)
- Restart API after changing `.env`: `docker compose restart api`

### Chat returns errors / no model

Pull the model inside the Ollama container:

```powershell
docker exec -it ollama ollama list
docker exec -it ollama ollama pull llama3.2:1b
```

### Slow responses on CPU

Use a smaller model and keep defaults in `.env`:

```dotenv
OLLAMA_KEEP_ALIVE=5m
OLLAMA_NUM_PARALLEL=1
OLLAMA_MAX_LOADED_MODELS=1
```

---

## Security notes

- Do not expose Ollama port `11434` directly to the internet.
- Replace example API keys before any non-local use.
- Put HTTPS in front of nginx (Cloudflare, Caddy, Traefik, etc.) before public deployment.
- Sessions expire after `SESSION_TTL_SECONDS` (default 7 days).

---

## Project layout

```
.
├── api-gateway/      # FastAPI gateway
├── frontend/         # React UI
├── nginx/            # Reverse proxy config
├── docker-compose.yml
├── docker-compose.gpu.yml
├── .env.example
└── QUICKSTART.md     # This file
```
