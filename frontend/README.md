Ollama Gateway Frontend

React + TypeScript web UI for the Ollama API Gateway.

## Development

```bash
npm install
npm run dev
```

## Build

```bash
npm run build
```

## Docker

Built as part of the main docker-compose stack.

## Features

- User login with API keys
- Admin login with username/password
- Session management
- Chat interface
- Admin panel for managing users and API keys

## Environment

The API base URL is `/api` which is proxied by nginx to the backend gateway.
