# Ollama Gateway Frontend

Modern React + TypeScript chat interface for the Ollama Gateway API.

## Features

- **User Authentication** - API key-based login with persistent session storage
- **Multi-Session Management** - Create, switch, and delete chat sessions
- **Real-time Chat** - Send messages and receive AI responses
- **Session Isolation** - Each user's chats are completely isolated via Redis
- **Responsive Design** - Works on desktop and mobile
- **Dark Theme** - Easy on the eyes

## Technology Stack

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Fast build tool and dev server
- **CSS Modules** - Scoped styling

## Development

### Prerequisites

- Node.js 20+
- npm or yarn

### Local Development

```bash
cd frontend
npm install
npm run dev
```

The app will be available at `http://localhost:3000` with API proxy to the gateway.

### Build for Production

```bash
npm run build
```

Outputs to `dist/` directory, served by nginx in Docker.

## Docker Deployment

The frontend is automatically built and deployed via `docker-compose.yml`:

```yaml
frontend:
  build:
    context: ./frontend
  container_name: ollama-frontend
```

Access at `http://localhost:8080` (same port as the gateway, nginx routes traffic).

## Usage

### First Time

1. Open `http://localhost:8080`
2. Enter your API key (e.g., `user1-key` from `.env`)
3. Click "Sign In"

### Creating Chats

1. Click "+ New Chat" in sidebar
2. Type a message and press Send
3. AI responds with full conversation context

### Managing Sessions

- **Switch Chat**: Click any chat in the sidebar
- **Delete Chat**: Hover over chat, click × button
- **Logout**: Click Logout button at bottom of sidebar

## API Integration

The frontend calls these gateway endpoints:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/health` | GET | Health check |
| `/api/sessions` | POST | Create session |
| `/api/sessions` | GET | List sessions |
| `/api/sessions/{id}/messages` | GET | Get history |
| `/api/sessions/{id}/chat` | POST | Send message |
| `/api/sessions/{id}` | DELETE | Delete session |

All requests (except health) require `Authorization: Bearer <api-key>`.

## File Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── Login.tsx        # API key login
│   │   ├── Sidebar.tsx      # Session list
│   │   └── Chat.tsx         # Message display + input
│   ├── api.ts               # API client
│   ├── App.tsx              # Main component
│   └── main.tsx             # Entry point
├── Dockerfile               # Multi-stage build
├── vite.config.ts           # Vite config
└── package.json
```

## Security Notes

- API keys are stored in `localStorage` (browser-only, not sent to server except in headers)
- HTTPS recommended for production
- CORS enabled on gateway for dev mode
- In production, specify exact `allow_origins` in gateway CORS config

## Troubleshooting

**Login fails:**
- Check API key is correct (matches `.env` `API_KEYS`)
- Verify gateway is running: `docker compose ps`
- Check gateway health: `http://localhost:8080/api/health`

**Messages not sending:**
- Ensure session was created successfully
- Check browser console for errors
- Verify model is pulled: `docker exec -it ollama ollama list`

**Sessions not loading:**
- Check Redis is running: `docker compose ps`
- Verify `redis_connected: true` in health endpoint

## License

Same as parent Ollama project (MIT).
