# Ollama Frontend

Production-grade web frontend for the Ollama Elite AI platform.

## 🚀 Features

- **Real-time Chat**: Stream responses from LLMs with full conversation history
- **OAuth Authentication**: Secure Google Sign-In via Firebase
- **Model Management**: Select and switch between different AI models
- **Responsive Design**: Mobile-first UI with Tailwind CSS
- **Type Safety**: Full TypeScript coverage
- **State Management**: Zustand for efficient state handling
- **Markdown Support**: Rich text rendering with syntax highlighting

## 📋 Prerequisites

- Node.js 18+ and npm 9+
- Firebase project with Google OAuth enabled
- Backend API running at `https://elevatediq.ai/ollama`

## 🛠️ Installation

```bash
# Install dependencies
npm install

# Copy environment template
cp .env.example .env.local

# Configure environment variables
nano .env.local
```

## 🔧 Configuration

Update `.env.local` with your credentials:

```bash
NEXT_PUBLIC_API_URL=https://elevatediq.ai/ollama
NEXT_PUBLIC_FIREBASE_API_KEY=your-firebase-api-key
NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=your-project.firebaseapp.com
NEXT_PUBLIC_FIREBASE_PROJECT_ID=your-project-id
# ... other Firebase config
```

## 🏃 Development

```bash
# Start development server (http://localhost:3000)
npm run dev

# Type checking
npm run type-check

# Linting
npm run lint

# Format code
npm run format
```

## 🏗️ Build

```bash
# Production build
npm run build

# Start production server
npm start
```

## 📁 Project Structure

```
frontend/
├── src/
│   ├── app/              # Next.js pages
│   │   ├── page.tsx      # Landing page
│   │   ├── chat/         # Chat interface
│   │   ├── layout.tsx    # Root layout
│   │   └── globals.css   # Global styles
│   ├── components/       # React components
│   │   ├── chat/         # Chat-specific components
│   │   ├── layout/       # Layout components
│   │   ├── providers/    # Context providers
│   │   └── ui/           # Reusable UI components
│   ├── lib/              # Core libraries
│   │   ├── firebase.ts   # Firebase config
│   │   └── api.ts        # API client
│   ├── store/            # Zustand stores
│   │   ├── authStore.ts  # Authentication state
│   │   └── chatStore.ts  # Chat state
│   └── types/            # TypeScript types
├── public/               # Static assets
├── package.json
├── tsconfig.json
├── next.config.js
└── tailwind.config.js
```

## 🔐 Security

- All API requests require Firebase authentication tokens
- HTTPS enforced in production
- Security headers configured via Next.js
- CORS restricted to `elevatediq.ai` domain
- CSP headers prevent XSS attacks

## 🧪 Testing

```bash
# Unit tests
npm test

# E2E tests
npm run test:e2e
```

## 🚀 Deployment

### Docker

```bash
# Build image
docker build -t ollama-frontend:latest .

# Run container
docker run -p 3000:3000 \
  -e NEXT_PUBLIC_API_URL=https://elevatediq.ai/ollama \
  ollama-frontend:latest
```

### GCP Cloud Run

```bash
# Deploy to Cloud Run
gcloud run deploy ollama-frontend \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars NEXT_PUBLIC_API_URL=https://elevatediq.ai/ollama
```

## 📊 Performance

- **First Contentful Paint**: <1.5s
- **Time to Interactive**: <3.0s
- **Lighthouse Score**: 95+
- **Bundle Size**: <200KB (gzipped)

## 🤝 Contributing

Follow the Elite Filesystem Standards:
- Maximum 5 levels deep directory structure
- One component per file
- Type hints on all functions
- Test coverage ≥90%

## 📄 License

MIT License - See LICENSE file for details

## 🔗 Links

- Backend API: https://elevatediq.ai/ollama
- Documentation: https://github.com/kushin77/ollama
- GCP Landing Zone: https://github.com/kushin77/GCP-landing-zone

---

**Version**: 1.0.0  
**Last Updated**: January 17, 2026  
**Maintained By**: kushin77/ollama engineering team
