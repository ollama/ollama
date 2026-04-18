# Ollama Frontend Architecture

## Overview

Production-grade Next.js 14 frontend for the Ollama Elite AI platform, providing real-time chat, OAuth authentication, and model management capabilities.

## Technology Stack

### Core Framework
- **Next.js 14.2.0**: React framework with App Router, SSR, and API routes
- **React 18.3.0**: UI library with concurrent rendering
- **TypeScript 5.6.0**: Type safety and developer experience

### State Management
- **Zustand 5.0.2**: Lightweight state management (vs Redux complexity)
  - `authStore`: Authentication state (user, loading, error)
  - `chatStore`: Chat state (conversations, messages, streaming)

### Styling
- **Tailwind CSS 3.4.0**: Utility-first CSS framework
- **Custom Dark Theme**: Primary (indigo) and Dark (gray) color palettes
- **Responsive Design**: Mobile-first with breakpoints at sm/md/lg/xl

### Authentication
- **Firebase Client SDK 11.1.0**: OAuth integration
- **Google Sign-In**: Primary authentication method
- **Token Management**: Automatic refresh with API interceptors

### API Communication
- **Axios 1.7.9**: HTTP client with interceptors
- **Server-Sent Events**: Real-time streaming chat responses
- **Error Handling**: 401/429/500 with retry logic

### UI Components
- **Heroicons 2.2.0**: SVG icon library
- **React Markdown 9.0.1**: Markdown rendering with syntax highlighting
- **React Hot Toast 2.4.1**: Toast notifications

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         BROWSER CLIENT                          │
│                     (elevatediq.ai/ollama)                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
            ┌────────────▼────────────┐
            │   Next.js App Router    │
            │   (Server-Side)         │
            │  - SEO Optimization     │
            │  - Security Headers     │
            │  - API Proxy            │
            └────────────┬────────────┘
                         │
        ┌────────────────▼────────────────┐
        │      React Components Layer     │
        │                                 │
        │  ┌──────────────────────────┐   │
        │  │  Pages                   │   │
        │  │  - Landing (/)           │   │
        │  │  - Chat (/chat)          │   │
        │  └──────────────────────────┘   │
        │                                 │
        │  ┌──────────────────────────┐   │
        │  │  Components              │   │
        │  │  - ChatSidebar           │   │
        │  │  - ChatMessages          │   │
        │  │  - ChatInput             │   │
        │  │  - ModelSelector         │   │
        │  │  - Header                │   │
        │  └──────────────────────────┘   │
        └────────────┬────────────────────┘
                     │
        ┌────────────▼────────────────────┐
        │  Zustand State Management       │
        │                                 │
        │  authStore: { user, signIn,     │
        │              signOut, loading } │
        │  chatStore: { conversations,    │
        │              messages,          │
        │              isStreaming }      │
        └────────────┬────────────────────┘
                     │
    ┌────────────────▼────────────────────────┐
    │         Service Integration Layer       │
    │                                         │
    │  ┌────────────────┐  ┌───────────────┐  │
    │  │ Firebase Auth  │  │  API Client   │  │
    │  │ - Google Sign  │  │  - Auth Token │  │
    │  │ - Token Mgmt   │  │  - Streaming  │  │
    │  └────────────────┘  └───────────────┘  │
    └────────────────┬────────────────────────┘
                     │
        ┌────────────▼────────────┐
        │   External Services     │
        │                         │
        │  - Firebase Auth        │
        │  - Backend API          │
        │    (elevatediq.ai)      │
        └─────────────────────────┘
```

## Directory Structure (Elite 5-Level Standards)

```
frontend/                           # Level 1: Project root
├── src/                            # Level 2: Source code
│   ├── app/                        # Level 3: Next.js App Router pages
│   │   ├── page.tsx                # Level 4: Landing page
│   │   ├── chat/                   # Level 4: Chat page container
│   │   │   └── page.tsx            # Level 5: Chat interface
│   │   ├── layout.tsx              # Level 4: Root layout
│   │   └── globals.css             # Level 4: Global styles
│   ├── components/                 # Level 3: React components
│   │   ├── chat/                   # Level 4: Chat-specific
│   │   │   ├── ChatSidebar.tsx     # Level 5: Conversation list
│   │   │   ├── ChatMessages.tsx    # Level 5: Message display
│   │   │   ├── ChatInput.tsx       # Level 5: Message input
│   │   │   └── ModelSelector.tsx   # Level 5: Model dropdown
│   │   ├── layout/                 # Level 4: Layout components
│   │   │   └── Header.tsx          # Level 5: Top navigation
│   │   ├── providers/              # Level 4: Context providers
│   │   │   └── AuthProvider.tsx    # Level 5: Auth initialization
│   │   └── ui/                     # Level 4: Reusable UI
│   │       └── LoadingSpinner.tsx  # Level 5: Spinner component
│   ├── lib/                        # Level 3: Core libraries
│   │   ├── firebase.ts             # Level 4: Firebase config
│   │   └── api.ts                  # Level 4: API client
│   ├── store/                      # Level 3: State management
│   │   ├── authStore.ts            # Level 4: Auth state
│   │   └── chatStore.ts            # Level 4: Chat state
│   └── types/                      # Level 3: TypeScript types
├── public/                         # Level 2: Static assets
├── scripts/                        # Level 2: Automation scripts
│   ├── install.sh                  # Level 3: Setup script
│   └── deploy-gcp.sh               # Level 3: GCP deployment
├── package.json                    # Level 2: Dependencies
├── tsconfig.json                   # Level 2: TypeScript config
├── next.config.js                  # Level 2: Next.js config
├── tailwind.config.js              # Level 2: Tailwind config
└── Dockerfile                      # Level 2: Container config
```

## Data Flow

### Authentication Flow

```
1. User clicks "Sign in with Google" (Landing Page)
   ↓
2. Firebase SDK opens OAuth popup
   ↓
3. User authorizes with Google
   ↓
4. Firebase returns user object + ID token
   ↓
5. authStore updates: { user, loading: false }
   ↓
6. Router redirects to /chat
   ↓
7. All API requests include Firebase token in Authorization header
```

### Chat Message Flow

```
1. User types message in ChatInput
   ↓
2. ChatInput calls chatStore.sendMessage(message)
   ↓
3. chatStore adds user message to messages array
   ↓
4. API client calls POST /api/v1/chat with streaming
   ↓
5. Backend streams tokens via Server-Sent Events
   ↓
6. chatStore appends tokens to assistant message in real-time
   ↓
7. ChatMessages component re-renders with streaming content
   ↓
8. Stream completes, isStreaming set to false
```

## State Management

### Auth Store (Zustand)

```typescript
interface AuthState {
  user: User | null
  loading: boolean
  error: string | null
  
  // Actions
  signIn: () => Promise<void>
  signOut: () => Promise<void>
  initialize: () => void
  clearError: () => void
}
```

**Responsibilities**:
- Manage Firebase authentication state
- Persist user session across page reloads
- Provide sign-in/sign-out methods
- Subscribe to auth state changes

### Chat Store (Zustand)

```typescript
interface ChatState {
  conversations: Conversation[]
  currentConversation: string | null
  messages: Message[]
  isStreaming: boolean
  error: string | null
  selectedModel: string
  
  // Actions
  loadConversations: () => Promise<void>
  createConversation: (title: string) => Promise<void>
  selectConversation: (id: string) => void
  sendMessage: (content: string) => Promise<void>
  setSelectedModel: (model: string) => void
  clearError: () => void
}
```

**Responsibilities**:
- Manage conversation list and current conversation
- Handle message history and streaming messages
- Coordinate API calls for chat functionality
- Track selected model and streaming state

## Security

### Authentication
- Firebase OAuth with Google Sign-In (no password storage)
- ID tokens refreshed automatically before expiration
- Tokens sent in Authorization header: `Bearer <token>`

### Network Security
- HTTPS enforced via Next.js security headers
- Strict CSP (Content Security Policy) preventing XSS
- X-Frame-Options preventing clickjacking
- HSTS (HTTP Strict Transport Security) with 1-year max-age

### CORS Policy
- API proxy prevents CORS issues in development
- Production: Backend CORS restricted to `elevatediq.ai`

### Input Validation
- All user inputs sanitized before rendering
- Markdown rendering with XSS protection (react-markdown)
- No dangerouslySetInnerHTML usage

## Performance Optimizations

### Code Splitting
- Dynamic imports for large components
- Route-based splitting via Next.js App Router
- Lazy loading for markdown renderer and syntax highlighter

### Image Optimization
- Next.js Image component for automatic optimization
- WebP format with fallback to PNG/JPEG
- Responsive images with srcset

### Bundle Size
- Tree shaking for unused code elimination
- Minification in production builds
- Gzip compression on static assets
- Target: <200KB gzipped bundle

### Caching Strategy
- Static assets cached for 1 year (immutable)
- API responses cached with TTL (conversation list)
- Service Worker for offline support (future)

## Testing Strategy

### Unit Tests (Jest + React Testing Library)
- Component rendering tests
- State management logic tests
- API client mocking
- Coverage target: ≥90%

### Integration Tests
- Auth flow end-to-end
- Chat message flow with mocked API
- Navigation and routing

### E2E Tests (Playwright)
- Full user journeys (sign in → chat → sign out)
- Cross-browser compatibility (Chrome, Firefox, Safari)
- Mobile responsiveness

## Deployment

### Development
```bash
npm run dev
# http://localhost:3000
```

### Production Build
```bash
npm run build
npm start
# Optimized production server
```

### Docker
```bash
docker build -t ollama-frontend:latest .
docker run -p 3000:3000 ollama-frontend:latest
```

### GCP Cloud Run
```bash
./scripts/deploy-gcp.sh
# Deploys to Cloud Run with Landing Zone labels
```

## Monitoring

### Metrics to Track
- Page load times (FCP, LCP, TTI)
- API response times
- Auth success/failure rates
- Chat message latency
- Bundle size over time

### Error Tracking
- Sentry for frontend errors
- Firebase Crashlytics for mobile
- Structured logging to GCP Cloud Logging

### Alerting
- Auth failure rate >5%
- Page load time >3s
- API error rate >1%
- Bundle size increase >10%

## Future Enhancements

1. **Progressive Web App (PWA)**
   - Service Worker for offline support
   - Install prompt for mobile users
   - Push notifications

2. **Real-time Collaboration**
   - WebSocket for live typing indicators
   - Shared conversations across devices
   - Multi-user chat rooms

3. **Advanced Features**
   - Voice input/output
   - Image generation UI
   - Document upload and RAG
   - Model fine-tuning interface

4. **Accessibility**
   - WCAG 2.1 AA compliance
   - Keyboard navigation
   - Screen reader optimization
   - High contrast mode

## References

- [Next.js Documentation](https://nextjs.org/docs)
- [Firebase Auth](https://firebase.google.com/docs/auth)
- [Zustand State Management](https://github.com/pmndrs/zustand)
- [GCP Landing Zone](https://github.com/kushin77/GCP-landing-zone)

---

**Version**: 1.0.0  
**Last Updated**: January 17, 2026  
**Author**: kushin77/ollama engineering team
