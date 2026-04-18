# Gov-AI-Scout OAuth Authentication Implementation

Complete OAuth implementation reference from the kushin77/Gov-AI-Scout repository, including authentication middleware, decorators, utilities, and configuration.

---

## Table of Contents

1. [Backend Authentication Middleware](#backend-authentication-middleware)
2. [Frontend Authentication Context](#frontend-authentication-context)
3. [Next.js Middleware & Security](#nextjs-middleware--security)
4. [Backend Security Middleware](#backend-security-middleware)
5. [Environment Configuration](#environment-configuration)
6. [Usage Patterns](#usage-patterns)

---

## Backend Authentication Middleware

**File:** `backend/src/middleware/auth.ts`

### Overview
- **Provider:** Firebase Admin SDK
- **Token Type:** JWT (ID tokens)
- **Authentication Method:** Bearer token in Authorization header
- **Token Validation:** Includes revocation checking (`checkRevoked: true`)

### Code Implementation

```typescript
import { Request, Response, NextFunction } from 'express';
import * as admin from 'firebase-admin';
import { logger } from '../config';

// Initialize Firebase Admin SDK
if (!admin.apps.length) {
    const projectId = process.env.FIREBASE_PROJECT_ID || process.env.GCP_PROJECT_ID;

    admin.initializeApp({
        projectId,
        credential: process.env.NODE_ENV === 'test'
            ? admin.credential.cert({
                projectId: 'test-project',
                clientEmail: 'test@test.iam.gserviceaccount.com',
                privateKey: '-----BEGIN PRIVATE KEY-----\ntest\n-----END PRIVATE KEY-----\n',
            } as admin.ServiceAccount)
            : admin.credential.applicationDefault(),
    });
}

export interface AuthenticatedRequest extends Request {
    user?: {
        uid: string;
        email?: string;
        role?: string;
    };
}

/**
 * Middleware to verify Firebase ID tokens
 * Adds decoded user info to req.user
 *
 * Usage:
 *   router.get('/api/protected', authenticateToken, (req, res) => {
 *     const userId = req.user?.uid;
 *     // ...
 *   });
 */
export async function authenticateToken(
    req: AuthenticatedRequest,
    res: Response,
    next: NextFunction
): Promise<void> {
    try {
        // Extract token from Authorization header
        const authHeader = req.headers.authorization;

        if (!authHeader || !authHeader.startsWith('Bearer ')) {
            res.status(401).json({
                error: 'Unauthorized',
                message: 'Missing or invalid authorization header',
            });
            return;
        }

        const token = authHeader.substring(7); // Remove 'Bearer ' prefix

        // Verify token with Firebase Admin SDK
        // checkRevoked = true ensures revoked tokens are rejected
        const decodedToken = await admin.auth().verifyIdToken(token, true);

        // Attach user info to request
        req.user = {
            uid: decodedToken.uid,
            email: decodedToken.email,
            role: decodedToken.role as string | undefined,
        };

        logger.debug({ uid: req.user.uid, path: req.path }, 'User authenticated');

        next();
    } catch (error) {
        logger.warn({ error, path: req.path }, 'Authentication failed');

        if ((error as any).code === 'auth/id-token-revoked') {
            res.status(401).json({
                error: 'Unauthorized',
                message: 'Token has been revoked. Please sign in again.',
            });
            return;
        }

        if ((error as any).code === 'auth/id-token-expired') {
            res.status(401).json({
                error: 'Unauthorized',
                message: 'Token has expired. Please sign in again.',
            });
            return;
        }

        res.status(401).json({
            error: 'Unauthorized',
            message: 'Invalid authentication token',
        });
    }
}

/**
 * Middleware to check if user has required role
 *
 * Usage:
 *   router.delete('/api/users/:id',
 *     authenticateToken,
 *     requireRole('admin'),
 *     deleteUserHandler
 *   );
 */
export function requireRole(...allowedRoles: string[]) {
    return (req: AuthenticatedRequest, res: Response, next: NextFunction): void => {
        if (!req.user) {
            res.status(401).json({
                error: 'Unauthorized',
                message: 'Authentication required',
            });
            return;
        }

        const userRole = req.user.role || 'user';

        if (!allowedRoles.includes(userRole)) {
            logger.warn({
                uid: req.user.uid,
                role: userRole,
                required: allowedRoles,
                path: req.path,
            }, 'Insufficient permissions');

            res.status(403).json({
                error: 'Forbidden',
                message: 'Insufficient permissions',
            });
            return;
        }

        next();
    };
}

/**
 * Revoke all refresh tokens for a user (force re-authentication)
 *
 * Usage:
 *   await revokeUserTokens(userId); // User must re-login
 */
export async function revokeUserTokens(uid: string): Promise<void> {
    try {
        await admin.auth().revokeRefreshTokens(uid);
        logger.info({ uid }, 'User tokens revoked');
    } catch (error) {
        logger.error({ error, uid }, 'Error revoking user tokens');
        throw error;
    }
}

export default {
    authenticateToken,
    requireRole,
    revokeUserTokens,
};
```

### Key Features

- **Bearer Token Validation:** Expects `Authorization: Bearer <token>` header
- **Revocation Checking:** `checkRevoked: true` prevents using revoked tokens
- **Role-Based Access:** `requireRole()` middleware for authorization
- **Error Handling:** Specific error codes for revoked, expired, and invalid tokens
- **Structured Logging:** Integrated with logger for audit trails

### Integration Example

```typescript
import { Router } from 'express';
import { authenticateToken, requireRole } from '../middleware/auth';

const router = Router();

// Protected route - any authenticated user
router.get('/api/profile', authenticateToken, (req, res) => {
    res.json({ user: req.user });
});

// Admin-only route
router.delete('/api/users/:id',
    authenticateToken,
    requireRole('admin'),
    async (req, res) => {
        // Delete user...
        res.json({ success: true });
    }
);

export default router;
```

---

## Frontend Authentication Context

**File:** `admin/lib/auth-context.tsx`

### Overview
- **Provider:** Firebase Authentication (client-side)
- **OAuth Method:** Google OAuth 2.0 via `signInWithPopup`
- **Storage:** React Context API with state management
- **Authorization:** Root admin email verification

### Code Implementation

```typescript
'use client';

import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import {
  initializeApp,
  getApp,
} from 'firebase/app';
import {
  getAuth,
  onAuthStateChanged,
  signInWithPopup,
  GoogleAuthProvider,
  signOut,
  User,
} from 'firebase/auth';

interface AdminUser extends Omit<User, 'displayName' | 'email'> {
  role?: 'admin' | 'moderator' | 'user';
  isRootAdmin?: boolean;
  email: string | null;
  displayName: string | null;
}

interface AuthContextType {
  isAuthenticated: boolean;
  user: AdminUser | null;
  loading: boolean;
  loginWithGoogle: () => Promise<void>;
  logout: () => Promise<void>;
  isAuthorized: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

const ROOT_ADMIN_EMAIL = process.env.NEXT_PUBLIC_ROOT_ADMIN_EMAIL || 'akushnir@bioenergystrategies.com';

// Initialize Firebase
const firebaseConfig = {
  apiKey: process.env.NEXT_PUBLIC_FIREBASE_API_KEY,
  authDomain: process.env.NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN,
  projectId: process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID,
  storageBucket: process.env.NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: process.env.NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID,
  appId: process.env.NEXT_PUBLIC_FIREBASE_APP_ID,
};

function initializeFirebase() {
  // Log config for debugging (only in development)
  if (typeof window !== 'undefined' && !window.location.hostname.includes('run.app')) {
    console.log('[Firebase] Config check:', {
      hasApiKey: !!firebaseConfig.apiKey,
      hasAuthDomain: !!firebaseConfig.authDomain,
      hasProjectId: !!firebaseConfig.projectId,
      apiKeyPrefix: firebaseConfig.apiKey?.substring(0, 10)
    });
  }

  // Validate required fields
  if (!firebaseConfig.apiKey || !firebaseConfig.projectId) {
    console.error('[Firebase] Missing required config:', firebaseConfig);
    throw new Error('Firebase configuration is incomplete. Check environment variables.');
  }

  try {
    return getApp();
  } catch {
    return initializeApp(firebaseConfig);
  }
}

const googleProvider = new GoogleAuthProvider();

export function AuthProvider({ children }: { children: ReactNode }) {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [user, setUser] = useState<AdminUser | null>(null);
  const [loading, setLoading] = useState(true);
  const [isAuthorized, setIsAuthorized] = useState(false);

  useEffect(() => {
    const app = initializeFirebase();
    const auth = getAuth(app);

    // Set a timeout to prevent infinite loading (5 seconds)
    const timeout = setTimeout(() => {
      setLoading(false);
    }, 5000);

    // Listen for auth state changes
    const unsubscribe = onAuthStateChanged(auth, (firebaseUser: User | null) => {
      clearTimeout(timeout);

      if (firebaseUser) {
        const adminUser: AdminUser = {
          ...firebaseUser,
          isRootAdmin: firebaseUser.email === ROOT_ADMIN_EMAIL,
          role: firebaseUser.email === ROOT_ADMIN_EMAIL ? 'admin' : 'user',
        };
        setUser(adminUser);
        setIsAuthenticated(true);
        setIsAuthorized(true);
      } else {
        setUser(null);
        setIsAuthenticated(false);
        setIsAuthorized(false);
      }

      setLoading(false);
    });

    return () => {
      clearTimeout(timeout);
      unsubscribe();
    };
  }, []);

  const loginWithGoogle = async () => {
    const app = initializeFirebase();
    const auth = getAuth(app);
    try {
      // Attempt popup sign-in first
      const result = await signInWithPopup(auth, googleProvider);

      if (result.user?.email === ROOT_ADMIN_EMAIL) {
        const adminUser: AdminUser = {
          ...result.user,
          isRootAdmin: true,
          role: 'admin',
        };
        setUser(adminUser);
        setIsAuthenticated(true);
        setIsAuthorized(true);
        return;
      }

      // Non-root admin users are not authorized for this app
      await signOut(auth);
      throw new Error('Unauthorized: Only root admin can access this dashboard');
    } catch (error: any) {
      // Fallbacks for common popup issues
      const code = error?.code || '';
      const message = error?.message || '';
      console.error('Google login error:', code, message);

      // If popup is blocked or environment doesn't support it, try redirect
      if (
        code === 'auth/popup-blocked' ||
        code === 'auth/popup-closed-by-user' ||
        code === 'auth/operation-not-supported-in-this-environment'
      ) {
        try {
          // Use redirect as a safe fallback
          const { signInWithRedirect } = await import('firebase/auth');
          await signInWithRedirect(auth, googleProvider);
          return;
        } catch (redirectErr) {
          console.error('Redirect sign-in failed:', redirectErr);
        }
      }

      setIsAuthenticated(false);
      setIsAuthorized(false);
      setUser(null);
      throw error;
    }
  };

  const logout = async () => {
    try {
      const app = initializeFirebase();
      const auth = getAuth(app);
      await signOut(auth);
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      setIsAuthenticated(false);
      setIsAuthorized(false);
      setUser(null);
    }
  };

  return (
    <AuthContext.Provider
      value={{
        isAuthenticated,
        user,
        loading,
        loginWithGoogle,
        logout,
        isAuthorized,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

/**
 * Hook to use authentication context
 *
 * Usage:
 *   const { isAuthenticated, user, loginWithGoogle } = useAuth();
 */
export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
}
```

### Key Features

- **Google OAuth Pop-up:** Primary authentication method with redirect fallback
- **Root Admin Check:** Only specific email can access the admin dashboard
- **Loading State:** 5-second timeout to prevent infinite loading
- **Fire Auth Integration:** Full Firebase Authentication lifecycle management
- **Error Handling:** Graceful fallback from popup to redirect flow

### Usage in Components

```tsx
'use client';

import { useAuth } from '@/lib/auth-context';

export default function Dashboard() {
  const { isAuthenticated, user, loginWithGoogle, logout, loading } = useAuth();

  if (loading) {
    return <div>Loading authentication...</div>;
  }

  if (!isAuthenticated) {
    return (
      <button onClick={loginWithGoogle}>
        Sign in with Google
      </button>
    );
  }

  return (
    <div>
      <h1>Welcome, {user?.displayName}</h1>
      <button onClick={logout}>Logout</button>
    </div>
  );
}
```

---

## Next.js Middleware & Security

**File:** `admin/middleware.ts`

### Overview
- **Framework:** Next.js App Router
- **Rate Limiting:** Per-IP rate limiting (1000 req/min)
- **CSRF Protection:** Token-based CSRF validation
- **Security Headers:** OWASP recommended headers
- **Origin Validation:** Allowlist-based CORS checking

### Code Implementation

```typescript
import { NextRequest, NextResponse } from "next/server";
import crypto from "crypto";

// Rate limiting store (in production, use Redis)
const rateLimitStore = new Map<
    string,
    { count: number; resetTime: number }
>();

// CSRF token store
const csrfTokens = new Map<string, string>();

// Rate limiting function
function rateLimit(
    identifier: string,
    limit: number = 100,
    windowMs: number = 60000
): boolean {
    const now = Date.now();
    const record = rateLimitStore.get(identifier);

    if (!record || now > record.resetTime) {
        rateLimitStore.set(identifier, { count: 1, resetTime: now + windowMs });
        return true;
    }

    if (record.count < limit) {
        record.count++;
        return true;
    }

    return false;
}

// Validate request origin
function isValidOrigin(request: NextRequest): boolean {
    const origin = request.headers.get("origin") || "";
    const allowedOrigins = [
        "https://gov.elevatediq.ai",
        "https://gov.elevatediq.ai",
        "https://gov.elevatediq.ai",
    ];

    // Allow localhost for development
    if (process.env.NODE_ENV === "development") {
        allowedOrigins.push("http://localhost:3000", "http://localhost:3001");
    }

    return allowedOrigins.includes(origin) || origin === "";
}

// Sanitize headers
function sanitizeHeaders(request: NextRequest): NextRequest {
    const headers = new Headers(request.headers);

    // Remove sensitive headers
    headers.delete("x-forwarded-host");
    headers.delete("x-original-url");
    headers.delete("x-forwarded-proto");

    // Ensure secure headers
    headers.set("x-content-type-options", "nosniff");
    headers.set("x-frame-options", "DENY");
    headers.set("x-xss-protection", "1; mode=block");

    return new NextRequest(request, { headers });
}

export function middleware(request: NextRequest) {
    const pathname = request.nextUrl.pathname;
    const response = NextResponse.next();

    // Skip middleware for static assets
    if (
        pathname.startsWith("/_next") ||
        pathname.startsWith("/public") ||
        pathname.match(/\.(jpg|jpeg|png|gif|ico|svg|webp)$/)
    ) {
        return response;
    }

    // Get client IP
    const clientIp =
        request.headers.get("x-forwarded-for")?.split(",")[0] ||
        request.headers.get("x-real-ip") ||
        "unknown";

    // Apply rate limiting
    if (!rateLimit(clientIp, 1000, 60000)) {
        return new NextResponse("Too many requests", { status: 429 });
    }

    // Validate origin for API requests
    if (pathname.startsWith("/api/")) {
        const origin = request.headers.get("origin");
        if (origin && !isValidOrigin(request)) {
            return new NextResponse("Forbidden origin", { status: 403 });
        }

        // Validate CSRF token for non-GET requests
        if (request.method !== "GET" && request.method !== "HEAD") {
            const csrfToken = request.headers.get("x-csrf-token");
            if (!csrfToken || !csrfTokens.has(csrfToken)) {
                return new NextResponse("CSRF token validation failed", { status: 403 });
            }
        }
    }

    // Sanitize headers
    const sanitized = sanitizeHeaders(request);

    // Add security headers to response
    response.headers.set("X-Content-Type-Options", "nosniff");
    response.headers.set("X-Frame-Options", "DENY");
    response.headers.set("X-XSS-Protection", "1; mode=block");
    response.headers.set("Referrer-Policy", "no-referrer");
    response.headers.set("Permissions-Policy", "camera=(), microphone=()");

    return response;
}

export const config = {
    matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
```

### Key Features

- **Rate Limiting:** 1000 requests per 60 seconds per IP
- **CSRF Protection:** Token validation for state-changing requests (POST, PUT, DELETE)
- **Origin Validation:** Allowlist-based CORS checking for API requests
- **Security Headers:** XSS protection, frame options, content-type sniffing prevention
- **Header Sanitization:** Removes sensitive headers that could leak information

### Production Considerations

```typescript
// For production, replace Map with Redis:
import Redis from 'ioredis';

const redis = new Redis(process.env.REDIS_URL);

async function rateLimit(identifier: string, limit: number = 100, windowMs: number = 60000): Promise<boolean> {
    const key = `ratelimit:${identifier}`;
    const count = await redis.incr(key);

    if (count === 1) {
        await redis.expire(key, Math.ceil(windowMs / 1000));
    }

    return count <= limit;
}
```

---

## Backend Security Middleware

**File:** `shared/backend-security.ts`

### Overview
- **Framework:** Express.js
- **Security Headers:** Helmet middleware
- **CORS:** Configured with allowlist
- **Rate Limiting:** Tiered approach (auth, API, global)
- **Input Sanitization:** MongoDB injection prevention
- **CSRF:** Token-based validation
- **Event Logging:** Security event tracking

### Code Implementation

```typescript
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import mongoSanitize from 'express-mongo-sanitize';
import cors from 'cors';
import hpp from 'hpp';
import express from 'express';

/**
 * FORT KNOX BACKEND SECURITY MIDDLEWARE
 * Comprehensive protection against all attack vectors
 */

// ===== 1. HELMET - SECURITY HEADERS =====
export const securityHeaders = helmet({
    contentSecurityPolicy: {
        directives: {
            defaultSrc: ["'self'"],
            scriptSrc: ["'self'", "'unsafe-inline'", "'unsafe-eval'"],
            styleSrc: ["'self'", "'unsafe-inline'"],
            imgSrc: ["'self'", 'data:', 'https:'],
            fontSrc: ["'self'", 'https:', 'data:'],
            connectSrc: ["'self'", 'https:'],
            frameSrc: ["'none'"],
            objectSrc: ["'none'"],
            upgradeInsecureRequests: [],
        },
    },
    crossOriginEmbedderPolicy: true,
    crossOriginOpenerPolicy: true,
    crossOriginResourcePolicy: { policy: 'same-origin' },
    dnsPrefetchControl: true,
    frameguard: { action: 'deny' },
    hidePoweredBy: true,
    hsts: { maxAge: 63072000, includeSubDomains: true, preload: true },
    noSniff: true,
    referrerPolicy: { policy: 'no-referrer' },
    xssFilter: true,
    permittedCrossDomainPolicies: { permittedPolicies: 'none' },
});

// ===== 2. CORS - CROSS-ORIGIN RESOURCE SHARING =====
export const corsOptions = cors({
    origin: function (origin, callback) {
        const allowedOrigins = [
            'https://gov.elevatediq.ai',
            'https://gov.elevatediq.ai',
            'https://app.gov.elevatediq.ai',
            'https://app.gov.elevatediq.ai',
        ];

        // Allow requests without origin (same-origin requests)
        if (!origin) {
            return callback(null, true);
        }

        if (allowedOrigins.includes(origin)) {
            return callback(null, true);
        }

        // Development environment
        if (process.env.NODE_ENV === 'development') {
            if (/^http:\/\/localhost:\d+$/.test(origin)) {
                return callback(null, true);
            }
        }

        callback(new Error('CORS blocked: Invalid origin'));
    },
    credentials: true,
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization', 'X-CSRF-Token', 'X-Requested-With'],
    optionsSuccessStatus: 200,
    maxAge: 3600,
});

// ===== 3. RATE LIMITING =====
export const createRateLimiter = (
    windowMs: number = 15 * 60 * 1000, // 15 minutes
    max: number = 100
) => {
    return rateLimit({
        windowMs,
        max,
        message: 'Too many requests from this IP, please try again later.',
        standardHeaders: true,
        legacyHeaders: false,
        skip: (req) => {
            // Skip rate limiting for health checks
            return req.path === '/health' || req.path === '/api/health';
        },
    });
};

// Specific rate limiters
export const authLimiter = rateLimit({
    windowMs: 15 * 60 * 1000,
    max: 5, // Limit each IP to 5 requests per 15 minutes
    message: 'Too many login attempts, please try again later.',
    skipSuccessfulRequests: true, // Don't count successful requests
});

export const apiLimiter = rateLimit({
    windowMs: 60 * 1000, // 1 minute
    max: 100, // Limit each IP to 100 requests per minute
    message: 'API rate limit exceeded.',
});

// ===== 4. INPUT SANITIZATION =====
export const sanitizationMiddleware = [
    express.json({ limit: '10mb' }),
    express.urlencoded({ limit: '10mb', extended: true }),
    mongoSanitize(), // Prevent MongoDB injection
    mongoSanitize({ replaceWith: '_' }), // Replace prohibited characters
];

// ===== 5. HTTP PARAMETER POLLUTION PROTECTION =====
export const hppMiddleware = hpp({
    whitelist: [
        'sort',
        'fields',
        'page',
        'limit',
        'search',
        'filter',
        'expand',
        'include',
    ],
});

// ===== 6. REQUEST VALIDATION & SIZE LIMITS =====
export const requestValidationMiddleware = (
    req: express.Request,
    res: express.Response,
    next: express.NextFunction
) => {
    // Validate content-type for POST/PUT/PATCH
    if (['POST', 'PUT', 'PATCH'].includes(req.method)) {
        const contentType = req.get('content-type');
        if (contentType && !['application/json', 'application/x-www-form-urlencoded', 'multipart/form-data'].some(type => contentType.includes(type))) {
            return res.status(415).json({ error: 'Unsupported Media Type' });
        }
    }

    // Validate request size
    if (req.method !== 'GET' && !req.is('multipart/form-data')) {
        const maxSize = 10 * 1024 * 1024; // 10MB
        const contentLength = req.get('content-length');
        if (contentLength && parseInt(contentLength) > maxSize) {
            return res.status(413).json({ error: 'Request entity too large' });
        }
    }

    next();
};

// ===== 7. CSRF TOKEN VALIDATION =====
const csrfTokens = new Map<string, { token: string; timestamp: number }>();

export const generateCSRFToken = (sessionId: string): string => {
    const crypto = require('crypto');
    const token = crypto.randomBytes(32).toString('hex');
    csrfTokens.set(token, {
        token,
        timestamp: Date.now(),
    });

    // Clean up expired tokens (older than 1 hour)
    for (const [key, value] of csrfTokens.entries()) {
        if (Date.now() - value.timestamp > 3600000) {
            csrfTokens.delete(key);
        }
    }

    return token;
};

export const verifyCsrfToken = (
    req: express.Request,
    res: express.Response,
    next: express.NextFunction
) => {
    // Skip CSRF check for GET requests
    if (['GET', 'HEAD', 'OPTIONS'].includes(req.method)) {
        return next();
    }

    // Skip CSRF check for health endpoints
    if (req.path === '/health' || req.path === '/api/health') {
        return next();
    }

    const token = req.get('x-csrf-token') || req.body._csrf;

    if (!token || !csrfTokens.has(token)) {
        return res.status(403).json({ error: 'CSRF token validation failed' });
    }

    csrfTokens.delete(token);
    next();
};

// ===== 8. REQUEST ID & LOGGING =====
export const requestIdMiddleware = (
    req: express.Request,
    res: express.Response,
    next: express.NextFunction
) => {
    const crypto = require('crypto');
    const requestId = req.get('x-request-id') || crypto.randomUUID();
    req.id = requestId;
    res.set('X-Request-ID', requestId);
    next();
};

// ===== 9. SECURITY EVENT LOGGING =====
export const securityEventLogger = (
    req: express.Request,
    res: express.Response,
    next: express.NextFunction
) => {
    const start = Date.now();

    res.on('finish', () => {
        const duration = Date.now() - start;
        const statusCode = res.statusCode;

        // Log suspicious activity
        if (statusCode >= 400) {
            const event = {
                timestamp: new Date().toISOString(),
                requestId: req.id,
                method: req.method,
                path: req.path,
                statusCode,
                clientIp: req.ip,
                userAgent: req.get('user-agent'),
                duration,
            };

            if (statusCode === 403 || statusCode === 401) {
                console.warn('[SECURITY EVENT] Potential attack detected:', event);
            }
        }
    });

    next();
};

// ===== 10. APPLY ALL SECURITY MIDDLEWARE =====
export const applySecurityMiddleware = (app: express.Application) => {
    // Disable X-Powered-By header
    app.disable('x-powered-by');

    // Security headers
    app.use(securityHeaders);

    // CORS
    app.use(corsOptions);

    // Rate limiting
    app.use(createRateLimiter());

    // Request ID
    app.use(requestIdMiddleware);

    // Input sanitization
    app.use(...sanitizationMiddleware);

    // HTTP Parameter Pollution protection
    app.use(hppMiddleware);

    // Request validation
    app.use(requestValidationMiddleware);

    // Security event logging
    app.use(securityEventLogger);
};

export default {
    securityHeaders,
    corsOptions,
    createRateLimiter,
    authLimiter,
    apiLimiter,
    sanitizationMiddleware,
    hppMiddleware,
    requestValidationMiddleware,
    generateCSRFToken,
    verifyCsrfToken,
    requestIdMiddleware,
    securityEventLogger,
    applySecurityMiddleware,
};
```

### Integration with Express App

```typescript
import express from 'express';
import { applySecurityMiddleware, authLimiter } from './shared/backend-security';

const app = express();

// Apply all security middleware
applySecurityMiddleware(app);

// Use specific rate limiter for auth endpoints
app.post('/auth/login', authLimiter, async (req, res) => {
    // Login logic...
});

export default app;
```

---

## Environment Configuration

### Frontend Environment Variables

**File:** `.env.example`

```bash
# Gov-AI-Scout Environment Configuration (GCP Cloud)
# ⚠️ These are managed via Cloud Run environment variables and Secret Manager
# This file is for REFERENCE ONLY - do not use for local development

# Project Configuration
NODE_ENV=production
GCP_PROJECT_ID=govai-scout
GCP_REGION=us-central1

# Firestore
FIRESTORE_PROJECT_ID=govai-scout

# Firebase (via Secret Manager in production)
# FIREBASE_API_KEY=<from Secret Manager>
# FIREBASE_AUTH_DOMAIN=<from Secret Manager>
# FIREBASE_PROJECT_ID=govai-scout
# FIREBASE_STORAGE_BUCKET=<from Secret Manager>
# FIREBASE_MESSAGING_SENDER_ID=<from Secret Manager>
# FIREBASE_APP_ID=<from Secret Manager>

# API Configuration (Cloud Run URLs)
NEXT_PUBLIC_API_URL=https://api.gov.elevatediq.ai
NEXT_PUBLIC_APP_URL=https://gov.elevatediq.ai

# Admin Dashboard
NEXT_PUBLIC_ROOT_ADMIN_EMAIL=akushnir@bioenergystrategies.com

# Analytics (optional)
# NEXT_PUBLIC_GA_ID=<from Secret Manager>
# NEXT_PUBLIC_MIXPANEL_TOKEN=<from Secret Manager>

# Backend API
PORT=8080
LOG_LEVEL=info

# External Services
# Reference Ollama via: https://github.com/kushin77/ollama
# OLLAMA_API_URL=<configured via Cloud Run env>
```

### Backend Environment Variables

```bash
# Firebase/GCP
FIREBASE_PROJECT_ID=govai-scout
FIREBASE_API_KEY=<from Secret Manager>
GCP_PROJECT_ID=govai-scout

# Node Environment
NODE_ENV=production
LOG_LEVEL=info
PORT=8080

# Database
DATABASE_URL=<from Secret Manager>
REDIS_URL=<from Secret Manager>

# Security
JWT_SECRET=<from Secret Manager>
CSRF_SECRET=<from Secret Manager>
```

---

## Usage Patterns

### 1. Protecting API Routes

```typescript
// Express backend
import { Router } from 'express';
import { authenticateToken, requireRole } from './middleware/auth';

const router = Router();

// Public endpoint
router.get('/api/public', (req, res) => {
    res.json({ message: 'Public data' });
});

// Protected endpoint (any authenticated user)
router.get('/api/protected', authenticateToken, (req, res) => {
    res.json({
        message: 'Protected data',
        userId: req.user?.uid
    });
});

// Admin-only endpoint
router.delete('/api/users/:id',
    authenticateToken,
    requireRole('admin'),
    async (req, res) => {
        // Delete user logic...
        res.json({ success: true });
    }
);

export default router;
```

### 2. Using Auth in React Components

```tsx
'use client';

import { useAuth } from '@/lib/auth-context';
import { useRouter } from 'next/navigation';
import { useEffect } from 'react';

export default function ProtectedPage() {
  const { isAuthenticated, isAuthorized, user, loading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!loading && !isAuthorized) {
      router.replace('/login');
    }
  }, [loading, isAuthorized, router]);

  if (loading) return <div>Loading...</div>;
  if (!isAuthorized) return <div>Not authorized</div>;

  return (
    <div>
      <h1>Welcome, {user?.displayName}</h1>
      <p>Email: {user?.email}</p>
    </div>
  );
}
```

### 3. Making Authenticated API Requests

```typescript
// Frontend - Getting Firebase token
import { getAuth } from 'firebase/auth';

async function fetchProtectedAPI(endpoint: string) {
  const auth = getAuth();
  const user = auth.currentUser;

  if (!user) {
    throw new Error('User not authenticated');
  }

  const token = await user.getIdToken();

  const response = await fetch(endpoint, {
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
  });

  return response.json();
}

// Usage
const data = await fetchProtectedAPI('/api/protected');
```

### 4. Token Revocation (Logout)

```typescript
// Backend - Force user to re-authenticate
import { revokeUserTokens } from './middleware/auth';

router.post('/auth/logout', authenticateToken, async (req, res) => {
  try {
    const userId = req.user!.uid;
    await revokeUserTokens(userId); // Revoke all tokens
    res.json({ success: true, message: 'Logged out successfully' });
  } catch (error) {
    res.status(500).json({ error: 'Logout failed' });
  }
});
```

### 5. Adding Authentication to a New Route

```typescript
// Step 1: Apply authentication middleware
import { Router } from 'express';
import { authenticateToken, requireRole } from '../middleware/auth';

const router = Router();

// Step 2: Create handler with auth
router.post('/api/resource',
  authenticateToken,           // Verify token
  async (req, res) => {
    const userId = req.user?.uid;
    // Your logic here...
    res.json({ success: true });
  }
);

// Step 3: Register route in main app
app.use(router);
```

---

## Security Best Practices Summary

1. **Always use `checkRevoked: true`** when verifying Firebase tokens
2. **Store tokens in httpOnly cookies** for browser-based clients (prevent XSS)
3. **Implement proper role-based access control** with `requireRole()` middleware
4. **Use rate limiting** to prevent brute-force attacks
5. **Validate CSRF tokens** on state-changing requests
6. **Implement proper error handling** to avoid information leakage
7. **Log all authentication events** for audit trails
8. **Use HTTPS/TLS** for all communication (especially token transmission)
9. **Rotate API keys and secrets** regularly
10. **Monitor for suspicious authentication patterns** in logs

---

## References

- [Firebase Admin SDK Documentation](https://firebase.google.com/docs/admin/setup)
- [Express.js Security Best Practices](https://expressjs.com/en/advanced/best-practice-security.html)
- [OWASP Authentication Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html)
- [Firebase Security Rules](https://firebase.google.com/docs/rules)

---

**Repository:** https://github.com/kushin77/Gov-AI-Scout
**Last Updated:** January 13, 2026
