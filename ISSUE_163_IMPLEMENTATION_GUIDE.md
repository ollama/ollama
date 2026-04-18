# Issue #163: Secret Scanning - Detect & Redact Secrets Implementation Guide

**Priority**: CRITICAL - Security
**Complexity**: High
**Effort**: 35 hours
**Status**: Ready for Implementation

## Problem Statement

Users may accidentally paste sensitive data (AWS keys, JWT tokens, GitHub PATs, passwords) into prompts. These secrets reach the model, are stored in logs, and could be exposed in model outputs if captured by third parties.

## Solution Overview

Implement a middleware layer that:
1. Scans all incoming prompts for secret patterns
2. Redacts/masks detected secrets before processing
3. Logs detection events for audit
4. Prevents secrets from reaching the model or logs

## Implementation

### Phase 1: Secret Pattern Detection

```go
// server/secrets/detector.go
package secrets

import (
    "regexp"
    "strings"
)

type SecretDetector struct {
    patterns map[string]*regexp.Regexp
}

func NewSecretDetector() *SecretDetector {
    return &SecretDetector{
        patterns: map[string]*regexp.Regexp{
            // AWS Keys: AKIA + 16 alphanumeric
            "aws_access_key": regexp.MustCompile(`AKIA[0-9A-Z]{16}`),

            // AWS Secret Key: 40 character base64-like
            "aws_secret_key": regexp.MustCompile(`(?i)aws_secret_access_key['\"]?\s*[:=]\s*['\"]?([A-Za-z0-9/+=]{40})`),

            // GitHub Personal Access Token
            "github_pat": regexp.MustCompile(`ghp_[A-Za-z0-9_]{36,255}`),

            // GitHub OAuth Token
            "github_oauth": regexp.MustCompile(`gho_[A-Za-z0-9_]{36,255}`),

            // GitHub App Token
            "github_app_token": regexp.MustCompile(`ghu_[A-Za-z0-9_]{36,255}`),

            // Private SSH Key (begin/end markers)
            "ssh_private_key": regexp.MustCompile(`-----BEGIN (?:RSA|DSA|EC|PGP)? ?PRIVATE KEY`),

            // Generic password patterns (e.g., password=value)
            "password": regexp.MustCompile(`(?i)password['\"]?\s*[:=]\s*['\"]?([^\s'\"]+)['\"]?`),

            // JWT tokens (consist of 3 base64url parts)
            "jwt_token": regexp.MustCompile(`eyJ[A-Za-z0-9_\-]+\.eyJ[A-Za-z0-9_\-]+\.([A-Za-z0-9_\-/+=]+)?`),

            // API Keys (generic: "api_key" = "...")
            "api_key": regexp.MustCompile(`(?i)api[_-]?key['\"]?\s*[:=]\s*['\"]?([A-Za-z0-9\-_]{20,})['\"]?`),

            // Database URLs with credentials
            "db_url": regexp.MustCompile(`(postgres|mysql|mongodb)://[^:]+:[^@]+@[^\s]+`),
        },
    }
}

// DetectSecrets scans text and returns locations of detected secrets
func (d *SecretDetector) DetectSecrets(text string) []Secret {
    var detected []Secret

    for secretType, pattern := range d.patterns {
        matches := pattern.FindAllStringIndex(text, -1)
        for _, match := range matches {
            detected = append(detected, Secret{
                Type:      secretType,
                Start:     match[0],
                End:       match[1],
                Value:     text[match[0]:match[1]],
                Severity:  d.getSeverity(secretType),
            })
        }
    }

    return detected
}

// RedactSecrets replaces detected secrets with [REDACTED]
func (d *SecretDetector) RedactSecrets(text string) (string, []Secret) {
    secrets := d.DetectSecrets(text)

    // Replace from end to start to preserve indices
    result := text
    for i := len(secrets) - 1; i >= 0; i-- {
        secret := secrets[i]
        result = result[:secret.Start] + "[REDACTED]" + result[secret.End:]
    }

    return result, secrets
}

func (d *SecretDetector) getSeverity(secretType string) string {
    switch secretType {
    case "aws_access_key", "aws_secret_key", "ssh_private_key":
        return "critical"
    case "github_pat", "api_key", "jwt_token":
        return "high"
    default:
        return "medium"
    }
}

// Secret represents a detected secret
type Secret struct {
    Type     string // Type of secret (aws_key, etc.)
    Start    int    // Starting position in text
    End      int    // Ending position in text
    Value    string // The actual secret value
    Severity string // Severity: critical, high, medium
}
```

### Phase 2: Request Middleware

```go
// server/middleware/secret_scanner.go
package middleware

import (
    "log"
    "net/http"

    "ollama/server/secrets"
)

type SecretScannerMiddleware struct {
    detector *secrets.SecretDetector
    logger   *log.Logger
}

func NewSecretScannerMiddleware(logger *log.Logger) *SecretScannerMiddleware {
    return &SecretScannerMiddleware{
        detector: secrets.NewSecretDetector(),
        logger:   logger,
    }
}

// ScanGenerateRequests middleware for /api/generate endpoint
func (m *SecretScannerMiddleware) ScanGenerateRequests(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        // Only scan POST requests to /api/generate
        if r.Method != http.MethodPost || r.URL.Path != "/api/generate" {
            next.ServeHTTP(w, r)
            return
        }

        // Parse request body
        var req struct {
            Prompt string `json:"prompt"`
            Model  string `json:"model"`
        }

        if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
            next.ServeHTTP(w, r)
            return
        }

        // Detect secrets
        detected := m.detector.DetectSecrets(req.Prompt)

        if len(detected) > 0 {
            // Log security event
            m.logger.Printf("SECURITY: Secrets detected in prompt from %s", r.RemoteAddr)
            for _, secret := range detected {
                m.logger.Printf("  - %s (severity: %s)", secret.Type, secret.Severity)
            }

            // Redact secrets before processing
            req.Prompt, _ = m.detector.RedactSecrets(req.Prompt)

            // Optionally: reject request if critical secrets found
            for _, secret := range detected {
                if secret.Severity == "critical" {
                    http.Error(w, "Request contains sensitive information", http.StatusBadRequest)
                    return
                }
            }
        }

        // Continue with processed request
        newBody, _ := json.Marshal(req)
        r.Body = io.NopCloser(bytes.NewReader(newBody))

        next.ServeHTTP(w, r)
    })
}

// ScanChatRequests middleware for /v1/chat/completions endpoint
func (m *SecretScannerMiddleware) ScanChatRequests(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        if r.Method != http.MethodPost {
            next.ServeHTTP(w, r)
            return
        }

        var req struct {
            Messages []struct {
                Content string `json:"content"`
            } `json:"messages"`
        }

        json.NewDecoder(r.Body).Decode(&req)

        // Scan all messages
        for i, msg := range req.Messages {
            if detected := m.detector.DetectSecrets(msg.Content); len(detected) > 0 {
                m.logger.Printf("SECURITY: Secrets in chat message from %s", r.RemoteAddr)
                req.Messages[i].Content, _ = m.detector.RedactSecrets(msg.Content)
            }
        }

        // Update request
        newBody, _ := json.Marshal(req)
        r.Body = io.NopCloser(bytes.NewReader(newBody))

        next.ServeHTTP(w, r)
    })
}
```

### Phase 3: Configuration & Audit Logging

```go
// config/secrets.yaml
secrets:
  scanning:
    enabled: true

    # Actions to take when secrets detected
    actions:
      redact: true                    # Redact before processing
      block_critical: true            # Block if critical secret found
      log_event: true                 # Log security event

    # Secret patterns (can be customized per environment)
    patterns:
      - type: "aws_key"
        enabled: true
      - type: "github_token"
        enabled: true
      - type: "jwt_token"
        enabled: true

    # Audit configuration
    audit:
      log_file: "/var/log/ollama/secrets_audit.log"
      log_redacted_secrets: false     # Never log actual secret values
      retention_days: 90              # Rotate logs
```

## Acceptance Criteria

- ✅ Detects AWS keys, GitHub tokens, JWTs, passwords, SSH keys
- ✅ Redacts detected secrets before model processing
- ✅ Logs security events to audit trail
- ✅ Blocks critical secrets if configured
- ✅ Supports custom secret patterns
- ✅ Zero false negatives on common secret types
- ✅ Performance impact <5ms per request

## Testing

```bash
# Test detection
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2",
    "prompt": "My AWS key is AKIA1234567890ABCDEF please help",
    "stream": false
  }'

# Expected: Secrets redacted in logs, audit entry created
# Expected: Prompt sent to model as "My AWS key is [REDACTED] please help"
```

## Deployment Checklist

- [ ] Implement secret pattern detection
- [ ] Integrate middleware into request pipeline
- [ ] Set up audit logging
- [ ] Test with sample secrets
- [ ] Performance testing
- [ ] Documentation for users
- [ ] Configuration guide

## Security Benefits

1. **Prevents Data Leakage**: Secrets don't reach logs or model
2. **Audit Trail**: Complete record of secret detection attempts
3. **Configurable**: Can customize patterns per environment
4. **Transparent**: Users see redacted text in responses

---

**Ready for Implementation**: Yes - detailed spec, middleware pattern clear, testing approach defined.
