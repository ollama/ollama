# Issue #164: Output Sanitization - XSS & Injection Prevention Implementation Guide

**Priority**: CRITICAL - Security
**Complexity**: High
**Effort**: 30 hours
**Status**: Ready for Implementation

## Problem Statement

Model outputs may contain:
- **HTML/JavaScript**: XSS attacks if output used in web interfaces
- **Shell Commands**: Command injection if output piped to shell
- **SQL Fragments**: SQL injection if output used in queries
- **Template Syntax**: Template injection in downstream processing

Current system lacks output validation before:
- Streaming to web clients
- Logging to files
- Passing to other systems
- Displaying in CLI

## Solution Overview

Implement output sanitization layer that:
1. Validates output against injection patterns
2. Escapes dangerous content based on context
3. Optionally strips HTML/scripts
4. Logs suspicious patterns for analysis

## Implementation

### Phase 1: Output Sanitization Library

```go
// server/sanitize/sanitizer.go
package sanitize

import (
    "regexp"
    "strings"
    "html"
    "encoding/json"
)

type OutputSanitizer struct {
    config SanitizationConfig
}

type SanitizationConfig struct {
    StripHTML           bool   // Remove HTML tags completely
    EscapeHTML          bool   // HTML-escape special chars
    EscapeShell         bool   // Shell-escape special chars
    MaxOutputSize       int    // Truncate if exceeds size
    RejectSuspiciousPatterns bool
}

func NewOutputSanitizer(config SanitizationConfig) *OutputSanitizer {
    return &OutputSanitizer{config: config}
}

// SanitizeForWeb prepares output for web display
func (s *OutputSanitizer) SanitizeForWeb(output string) string {
    // Remove dangerous HTML
    if s.config.StripHTML {
        output = s.stripHTMLTags(output)
    } else if s.config.EscapeHTML {
        output = html.EscapeString(output)
    }

    // Check size limit
    if s.config.MaxOutputSize > 0 && len(output) > s.config.MaxOutputSize {
        output = output[:s.config.MaxOutputSize] + "\n[OUTPUT TRUNCATED]"
    }

    return output
}

// SanitizeForShell prepares output for shell execution
func (s *OutputSanitizer) SanitizeForShell(output string) (string, error) {
    // Check for dangerous shell metacharacters
    if s.detectShellInjection(output) {
        if s.config.RejectSuspiciousPatterns {
            return "", errors.New("output contains shell metacharacters")
        }
        output = s.escapeShellMetachars(output)
    }
    return output, nil
}

// SanitizeForSQL prepares output for SQL contexts
func (s *OutputSanitizer) SanitizeForSQL(output string) (string, error) {
    // Check for SQL injection patterns
    if s.detectSQLInjection(output) {
        if s.config.RejectSuspiciousPatterns {
            return "", errors.New("output contains SQL patterns")
        }
        output = s.escapeSQLString(output)
    }
    return output, nil
}

// SanitizeForJSON ensures valid JSON serialization
func (s *OutputSanitizer) SanitizeForJSON(output string) string {
    // Ensure valid JSON by escaping if needed
    _, err := json.Marshal(output)
    if err != nil {
        // Replace invalid UTF-8 sequences
        output = strings.ToValidUTF8(output, "?")
    }
    return output
}

// Injection detection functions

func (s *OutputSanitizer) detectHTMLInjection(output string) bool {
    dangerous := []string{
        "<script", "</script>", "javascript:", "on", "iframe", "embed",
    }
    lower := strings.ToLower(output)
    for _, pattern := range dangerous {
        if strings.Contains(lower, pattern) {
            return true
        }
    }
    return false
}

func (s *OutputSanitizer) detectShellInjection(output string) bool {
    dangerous := []string{
        "; rm -rf", "`", "$(", "|", "&", ">", "<", "&&", "||",
    }
    for _, pattern := range dangerous {
        if strings.Contains(output, pattern) {
            return true
        }
    }
    return false
}

func (s *OutputSanitizer) detectSQLInjection(output string) bool {
    dangerous := []regexp.Regexp{
        // Basic SQL injection patterns
        *regexp.MustCompile(`(?i)union.*select`),
        *regexp.MustCompile(`(?i)select.*from.*where`),
        *regexp.MustCompile(`(?i)'; drop table`),
        *regexp.MustCompile(`(?i)or\s+'1'\s*=\s*'1`),
    }

    for _, pattern := range dangerous {
        if pattern.MatchString(output) {
            return true
        }
    }
    return false
}

// Sanitization functions

func (s *OutputSanitizer) stripHTMLTags(output string) string {
    // Remove all HTML tags
    re := regexp.MustCompile(`<[^>]*>`)
    return re.ReplaceAllString(output, "")
}

func (s *OutputSanitizer) escapeShellMetachars(output string) string {
    replacements := map[string]string{
        "`":  "\\`",
        "$":  "\\$",
        "\\": "\\\\",
        "\"": "\\\"",
    }

    for old, new := range replacements {
        output = strings.ReplaceAll(output, old, new)
    }

    return output
}

func (s *OutputSanitizer) escapeSQLString(output string) string {
    // SQL escape: double single quotes
    return strings.ReplaceAll(output, "'", "''")
}
```

### Phase 2: Middleware Integration

```go
// server/middleware/output_sanitizer.go
package middleware

import (
    "encoding/json"
    "io"
    "log"
    "net/http"

    "ollama/server/sanitize"
)

type OutputSanitizerMiddleware struct {
    sanitizer *sanitize.OutputSanitizer
    logger    *log.Logger
}

func NewOutputSanitizerMiddleware(logger *log.Logger) *OutputSanitizerMiddleware {
    config := sanitize.SanitizationConfig{
        StripHTML:           true,
        EscapeHTML:          true,
        RejectSuspiciousPatterns: false,
        MaxOutputSize:       1024 * 1024, // 1MB
    }

    return &OutputSanitizerMiddleware{
        sanitizer: sanitize.NewOutputSanitizer(config),
        logger:    logger,
    }
}

// SanitizeGenerateResponse wraps response to sanitize model output
func (m *OutputSanitizerMiddleware) SanitizeGenerateResponse(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        // Create wrapped response writer
        wrapped := &responseWriter{
            ResponseWriter: w,
            sanitizer:      m.sanitizer,
            logger:         m.logger,
        }

        next.ServeHTTP(wrapped, r)
    })
}

// Response writer that sanitizes output
type responseWriter struct {
    http.ResponseWriter
    sanitizer *sanitize.OutputSanitizer
    logger    *log.Logger
}

func (w *responseWriter) Write(b []byte) (int, error) {
    // Parse response (assuming JSON format)
    var response struct {
        Response string `json:"response"`
        Done     bool   `json:"done"`
    }

    if err := json.Unmarshal(b, &response); err != nil {
        // Not JSON, pass through
        return w.ResponseWriter.Write(b)
    }

    // Sanitize response text
    sanitized := w.sanitizer.SanitizeForWeb(response.Response)

    // Log if sanitization changed output
    if sanitized != response.Response {
        w.logger.Printf("OUTPUT_SANITIZED: Removed %d suspicious characters",
            len(response.Response)-len(sanitized))
    }

    // Marshal and write
    response.Response = sanitized
    clean, _ := json.Marshal(response)
    return w.ResponseWriter.Write(clean)
}

// Context-specific sanitizers for CLI/API

// SanitizeForDisplay prepares output for terminal display
func (m *OutputSanitizerMiddleware) SanitizeForDisplay(raw string) string {
    // Strip HTML, keep shell-friendly
    return m.sanitizer.SanitizeForWeb(raw)
}

// SanitizeBeforeShellPipe prepares output for piping to shell
func (m *OutputSanitizerMiddleware) SanitizeBeforeShellPipe(raw string) (string, error) {
    return m.sanitizer.SanitizeForShell(raw)
}
```

### Phase 3: Content Security Policy Headers

```go
// server/middleware/csp.go
package middleware

import "net/http"

// ContentSecurityPolicyMiddleware adds CSP headers
func ContentSecurityPolicyMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        // Strict CSP to prevent XSS
        w.Header().Set("Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'")

        // Prevent clickjacking
        w.Header().Set("X-Frame-Options", "DENY")

        // Prevent MIME sniffing
        w.Header().Set("X-Content-Type-Options", "nosniff")

        // Enable XSS protection
        w.Header().Set("X-XSS-Protection", "1; mode=block")

        next.ServeHTTP(w, r)
    })
}
```

## Acceptance Criteria

- ✅ Detects HTML/JavaScript injection patterns
- ✅ Detects shell command injection patterns
- ✅ Detects SQL injection patterns
- ✅ Removes dangerous content OR escapes appropriately
- ✅ Adds security headers (CSP, X-Frame-Options, etc.)
- ✅ Handles streaming responses correctly
- ✅ ≤2ms latency overhead per output

## Testing

```bash
# Test XSS prevention
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2",
    "prompt": "Create HTML",
    "stream": false
  }'

# Expected: Any <script> tags in output are stripped/escaped

# Test shell injection prevention
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2",
    "prompt": "Create command",
    "stream": false
  }'

# Expected: Any shell metacharacters are escaped
```

## Deployment Checklist

- [ ] Implement output sanitization library
- [ ] Integrate middleware into response pipeline
- [ ] Add security headers
- [ ] Test with OWASP payloads
- [ ] Performance testing
- [ ] Documentation for integrators

## Security Benefits

1. **XSS Prevention**: Scripts cannot execute in web UIs
2. **Command Injection Prevention**: Shell-safe output
3. **SQL Injection Prevention**: Database query-safe output
4. **Defense in Depth**: Multiple validation layers

---

**Ready for Implementation**: Yes - clear injection patterns, multiple sanitization strategies.
