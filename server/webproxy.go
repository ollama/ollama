package server

import (
    "bytes"
    "fmt"
    "io"
    "net/http"
    "net/url"
    "strconv"
    "time"

    "github.com/gin-gonic/gin"
    "github.com/ollama/ollama/auth"
)

// signFunc is a variable to allow tests to override signing behavior.
var signFunc = auth.Sign

// httpClient is injectable for tests to capture outbound requests.
var httpClient = &http.Client{Timeout: 30 * time.Second}

// proxyToMain forwards the incoming request body to the main ollama server
// and sets an Authorization token if signing is available locally.
func (s *Server) proxyToMain(c *gin.Context, path string) {
    ctx := c.Request.Context()

    body, err := io.ReadAll(c.Request.Body)
    if err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": "failed to read request body"})
        return
    }

    now := strconv.FormatInt(time.Now().Unix(), 10)
    chal := fmt.Sprintf("%s,%s?ts=%s", http.MethodPost, path, now)

    token, err := signFunc(ctx, []byte(chal))
    if err != nil {
        // If signing fails, return an error so callers know the proxy couldn't
        // obtain a token. Clients may fallback to asking the user for a key.
        c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to sign request"})
        return
    }

    remoteURL := &url.URL{Scheme: "https", Host: "ollama.com", Path: path}
    q := remoteURL.Query()
    q.Set("ts", now)
    remoteURL.RawQuery = q.Encode()

    req, err := http.NewRequestWithContext(ctx, http.MethodPost, remoteURL.String(), bytes.NewReader(body))
    if err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to create outbound request"})
        return
    }

    // Preserve content type if provided
    if ct := c.GetHeader("Content-Type"); ct != "" {
        req.Header.Set("Content-Type", ct)
    } else {
        req.Header.Set("Content-Type", "application/json")
    }

    if token != "" {
        req.Header.Set("Authorization", token)
    }

    resp, err := httpClient.Do(req)
    if err != nil {
        c.JSON(http.StatusBadGateway, gin.H{"error": "failed to contact main server"})
        return
    }
    defer resp.Body.Close()

    // proxy status and content-type back to caller
    c.Status(resp.StatusCode)
    if ct := resp.Header.Get("Content-Type"); ct != "" {
        c.Header("Content-Type", ct)
    }

    // stream body
    if _, err := io.Copy(c.Writer, resp.Body); err != nil {
        // nothing much to do here, connection likely closed by client
        return
    }
}

func (s *Server) WebSearchHandler(c *gin.Context) {
    s.proxyToMain(c, "/api/web_search")
}

func (s *Server) WebFetchHandler(c *gin.Context) {
    s.proxyToMain(c, "/api/web_fetch")
}
