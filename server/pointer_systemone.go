package server

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"os"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/decision"
)

const pointerSystemOneUnavailable = "model uses a pointer-head decision scorer. Ollama's letter-token System One path does not reproduce its probabilities. Start the model's pointer-head server and set OLLAMA_POINTER_RUNNER to its loopback URL."

// handlePointerSystemOne never falls back to letter-token scoring. Without a
// loopback runner it refuses. With one, it forwards the original request.
func (s *Server) handlePointerSystemOne(c *gin.Context, req decision.Request) {
	endpoint, err := decision.PointerRunnerEndpoint(os.Getenv("OLLAMA_POINTER_RUNNER"))
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	if endpoint == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": pointerSystemOneUnavailable})
		return
	}
	body, err := json.Marshal(req)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	upstream, err := http.NewRequestWithContext(c.Request.Context(), http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	upstream.Header.Set("Content-Type", "application/json")
	upstream.Header.Set("Accept", "application/json")
	resp, err := http.DefaultClient.Do(upstream)
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": "pointer-head runner failed"})
		return
	}
	defer resp.Body.Close()
	out, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": "pointer-head runner returned an unreadable response"})
		return
	}
	contentType := resp.Header.Get("Content-Type")
	if contentType == "" {
		contentType = "application/json"
	}
	c.Data(resp.StatusCode, contentType, out)
}
