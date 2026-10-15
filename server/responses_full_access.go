package server

import (
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/openai"
)

// responsesFullAccessMiddleware runs before cloud forwarding so the same tool
// contract reaches local models and cloud models through the local server.
func responsesFullAccessMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		body, err := readResponsesRequestBody(c)
		if err != nil {
			writeResponsesCompactionError(c, http.StatusBadRequest, "invalid_request_error", err.Error())
			return
		}
		normalized, changed, err := openai.NormalizeResponsesFullAccessTools(body)
		if err != nil {
			writeResponsesCompactionError(c, http.StatusBadRequest, "invalid_request_error", err.Error())
			return
		}
		if changed {
			resetResponsesRequestBody(c.Request, normalized)
		}
		c.Next()
	}
}
