package middleware

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cohere"
)

type CohereEmbedWriter struct {
	BaseWriter
	req cohere.EmbedRequest
}

func (w *CohereEmbedWriter) writeError(data []byte) (int, error) {
	var serr api.StatusError
	if err := json.Unmarshal(data, &serr); err != nil {
		serr.ErrorMessage = string(data)
	}

	w.ResponseWriter.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w.ResponseWriter).Encode(cohere.NewError(serr.Error())); err != nil {
		return 0, err
	}

	return len(data), nil
}

func (w *CohereEmbedWriter) writeResponse(data []byte) (int, error) {
	var embedResponse api.EmbedResponse
	if err := json.Unmarshal(data, &embedResponse); err != nil {
		return 0, err
	}

	resp, err := cohere.ToEmbedResponse(w.req, embedResponse)
	if err != nil {
		return 0, err
	}

	w.ResponseWriter.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w.ResponseWriter).Encode(resp); err != nil {
		return 0, err
	}

	return len(data), nil
}

func (w *CohereEmbedWriter) Write(data []byte) (int, error) {
	if w.ResponseWriter.Status() != http.StatusOK {
		return w.writeError(data)
	}

	return w.writeResponse(data)
}

func CohereEmbedMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		var req cohere.EmbedRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, cohere.NewError(err.Error()))
			return
		}

		if err := cohere.ValidateEmbeddingTypes(req.EmbeddingTypes); err != nil {
			c.AbortWithStatusJSON(http.StatusNotImplemented, cohere.NewError(err.Error()))
			return
		}

		embedReq, err := cohere.FromEmbedRequest(req)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, cohere.NewError(err.Error()))
			return
		}

		var b bytes.Buffer
		if err := json.NewEncoder(&b).Encode(embedReq); err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, cohere.NewError(err.Error()))
			return
		}

		c.Request.Body = io.NopCloser(&b)
		c.Writer = &CohereEmbedWriter{
			BaseWriter: BaseWriter{ResponseWriter: c.Writer},
			req:        req,
		}

		c.Next()
	}
}
