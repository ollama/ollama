package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/stretchr/testify/assert"
)

func TestStrictJSONValidation(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Run("GenerateHandler Unknown Field", func(t *testing.T) {
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)

		body := map[string]any{
			"model": "llama2",
			"prompt": "hello",
			"unknown_field": "should fail",
		}
		jsonBody, _ := json.Marshal(body)
		c.Request, _ = http.NewRequest("POST", "/api/generate", bytes.NewBuffer(jsonBody))

		s := &Server{}
		s.GenerateHandler(c)

		assert.Equal(t, http.StatusBadRequest, w.Code)
		assert.Contains(t, w.Body.String(), "unknown field")
	})

	t.Run("ChatHandler Unknown Field", func(t *testing.T) {
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)

		body := map[string]any{
			"model": "llama2",
			"messages": []api.Message{{Role: "user", Content: "hello"}},
			"unknown_field": "should fail",
		}
		jsonBody, _ := json.Marshal(body)
		c.Request, _ = http.NewRequest("POST", "/api/chat", bytes.NewBuffer(jsonBody))

		s := &Server{}
		s.ChatHandler(c)

		assert.Equal(t, http.StatusBadRequest, w.Code)
		assert.Contains(t, w.Body.String(), "unknown field")
	})
	
	t.Run("EmbedHandler Unknown Field", func(t *testing.T) {
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)

		body := map[string]any{
			"model": "all-minilm",
			"input": "test",
			"unknown_field": "should fail",
		}
		jsonBody, _ := json.Marshal(body)
		c.Request, _ = http.NewRequest("POST", "/api/embed", bytes.NewBuffer(jsonBody))

		s := &Server{}
		s.EmbedHandler(c)

		assert.Equal(t, http.StatusBadRequest, w.Code)
		assert.Contains(t, w.Body.String(), "unknown field")
	})
}
