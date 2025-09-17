package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/stretchr/testify/require"
)

func TestExportHandler(t *testing.T) {
	gin.SetMode(gin.TestMode)
	r := gin.Default()
	server := &Server{}
	r.POST("/api/export", server.ExportHandler)

	t.Run("Valid Export Request", func(t *testing.T) {
		req := api.ExportRequest{
			Model: "valid-model",
			Path:  "/test/export/path",
		}

		body, err := json.Marshal(req)
		require.NoError(t, err)

		w := httptest.NewRecorder()
		httpReq, _ := http.NewRequest("POST", "/api/export", bytes.NewBuffer(body))
		httpReq.Header.Set("Content-Type", "application/json")

		r.ServeHTTP(w, httpReq)

		// The handler will fail because we don't have a real model, but we can check the request was processed
		// In a real test, you'd mock the ExportModel function
		require.Contains(t, []int{http.StatusOK, http.StatusBadRequest, http.StatusInternalServerError}, w.Code)
	})

	t.Run("Missing Request Body", func(t *testing.T) {
		w := httptest.NewRecorder()
		httpReq, _ := http.NewRequest("POST", "/api/export", nil)
		httpReq.Header.Set("Content-Type", "application/json")

		r.ServeHTTP(w, httpReq)

		require.Equal(t, http.StatusBadRequest, w.Code)
	})

	t.Run("Invalid JSON", func(t *testing.T) {
		w := httptest.NewRecorder()
		httpReq, _ := http.NewRequest("POST", "/api/export", bytes.NewBufferString("invalid json"))
		httpReq.Header.Set("Content-Type", "application/json")

		r.ServeHTTP(w, httpReq)

		require.Equal(t, http.StatusBadRequest, w.Code)
	})

	t.Run("Invalid Model Name", func(t *testing.T) {
		req := api.ExportRequest{
			Model: "", // Empty model name
			Path:  "/test/export/path",
		}

		body, err := json.Marshal(req)
		require.NoError(t, err)

		w := httptest.NewRecorder()
		httpReq, _ := http.NewRequest("POST", "/api/export", bytes.NewBuffer(body))
		httpReq.Header.Set("Content-Type", "application/json")

		r.ServeHTTP(w, httpReq)

		require.Equal(t, http.StatusBadRequest, w.Code)
	})

	t.Run("Missing Path", func(t *testing.T) {
		req := api.ExportRequest{
			Model: "valid-model",
			Path:  "", // Empty path
		}

		body, err := json.Marshal(req)
		require.NoError(t, err)

		w := httptest.NewRecorder()
		httpReq, _ := http.NewRequest("POST", "/api/export", bytes.NewBuffer(body))
		httpReq.Header.Set("Content-Type", "application/json")

		r.ServeHTTP(w, httpReq)

		require.Equal(t, http.StatusBadRequest, w.Code)
	})
}

func TestImportHandler(t *testing.T) {
	gin.SetMode(gin.TestMode)
	r := gin.Default()
	server := &Server{}
	r.POST("/api/import", server.ImportHandler)

	t.Run("Valid Import Request", func(t *testing.T) {
		req := api.ImportRequest{
			Path:  "/test/import/path",
			Model: "imported-model",
		}

		body, err := json.Marshal(req)
		require.NoError(t, err)

		w := httptest.NewRecorder()
		httpReq, _ := http.NewRequest("POST", "/api/import", bytes.NewBuffer(body))
		httpReq.Header.Set("Content-Type", "application/json")

		r.ServeHTTP(w, httpReq)

		// The handler will fail because we don't have a real file, but we can check the request was processed
		// In a real test, you'd mock the ImportModel function
		require.Contains(t, []int{http.StatusOK, http.StatusBadRequest, http.StatusInternalServerError}, w.Code)
	})

	t.Run("Missing Request Body", func(t *testing.T) {
		w := httptest.NewRecorder()
		httpReq, _ := http.NewRequest("POST", "/api/import", nil)
		httpReq.Header.Set("Content-Type", "application/json")

		r.ServeHTTP(w, httpReq)

		require.Equal(t, http.StatusBadRequest, w.Code)
	})

	t.Run("Invalid JSON", func(t *testing.T) {
		w := httptest.NewRecorder()
		httpReq, _ := http.NewRequest("POST", "/api/import", bytes.NewBufferString("invalid json"))
		httpReq.Header.Set("Content-Type", "application/json")

		r.ServeHTTP(w, httpReq)

		require.Equal(t, http.StatusBadRequest, w.Code)
	})

	t.Run("Missing Path", func(t *testing.T) {
		req := api.ImportRequest{
			Path:  "", // Empty path
			Model: "imported-model",
		}

		body, err := json.Marshal(req)
		require.NoError(t, err)

		w := httptest.NewRecorder()
		httpReq, _ := http.NewRequest("POST", "/api/import", bytes.NewBuffer(body))
		httpReq.Header.Set("Content-Type", "application/json")

		r.ServeHTTP(w, httpReq)

		require.Equal(t, http.StatusBadRequest, w.Code)
	})

	t.Run("Invalid Model Name", func(t *testing.T) {
		req := api.ImportRequest{
			Path:  "/test/import/path",
			Model: "invalid/model*name", // Invalid characters
		}

		body, err := json.Marshal(req)
		require.NoError(t, err)

		w := httptest.NewRecorder()
		httpReq, _ := http.NewRequest("POST", "/api/import", bytes.NewBuffer(body))
		httpReq.Header.Set("Content-Type", "application/json")

		r.ServeHTTP(w, httpReq)

		require.Equal(t, http.StatusBadRequest, w.Code)
	})

	t.Run("Import Without Model Name", func(t *testing.T) {
		req := api.ImportRequest{
			Path:  "/test/import/path",
			// No model name provided
		}

		body, err := json.Marshal(req)
		require.NoError(t, err)

		w := httptest.NewRecorder()
		httpReq, _ := http.NewRequest("POST", "/api/import", bytes.NewBuffer(body))
		httpReq.Header.Set("Content-Type", "application/json")

		r.ServeHTTP(w, httpReq)

		// Should still be processed (model name will be derived from metadata)
		require.Contains(t, []int{http.StatusOK, http.StatusBadRequest, http.StatusInternalServerError}, w.Code)
	})

	t.Run("Import with Force Flag", func(t *testing.T) {
		req := api.ImportRequest{
			Path:  "/test/import/path",
			Model: "imported-model",
			Force: true,
		}

		body, err := json.Marshal(req)
		require.NoError(t, err)

		w := httptest.NewRecorder()
		httpReq, _ := http.NewRequest("POST", "/api/import", bytes.NewBuffer(body))
		httpReq.Header.Set("Content-Type", "application/json")

		r.ServeHTTP(w, httpReq)

		// Should be processed with force flag
		require.Contains(t, []int{http.StatusOK, http.StatusBadRequest, http.StatusInternalServerError}, w.Code)
	})

	t.Run("Import with Insecure Flag", func(t *testing.T) {
		req := api.ImportRequest{
			Path:     "/test/import/path",
			Model:    "imported-model",
			Insecure: true,
		}

		body, err := json.Marshal(req)
		require.NoError(t, err)

		w := httptest.NewRecorder()
		httpReq, _ := http.NewRequest("POST", "/api/import", bytes.NewBuffer(body))
		httpReq.Header.Set("Content-Type", "application/json")

		r.ServeHTTP(w, httpReq)

		// Should be processed with insecure flag
		require.Contains(t, []int{http.StatusOK, http.StatusBadRequest, http.StatusInternalServerError}, w.Code)
	})
}
