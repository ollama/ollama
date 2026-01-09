package server

import (
	"encoding/json"
	"net/http"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
)

func TestUsageHandler(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Run("empty server", func(t *testing.T) {
		s := Server{
			sched: &Scheduler{
				loaded: make(map[string]*runnerRef),
			},
		}

		w := createRequest(t, s.UsageHandler, nil)
		if w.Code != http.StatusOK {
			t.Fatalf("expected status code 200, actual %d", w.Code)
		}

		var resp api.UsageResponse
		if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
			t.Fatal(err)
		}

		// GPUs may or may not be present depending on system
		// Just verify we can decode the response
	})

	t.Run("response structure", func(t *testing.T) {
		s := Server{
			sched: &Scheduler{
				loaded: make(map[string]*runnerRef),
			},
		}

		w := createRequest(t, s.UsageHandler, nil)
		if w.Code != http.StatusOK {
			t.Fatalf("expected status code 200, actual %d", w.Code)
		}

		// Verify we can decode the response as valid JSON
		var resp map[string]any
		if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
			t.Fatal(err)
		}

		// The response should be a valid object (not null)
		if resp == nil {
			t.Error("expected non-nil response")
		}
	})
}
