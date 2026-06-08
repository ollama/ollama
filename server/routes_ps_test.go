package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/types/model"
)

func TestPsHandlerConcurrentAccess(t *testing.T) {
	gin.SetMode(gin.TestMode)

	ctx := t.Context()
	s := InitScheduler(ctx)

	srv := &Server{sched: s}

	// Populate the loaded map with a mock runner
	s.loadedMu.Lock()
	s.loaded["test-model"] = &runnerRef{
		model: &Model{
			ShortName: "test:latest",
			Config: model.ConfigV2{
				ModelFormat: "gguf",
				ModelFamily: "llama",
				ModelType:   "7B",
				FileType:    "Q4_0",
			},
		},
		llama:           &mockLlm{totalSize: 1000, vramSize: 500, vramByGPU: map[ml.DeviceID]uint64{}},
		totalSize:       1000,
		vramSize:        500,
		sessionDuration: 5 * time.Minute,
		expiresAt:       time.Now().Add(5 * time.Minute),
	}
	s.loadedMu.Unlock()

	// Run PsHandler and concurrent map writes in parallel.
	// Without proper locking on s.sched.loaded, this triggers a fatal
	// "concurrent map read and map write" detected by the race detector.
	var wg sync.WaitGroup
	const goroutines = 20

	// Half the goroutines call PsHandler (map read)
	for range goroutines / 2 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			w := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(w)
			c.Request, _ = http.NewRequest("GET", "/api/ps", nil)
			srv.PsHandler(c)
			require.Equal(t, http.StatusOK, w.Code)
		}()
	}

	// Half the goroutines mutate the loaded map (map write)
	for i := range goroutines / 2 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			key := "concurrent-model-" + string(rune('a'+i))
			s.loadedMu.Lock()
			s.loaded[key] = &runnerRef{
				model: &Model{
					ShortName: key,
					Config: model.ConfigV2{
						ModelFormat: "gguf",
						ModelFamily: "llama",
						ModelType:   "7B",
						FileType:    "Q4_0",
					},
				},
				totalSize:       100,
				sessionDuration: 5 * time.Minute,
				expiresAt:       time.Now().Add(5 * time.Minute),
			}
			s.loadedMu.Unlock()
		}()
	}

	wg.Wait()
}

func TestPsHandler(t *testing.T) {
	gin.SetMode(gin.TestMode)

	ctx := t.Context()
	s := InitScheduler(ctx)

	srv := &Server{sched: s}

	// Empty loaded map should return empty list
	w := createRequest(t, srv.PsHandler, nil)
	require.Equal(t, http.StatusOK, w.Code)

	var resp api.ProcessResponse
	require.NoError(t, json.NewDecoder(w.Body).Decode(&resp))
	require.Empty(t, resp.Models)

	// Add a model and verify it appears in the response
	expires := time.Now().Add(5 * time.Minute)
	s.loadedMu.Lock()
	s.loaded["model-a"] = &runnerRef{
		model: &Model{
			ShortName: "modelA:latest",
			Digest:    "sha256:abc123",
			Config: model.ConfigV2{
				ModelFormat:   "gguf",
				ModelFamily:   "llama",
				ModelFamilies: []string{"llama"},
				ModelType:     "7B",
				FileType:      "Q4_0",
			},
		},
		llama:           &mockLlm{totalSize: 2000, vramSize: 1500, vramByGPU: map[ml.DeviceID]uint64{}},
		totalSize:       1000,
		vramSize:        500,
		sessionDuration: 5 * time.Minute,
		expiresAt:       expires,
	}
	s.loadedMu.Unlock()

	w = createRequest(t, srv.PsHandler, nil)
	require.Equal(t, http.StatusOK, w.Code)

	require.NoError(t, json.NewDecoder(w.Body).Decode(&resp))
	require.Len(t, resp.Models, 1)
	require.Equal(t, "modelA:latest", resp.Models[0].Name)
	require.Equal(t, int64(2000), resp.Models[0].Size)
	require.Equal(t, int64(1500), resp.Models[0].SizeVRAM)
}
