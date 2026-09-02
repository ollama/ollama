package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/ml"
)

func psResponse(t *testing.T, runner *runnerRef) (api.ProcessResponse, string) {
	t.Helper()

	s := &Server{sched: &Scheduler{loaded: map[string]*runnerRef{"m": runner}}}

	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodGet, "/api/ps", nil)

	s.PsHandler(c)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var got api.ProcessResponse
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatalf("decoding: %v (body %q)", err, w.Body.String())
	}
	return got, w.Body.String()
}

// A model split across two cards names both, so a client can attribute residency
// per device by joining these ids against /api/info's supported_gpus.
func TestPsHandlerReportsDevicesForSplitModel(t *testing.T) {
	got, _ := psResponse(t, &runnerRef{
		model:     &Model{ShortName: "llama3:70b"},
		gpus:      []ml.DeviceID{{ID: "0", Library: "CUDA"}, {ID: "1", Library: "CUDA"}},
		totalSize: 42 * format.GigaByte,
		vramSize:  40 * format.GigaByte,
		expiresAt: time.Now().Add(5 * time.Minute),
	})

	if len(got.Models) != 1 {
		t.Fatalf("expected 1 model, got %d", len(got.Models))
	}

	gpus := got.Models[0].GPUs
	if len(gpus) != 2 {
		t.Fatalf("expected both devices, got %d: %+v", len(gpus), gpus)
	}
	for i, want := range []string{"0", "1"} {
		if gpus[i].ID != want {
			t.Errorf("gpu %d: id %q, want %q", i, gpus[i].ID, want)
		}
		if gpus[i].Runner != "CUDA" {
			t.Errorf("gpu %d: runner %q, want CUDA", i, gpus[i].Runner)
		}
	}

	// SizeVRAM stays the total across devices; it is not divided between them.
	if got.Models[0].SizeVRAM != int64(40*format.GigaByte) {
		t.Errorf("size_vram: got %d, want %d", got.Models[0].SizeVRAM, 40*format.GigaByte)
	}
}

// A CPU-resident model has no devices, and the field is omitted rather than
// serialised as null.
func TestPsHandlerOmitsDevicesOnCPU(t *testing.T) {
	got, raw := psResponse(t, &runnerRef{
		model:     &Model{ShortName: "smol:1b"},
		totalSize: 2 * format.GigaByte,
		expiresAt: time.Now().Add(5 * time.Minute),
	})

	if len(got.Models[0].GPUs) != 0 {
		t.Errorf("expected no devices, got %+v", got.Models[0].GPUs)
	}
	if strings.Contains(raw, "\"gpus\"") {
		t.Errorf("gpus should be omitted for a CPU-resident model, got %s", raw)
	}
}
