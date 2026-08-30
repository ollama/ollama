package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"runtime"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/version"
)

// infoTestServer builds a server whose discovery is the fake pair from sched_test.go
// (one 24GB Metal device, 32GB of system memory), so the reported values are
// deterministic and the test does not depend on the machine it runs on.
func infoTestServer(t *testing.T) *Server {
	t.Helper()
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	return &Server{
		sched: &Scheduler{
			getGpuFn:        getGpuFn,
			getSystemInfoFn: getSystemInfoFn,
		},
	}
}

func infoResponse(t *testing.T, s *Server) api.InfoResponse {
	t.Helper()

	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodGet, "/api/info", nil)

	s.InfoHandler(c)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", w.Code, w.Body.String())
	}

	var got api.InfoResponse
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatalf("decoding response: %v (body %q)", err, w.Body.String())
	}
	return got
}

// createInfoTestModel writes a model into OLLAMA_MODELS. It mirrors the model
// creation inside TestRoutes, which lives in a closure and cannot be reused here.
func createInfoTestModel(t *testing.T, name, digest string) {
	t.Helper()

	fn := func(resp api.ProgressResponse) { t.Logf("Status: %s", resp.Status) }

	baseLayers, err := ggufLayers(digest, "test.gguf", fn)
	if err != nil {
		t.Fatalf("failed to build layers: %v", err)
	}

	config := &model.ConfigV2{
		OS:           "linux",
		Architecture: "amd64",
	}

	r := api.CreateRequest{Name: name, Files: map[string]string{"test.gguf": digest}}
	if err := createModel(r, model.ParseName(name), baseLayers, config, fn); err != nil {
		t.Fatalf("failed to create model %s: %v", name, err)
	}
}

func TestInfoHandlerReportsDiscoveredDevices(t *testing.T) {
	got := infoResponse(t, infoTestServer(t))

	if len(got.ComputeInfo.SupportedGPUs) != 1 {
		t.Fatalf("expected 1 GPU, got %d", len(got.ComputeInfo.SupportedGPUs))
	}

	gpu := got.ComputeInfo.SupportedGPUs[0]
	if gpu.TotalMemory != 24*format.GigaByte {
		t.Errorf("total memory: got %d, want %d", gpu.TotalMemory, 24*format.GigaByte)
	}
	if gpu.FreeMemory != 12*format.GigaByte {
		t.Errorf("free memory: got %d, want %d", gpu.FreeMemory, 12*format.GigaByte)
	}
	if gpu.Runner != "Metal" {
		t.Errorf("runner: got %q, want %q", gpu.Runner, "Metal")
	}
}

func TestInfoHandlerReportsSystemCompute(t *testing.T) {
	got := infoResponse(t, infoTestServer(t))

	sys := got.ComputeInfo.SystemCompute
	if sys.TotalMemory != 32*format.GigaByte {
		t.Errorf("system total memory: got %d, want %d", sys.TotalMemory, 32*format.GigaByte)
	}
	if sys.FreeMemory != 26*format.GigaByte {
		t.Errorf("system free memory: got %d, want %d", sys.FreeMemory, 26*format.GigaByte)
	}
	if sys.CPUCores != runtime.NumCPU() {
		t.Errorf("cpu cores: got %d, want %d", sys.CPUCores, runtime.NumCPU())
	}
	if got.Version != version.Version {
		t.Errorf("version: got %q, want %q", got.Version, version.Version)
	}
}

// With nothing loaded the memory counters are zero rather than absent, so a client
// can render "0 B in use" without special-casing an idle server.
func TestInfoHandlerIdleServerReportsZeroUsage(t *testing.T) {
	got := infoResponse(t, infoTestServer(t))

	if got.Models.Running != 0 {
		t.Errorf("running: got %d, want 0", got.Models.Running)
	}
	if got.Models.VRAMUsed != 0 {
		t.Errorf("vram used: got %d, want 0", got.Models.VRAMUsed)
	}
	if got.Models.Count != 0 {
		t.Errorf("model count: got %d, want 0 for an empty store", got.Models.Count)
	}
}

// A layer shared by several models occupies the store once, so filesystem usage
// counts each digest a single time: adding a second model built from the same
// file must not grow the reported total.
func TestInfoHandlerCountsSharedLayersOnce(t *testing.T) {
	s := infoTestServer(t)
	_, digest := createTestFile(t, "ollama-model")

	createInfoTestModel(t, "shared-a", digest)
	first := infoResponse(t, s)
	if first.Models.Count != 1 {
		t.Fatalf("model count: got %d, want 1", first.Models.Count)
	}
	if first.Models.FilesystemUsed == 0 {
		t.Fatal("filesystem used should be non-zero once a model exists")
	}

	// same source file → same layer digests, already counted
	createInfoTestModel(t, "shared-b", digest)
	second := infoResponse(t, s)

	if second.Models.Count != 2 {
		t.Fatalf("model count: got %d, want 2", second.Models.Count)
	}
	if second.Models.FilesystemUsed != first.Models.FilesystemUsed {
		t.Errorf("shared layers counted twice: got %d, want %d (unchanged)",
			second.Models.FilesystemUsed, first.Models.FilesystemUsed)
	}
}

// Metal reports no compute capability or driver version, so those fields are
// omitted rather than sent as a meaningless "0.0".
func TestInfoHandlerOmitsUnknownComputeAndDriver(t *testing.T) {
	got := infoResponse(t, infoTestServer(t))

	gpu := got.ComputeInfo.SupportedGPUs[0]
	if gpu.Compute != "" {
		t.Errorf("compute: got %q, want empty for a backend that doesn't report it", gpu.Compute)
	}
	if gpu.Driver != "" {
		t.Errorf("driver: got %q, want empty for a backend that doesn't report it", gpu.Driver)
	}
}

// ...but a backend that does report them (CUDA) still gets them through.
func TestInfoHandlerReportsKnownComputeAndDriver(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	s := &Server{
		sched: &Scheduler{
			getSystemInfoFn: getSystemInfoFn,
			getGpuFn: func(ctx context.Context, runners []ml.FilteredRunnerDiscovery) []ml.DeviceInfo {
				return []ml.DeviceInfo{{
					DeviceID:     ml.DeviceID{ID: "0", Library: "CUDA"},
					Name:         "NVIDIA GeForce RTX 3090",
					TotalMemory:  24 * format.GigaByte,
					FreeMemory:   20 * format.GigaByte,
					ComputeMajor: 8,
					ComputeMinor: 6,
					DriverMajor:  12,
					DriverMinor:  4,
				}}
			},
		},
	}

	gpu := infoResponse(t, s).ComputeInfo.SupportedGPUs[0]
	if gpu.Compute != "8.6" {
		t.Errorf("compute: got %q, want %q", gpu.Compute, "8.6")
	}
	if gpu.Driver != "12.4" {
		t.Errorf("driver: got %q, want %q", gpu.Driver, "12.4")
	}
	if gpu.Runner != "CUDA" {
		t.Errorf("runner: got %q, want %q", gpu.Runner, "CUDA")
	}
}
