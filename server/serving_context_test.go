package server

import (
	"encoding/json"
	"strconv"
	"testing"

	"github.com/ollama/ollama/api"
	model "github.com/ollama/ollama/types/model"
)

func TestLocalServingContextLength(t *testing.T) {
	for _, tt := range []struct {
		name, env            string
		modelCtx, requestCtx any
		maximum              int
		want                 int
	}{
		{name: "unloaded explicit server context", env: "65536", maximum: 262144, want: 65536},
		{name: "model overrides server", env: "65536", modelCtx: float64(32768), maximum: 262144, want: 32768},
		{name: "request overrides model and server", env: "65536", modelCtx: float64(32768), requestCtx: float64(16384), maximum: 262144, want: 16384},
		{name: "clamped to architecture", env: "524288", maximum: 262144, want: 262144},
		{name: "automatic default is unknown before allocation", maximum: 262144},
		{name: "unknown architecture", env: "65536"},
		{name: "invalid option", env: "65536", requestCtx: "invalid", maximum: 262144},
		{name: "invalid model context", modelCtx: float64(-1), maximum: 262144},
	} {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("OLLAMA_CONTEXT_LENGTH", tt.env)
			m := &Model{Config: model.ConfigV2{ModelFormat: "gguf"}, Options: map[string]any{}, metadata: ggufMetadata{KV: map[string]any{"general.architecture": "test", "test.context_length": json.Number(strconv.Itoa(tt.maximum))}}}
			if tt.modelCtx != nil {
				m.Options["num_ctx"] = tt.modelCtx
			}
			opts := map[string]any{}
			if tt.requestCtx != nil {
				opts["num_ctx"] = tt.requestCtx
			}
			s := Server{defaultNumCtx: 262144}
			if got := s.localServingContextLength(m, opts); got != tt.want {
				t.Fatalf("serving context = %d, want %d", got, tt.want)
			}
			m.Config.RemoteHost = "https://ollama.com"
			if got := s.localServingContextLength(m, opts); got != 0 {
				t.Fatalf("remote context = %d, want unknown", got)
			}
		})
	}
}

func TestShowServingContextIsFreshAndDoesNotLoad(t *testing.T) {
	setTestHome(t, t.TempDir())
	createShowCacheModel(t, "serving-context", map[string]any{"test.context_length": uint32(262144)})
	s := Server{defaultNumCtx: 262144, modelCaches: &modelCaches{show: newModelShowCache()}}
	for _, tt := range []struct {
		env     string
		request map[string]any
		want    int
	}{
		{env: "65536", want: 65536},
		{env: "32768", want: 32768},
		{env: "65536", request: map[string]any{"num_ctx": 16384}, want: 16384},
		{want: 0},
	} {
		t.Setenv("OLLAMA_CONTEXT_LENGTH", tt.env)
		w := createRequest(t, s.ShowHandler, api.ShowRequest{Model: "serving-context", Options: tt.request})
		if w.Code != 200 {
			t.Fatalf("show: %d %s", w.Code, w.Body.String())
		}
		var resp api.ShowResponse
		if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
			t.Fatal(err)
		}
		if resp.ServingContextLength != tt.want {
			t.Fatalf("serving context = %d, want %d", resp.ServingContextLength, tt.want)
		}
	}
	if s.sched != nil {
		t.Fatal("metadata should not create a scheduler")
	}
}
