package server

import (
	"net/http"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/parser"
	"github.com/ollama/ollama/types/model"
)

func TestCreateCapabilities(t *testing.T) {
	gin.SetMode(gin.TestMode)
	for _, format := range []string{"gguf", "safetensors"} {
		t.Run(format, func(t *testing.T) {
			t.Setenv("OLLAMA_MODELS", t.TempDir())
			var s Server
			baseCaps := []string{"completion", "vision"}
			if format == "safetensors" {
				createSafetensorsTestModel(t, "base", model.ConfigV2{
					ModelFormat: format, Renderer: "qwen3.5", Capabilities: baseCaps,
				}, nil)
			} else {
				_, digest := createBinFile(t, map[string]any{"general.architecture": "qwen35"}, nil)
				w := createRequest(t, s.CreateHandler, api.CreateRequest{
					Model: "base", Files: map[string]string{"model.gguf": digest},
					Capabilities: baseCaps, Stream: &stream,
				})
				if w.Code != http.StatusOK {
					t.Fatalf("create base: %d %s", w.Code, w.Body)
				}
			}

			modelfile := "FROM base\nCAPABILITY decision\nCAPABILITY decision\n"
			for _, name := range []string{"declared", "roundtrip", "inherited"} {
				mf, err := parser.ParseFile(strings.NewReader(modelfile))
				if err != nil {
					t.Fatal(err)
				}
				req, err := mf.CreateRequest(t.TempDir())
				if err != nil {
					t.Fatal(err)
				}
				req.Model, req.Stream = name, &stream
				// The CLI uploads local files under their base names.
				files := make(map[string]string, len(req.Files))
				for path, digest := range req.Files {
					files[filepath.Base(path)] = digest
				}
				req.Files = files
				w := createRequest(t, s.CreateHandler, req)
				if w.Code != http.StatusOK {
					t.Fatalf("create %s: %d %s", name, w.Code, w.Body)
				}
				cfg := readCreatedModelConfig(t, name)
				if want := []string{"decision"}; !slices.Equal(cfg.Capabilities, want) {
					t.Fatalf("%s capabilities = %v, want %v", name, cfg.Capabilities, want)
				}
				if !cfg.CapabilitiesExplicit {
					t.Fatalf("%s lost its explicit capability declaration", name)
				}
				shown, err := GetModelInfo(api.ShowRequest{Model: name})
				if err != nil {
					t.Fatal(err)
				}
				if !slices.Equal(shown.Capabilities, []model.Capability{model.CapabilityDecision}) || !strings.Contains(shown.Modelfile, "CAPABILITY decision\n") {
					t.Fatalf("show %s lost capability: %+v", name, shown)
				}
				w = createRequest(t, s.GenerateHandler, api.GenerateRequest{Model: name, Prompt: "hello"})
				if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), "does not support generate") {
					t.Fatalf("generate %s: %d %s", name, w.Code, w.Body)
				}
				w = createRequest(t, s.ChatHandler, api.ChatRequest{Model: name, Messages: []api.Message{{Role: "user", Content: "hello"}}})
				if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), "does not support chat") {
					t.Fatalf("chat %s: %d %s", name, w.Code, w.Body)
				}
				modelfile = shown.Modelfile
				if name == "roundtrip" {
					modelfile = "FROM roundtrip\n"
				}
			}
			w := createRequest(t, s.CreateHandler, api.CreateRequest{
				Model: "replaced", From: "inherited", Capabilities: []string{"completion"}, Stream: &stream,
			})
			if w.Code != http.StatusOK {
				t.Fatalf("replace capabilities: %d %s", w.Code, w.Body)
			}
			shown, err := GetModelInfo(api.ShowRequest{Model: "replaced"})
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(shown.Capabilities, []model.Capability{model.CapabilityCompletion}) {
				t.Fatalf("replacement capabilities = %v, want [completion]", shown.Capabilities)
			}
		})
	}
}

func TestLegacyModelfileCapabilities(t *testing.T) {
	m := Model{ModelPath: "base", Config: model.ConfigV2{
		ModelFormat: "safetensors", Capabilities: []string{"completion"}, Parser: "qwen3.5",
	}}
	mf, err := parser.ParseFile(strings.NewReader(m.String()))
	if err != nil {
		t.Fatal(err)
	}
	req, err := mf.CreateRequest(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"completion", "tools", "thinking"}; !slices.Equal(req.Capabilities, want) {
		t.Fatalf("exported capabilities = %v, want %v", req.Capabilities, want)
	}
}

func TestCreateRejectsUnknownCapability(t *testing.T) {
	gin.SetMode(gin.TestMode)
	var s Server
	for _, capability := range []string{"", "system-one", "decision tools"} {
		w := createRequest(t, s.CreateHandler, api.CreateRequest{
			Model: "invalid", From: "missing", Capabilities: []string{capability}, Stream: &stream,
		})
		if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), "unknown capability") {
			t.Fatalf("capability %q: %d %s", capability, w.Code, w.Body)
		}
	}
}
