package server

import (
	"net/http"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/parser"
	"github.com/ollama/ollama/template"
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
				_, digest := createBinFile(t, map[string]any{
					"general.architecture": "qwen35", "qwen35.decision.type": "clef",
				}, nil)
				w := createRequest(t, s.CreateHandler, api.CreateRequest{
					Model: "base", Files: map[string]string{"model.gguf": digest},
					Capabilities: baseCaps, Stream: &stream,
				})
				if w.Code != http.StatusOK {
					t.Fatalf("create base: %d %s", w.Code, w.Body)
				}
			}

			modelfile := "FROM base\nCAPABILITY decision\nCAPABILITY completion\nCAPABILITY decision\n"
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
				if want := []string{"decision", "completion"}; !cfg.CapabilitiesExplicit || !slices.Equal(cfg.Capabilities, want) {
					t.Fatalf("%s capabilities = %v, want %v", name, cfg.Capabilities, want)
				}
				shown, err := GetModelInfo(api.ShowRequest{Model: name})
				if err != nil {
					t.Fatal(err)
				}
				if !slices.Contains(shown.Capabilities, model.CapabilityDecision) || !strings.Contains(shown.Modelfile, "CAPABILITY decision\n") {
					t.Fatalf("show %s lost capability: %+v", name, shown)
				}
				if want := []model.Capability{model.CapabilityDecision, model.CapabilityCompletion}; !slices.Equal(shown.Capabilities, want) || shown.Thinking != nil {
					t.Fatalf("show %s capabilities = %v thinking = %v, want only %v", name, shown.Capabilities, shown.Thinking, want)
				}
				cache := newModelShowCache()
				for i := range 2 {
					cached, err := cache.GetLocal(api.ShowRequest{Model: name})
					if err != nil {
						t.Fatal(err)
					}
					if !slices.Equal(cached.Capabilities, shown.Capabilities) {
						t.Fatalf("cached show %s (%d): %v, want %v", name, i, cached.Capabilities, shown.Capabilities)
					}
				}

				m, err := GetModel(name)
				if err != nil {
					t.Fatal(err)
				}
				if err := m.CheckCapabilities(model.CapabilityCompletion, model.CapabilityDecision); err != nil {
					t.Fatalf("%s serving capabilities: %v", name, err)
				}
				modelfile = shown.Modelfile
				if name == "roundtrip" {
					modelfile = "FROM roundtrip\n"
				}
			}
			listed, err := listModels(t.Context())
			if err != nil {
				t.Fatal(err)
			}
			for _, m := range listed {
				want := []model.Capability{model.CapabilityDecision, model.CapabilityCompletion}
				if m.Name == "base:latest" {
					want = []model.Capability{model.CapabilityCompletion, model.CapabilityVision}
				}
				if !slices.Equal(m.Capabilities, want) {
					t.Fatalf("list %s capabilities = %v, want %v", m.Name, m.Capabilities, want)
				}
			}
		})
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

func TestLegacyCapabilityExport(t *testing.T) {
	for _, tt := range []struct {
		name string
		caps []string
		want []string
	}{
		{"completion", []string{"completion"}, []string{"completion", "tools", "thinking"}},
		{"decision", []string{"decision", "completion"}, []string{"decision"}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			m := Model{Template: template.DefaultTemplate, Config: model.ConfigV2{
				ModelFormat: "safetensors", Capabilities: tt.caps, Parser: "qwen3.5",
			}}
			mf, err := parser.ParseFile(strings.NewReader(m.String()))
			if err != nil {
				t.Fatal(err)
			}
			var got []string
			for _, command := range mf.Commands {
				if command.Name == "capability" {
					got = append(got, command.Args)
				}
			}
			if !slices.Equal(got, tt.want) {
				t.Fatalf("exported capabilities = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestExplicitDecisionCapabilities(t *testing.T) {
	for _, format := range []string{"gguf", "safetensors"} {
		for _, caps := range [][]string{
			{"decision"},
			{"completion", "decision"},
			{"completion", "decision", "thinking", "tools", "vision"},
		} {
			t.Run(format+"/"+strings.Join(caps, "+"), func(t *testing.T) {
				t.Setenv("OLLAMA_MODELS", t.TempDir())
				cfg := model.ConfigV2{
					ModelFormat: format, Parser: "qwen3.5", Renderer: "qwen3.5",
					Capabilities: caps, CapabilitiesExplicit: true,
				}
				_, digest := createBinFile(t, map[string]any{
					"general.architecture": "qwen35", "qwen35.decision.type": "clef",
					"qwen35.vision.block_count": uint32(1),
				}, nil)
				name := "declared"
				if format == "gguf" {
					config, err := createConfigLayer(cfg)
					if err != nil {
						t.Fatal(err)
					}
					if err := manifest.WriteManifest(model.ParseName(name), *config, []manifest.Layer{{
						MediaType: "application/vnd.ollama.image.model", Digest: digest,
					}}); err != nil {
						t.Fatal(err)
					}
				} else {
					createSafetensorsTestModel(t, name, cfg, []manifest.Layer{{
						MediaType: "application/vnd.ollama.image.projector", Digest: digest,
					}})
				}
				m, err := GetModel(name)
				if err != nil {
					t.Fatal(err)
				}
				var want []model.Capability
				for _, cap := range caps {
					want = append(want, model.Capability(cap))
				}
				for _, source := range []templateCapabilitySource{templateCapabilitySelected, templateCapabilityGo, templateCapabilityChat} {
					if got := m.capabilitiesForTemplate(source); !slices.Equal(got, want) {
						t.Fatalf("serving capabilities (%d) = %v, want %v", source, got, want)
					}
				}
				if err := m.CheckCapabilities(want...); err != nil {
					t.Fatal(err)
				}
				if !slices.Contains(want, model.CapabilityCompletion) && m.CheckCapabilities(model.CapabilityCompletion) == nil {
					t.Fatal("decision-only model accepts completion")
				}
				shown, err := GetModelInfo(api.ShowRequest{Model: name})
				if err != nil {
					t.Fatal(err)
				}
				if !slices.Equal(shown.Capabilities, want) {
					t.Fatalf("show capabilities = %v, want %v", shown.Capabilities, want)
				}
				if thinking := slices.Contains(want, model.CapabilityThinking); thinking != (shown.Thinking != nil && shown.Thinking.Supports(true)) {
					t.Fatalf("show thinking = %v, want thinking supported: %t", shown.Thinking, thinking)
				}
				listed, err := listModels(t.Context())
				if err != nil {
					t.Fatal(err)
				}
				if len(listed) != 1 || !slices.Equal(listed[0].Capabilities, want) {
					t.Fatalf("list = %+v, want one model with capabilities %v", listed, want)
				}
			})
		}
	}
}

func TestDecisionPublicCapabilities(t *testing.T) {
	for _, format := range []string{"gguf", "safetensors"} {
		for _, declared := range [][]string{nil, {"vision"}, {"vision", "audio"}} {
			name := strings.Join(append([]string{format, "decision"}, declared...), "-")
			t.Run(name, func(t *testing.T) {
				t.Setenv("OLLAMA_MODELS", t.TempDir())
				cfg := model.ConfigV2{ModelFormat: format, Parser: "qwen3.5", Capabilities: append([]string{"decision"}, declared...)}
				want := []model.Capability{model.CapabilityDecision}
				// The GGUF inherits vision from its base model; the safetensors
				// fixture ships a projector.
				if len(declared) > 0 || format == "safetensors" {
					want = append(want, model.CapabilityVision)
				}
				if slices.Contains(declared, "audio") {
					want = append(want, model.CapabilityAudio)
				}
				_, digest := createBinFile(t, map[string]any{
					"general.architecture": "qwen35", "qwen35.vision.block_count": uint32(1),
				}, nil)
				if format == "gguf" {
					config, err := createConfigLayer(cfg)
					if err != nil {
						t.Fatal(err)
					}
					if err := manifest.WriteManifest(model.ParseName(name), *config, []manifest.Layer{{
						MediaType: "application/vnd.ollama.image.model", Digest: digest,
					}}); err != nil {
						t.Fatal(err)
					}
				} else {
					createSafetensorsTestModel(t, name, cfg, []manifest.Layer{{
						MediaType: "application/vnd.ollama.image.projector", Digest: digest,
					}})
				}
				m, err := GetModel(name)
				if err != nil {
					t.Fatal(err)
				}
				if err := m.CheckCapabilities(model.CapabilityDecision, model.CapabilityVision, model.CapabilityThinking); err != nil {
					t.Fatalf("fixture must retain runtime capabilities: %v", err)
				}
				shown, err := GetModelInfo(api.ShowRequest{Model: name})
				if err != nil {
					t.Fatal(err)
				}
				if !slices.Equal(shown.Capabilities, want) || shown.Thinking != nil {
					t.Fatalf("show capabilities = %v thinking = %v, want %v and no thinking", shown.Capabilities, shown.Thinking, want)
				}
				listed, err := listModels(t.Context())
				if err != nil {
					t.Fatal(err)
				}
				if len(listed) != 1 || !slices.Equal(listed[0].Capabilities, want) {
					t.Fatalf("list models = %+v, want one model with capabilities %v", listed, want)
				}
			})
		}
	}
}
