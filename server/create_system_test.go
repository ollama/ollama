package server

import (
	"encoding/json"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/create"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/parser"
	"github.com/ollama/ollama/types/model"
)

func TestCreateSystemOverride(t *testing.T) {
	for _, format := range []string{"gguf", "safetensors"} {
		t.Run(format, func(t *testing.T) {
			t.Setenv("OLLAMA_MODELS", t.TempDir())
			gin.SetMode(gin.TestMode)
			var s Server
			router := gin.New()
			router.POST("/api/create", s.CreateHandler)
			router.POST("/api/show", s.ShowHandler)
			ts := httptest.NewServer(router)
			defer ts.Close()
			baseURL, err := url.Parse(ts.URL)
			if err != nil {
				t.Fatal(err)
			}
			client := api.NewClient(baseURL, ts.Client())
			if format == "safetensors" {
				layer, err := manifest.NewLayer(strings.NewReader("inherited system"), "application/vnd.ollama.image.system")
				if err != nil {
					t.Fatal(err)
				}
				createSafetensorsTestModel(t, "base", model.ConfigV2{ModelFormat: "safetensors", Renderer: "qwen3.5", Capabilities: []string{"completion"}}, []manifest.Layer{layer})
			} else {
				_, digest := createBinFile(t, nil, nil)
				if err := client.Create(t.Context(), &api.CreateRequest{
					Model: "base", Files: map[string]string{"model.gguf": digest},
					System: "inherited system", Stream: &stream,
				}, func(api.ProgressResponse) error { return nil }); err != nil {
					t.Fatal(err)
				}
			}
			for _, tc := range []struct{ name, directive, want string }{
				{"omitted", "", "inherited system"},
				{"nonempty", "SYSTEM replacement\n", "replacement"},
				{"empty", "SYSTEM \"\"\n", ""},
				{"empty-triple", "SYSTEM \"\"\"\"\"\"\n", ""},
			} {
				t.Run(tc.name, func(t *testing.T) {
					mf, err := parser.ParseFile(strings.NewReader("FROM base\n" + tc.directive))
					if err != nil {
						t.Fatal(err)
					}
					req, err := mf.CreateRequest(t.TempDir())
					if err != nil {
						t.Fatal(err)
					}
					req.Model, req.Stream = tc.name, &stream
					if err := client.Create(t.Context(), req, func(api.ProgressResponse) error { return nil }); err != nil {
						t.Fatal(err)
					}
					shown, err := client.Show(t.Context(), &api.ShowRequest{Model: tc.name})
					if err != nil {
						t.Fatal(err)
					}
					if shown.System != tc.want {
						t.Fatalf("created system = %q, want %q", shown.System, tc.want)
					}
				})
			}

			err = client.Create(t.Context(), &api.CreateRequest{
				Model: "list-with-system", List: []string{"base"}, SystemSet: true, Stream: &stream,
			}, func(api.ProgressResponse) error { return nil })
			if err == nil || !strings.Contains(err.Error(), "manifest list creation cannot be combined") {
				t.Fatalf("explicit empty system on manifest list: %v, want incompatible options error", err)
			}
		})
	}
}

func TestWriteSafetensorsManifestSystemOverride(t *testing.T) {
	for _, tc := range []struct{ name, input, want string }{
		{"omitted", `{}`, "inherited system"},
		{"nonempty", `{"system":"replacement"}`, "replacement"},
		{"empty", `{"system":""}`, ""},
		{"null", `{"system":null}`, "inherited system"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("OLLAMA_MODELS", t.TempDir())
			var req api.CreateRequest
			if err := json.Unmarshal([]byte(tc.input), &req); err != nil {
				t.Fatal(err)
			}
			req.Model = "child"
			info := create.ManifestInfo{Layers: []create.LayerInfo{
				createTestLayerInfo(t, "", "application/vnd.ollama.image.system", []byte("inherited system")),
			}}
			if err := writeSafetensorsManifest(req, "", func(api.ProgressResponse) {})(t.Context(), req.Model, info); err != nil {
				t.Fatal(err)
			}
			shown, err := GetModelInfo(api.ShowRequest{Model: req.Model})
			if err != nil {
				t.Fatal(err)
			}
			if shown.System != tc.want {
				t.Fatalf("created system = %q, want %q", shown.System, tc.want)
			}
		})
	}
}
