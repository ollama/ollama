package server

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

func TestExtractionModelEndpoints(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	config, err := manifest.NewLayer(strings.NewReader(`{"model_format":"safetensors","capabilities":["extraction"],"file_type":"F32"}`), "application/vnd.docker.container.image.v1+json")
	if err != nil {
		t.Fatal(err)
	}
	encoder, err := manifest.NewLayer(strings.NewReader(`{"architectures":["GLiNER"],"model_type":"gliner","hidden_size":512,"encoder_config":{"hidden_size":768,"num_hidden_layers":6,"max_position_embeddings":512}}`), "application/vnd.ollama.image.json")
	if err != nil {
		t.Fatal(err)
	}
	encoder.Name = "config.json"
	name := model.ParseName("gliner")
	if err := manifest.WriteManifest(name, config, []manifest.Layer{encoder}); err != nil {
		t.Fatal(err)
	}
	resp, err := GetModelInfo(api.ShowRequest{Model: name.String(), Verbose: true})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Capabilities) != 1 || resp.Capabilities[0] != model.CapabilityExtraction {
		t.Fatalf("capabilities: %v", resp.Capabilities)
	}
	if resp.ModelInfo["general.architecture"] != "gliner" || resp.ModelInfo["gliner.context_length"] != 512 || resp.ModelInfo["gliner.block_count"] != 6 {
		t.Fatalf("model info: %v", resp.ModelInfo)
	}
	if resp.Details.QuantizationLevel != "F32" || !strings.Contains(resp.Modelfile, "FROM gliner:latest\n") {
		t.Fatalf("details: %+v; Modelfile: %s", resp.Details, resp.Modelfile)
	}
	// Neither embedding endpoint should try to read a nonexistent GGUF layer
	// or load a runner for an extraction-only model.
	s := &Server{}
	for _, endpoint := range []struct {
		path    string
		handler gin.HandlerFunc
	}{
		{"/api/embed", s.EmbedHandler},
		{"/api/embeddings", s.EmbeddingsHandler},
	} {
		t.Run(endpoint.path, func(t *testing.T) {
			w := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(w)
			c.Request = httptest.NewRequest(http.MethodPost, endpoint.path, strings.NewReader(`{"model":"gliner","input":"text","prompt":"text"}`))
			c.Request.Header.Set("Content-Type", "application/json")
			endpoint.handler(c)
			if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), "use /api/extract") {
				t.Fatalf("%d %s", w.Code, w.Body)
			}
		})
	}
}

func TestExtractHandlerRejectsInvalidInputBeforeLoading(t *testing.T) {
	for _, body := range []string{
		`{`, `{}`, `{"model":"gliner","labels":[]}`,
		`{"model":"gliner","labels":["person","person"]}`,
		`{"model":"gliner","labels":["person"],"threshold":2}`,
		`{"model":"gliner","labels":["person"],"input":["text"]}`,
	} {
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Request = httptest.NewRequest(http.MethodPost, "/api/extract", strings.NewReader(body))
		c.Request.Header.Set("Content-Type", "application/json")
		(&Server{}).ExtractHandler(c)
		if w.Code != http.StatusBadRequest {
			t.Fatalf("%s: %d %s", body, w.Code, w.Body)
		}
	}
}

func TestExtractionCapability(t *testing.T) {
	m := &Model{Config: model.ConfigV2{ModelFormat: "safetensors", Capabilities: []string{"extraction"}}}
	if err := m.CheckCapabilities(model.CapabilityExtraction); err != nil {
		t.Fatal(err)
	}
	if err := m.CheckCapabilities(model.CapabilityCompletion); !errors.Is(err, errCapabilityCompletion) {
		t.Fatal(err)
	}
	m = &Model{Config: model.ConfigV2{ModelFormat: "safetensors", Capabilities: []string{"completion"}}}
	if err := m.CheckCapabilities(model.CapabilityExtraction); !errors.Is(err, errCapabilityExtraction) {
		t.Fatal(err)
	}
}
