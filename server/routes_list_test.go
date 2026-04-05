package server

import (
	"encoding/json"
	"net/http"
	"slices"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

func TestList(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Setenv("OLLAMA_MODELS", t.TempDir())

	expectNames := []string{
		"mistral:7b-instruct-q4_0",
		"zephyr:7b-beta-q5_K_M",
		"apple/OpenELM:latest",
		"boreas:2b-code-v1.5-q6_K",
		"notus:7b-v1-IQ2_S",
		// TODO: host:port currently fails on windows (#4107)
		// "localhost:5000/library/eurus:700b-v0.5-iq3_XXS",
		"mynamespace/apeliotes:latest",
		"myhost/mynamespace/lips:code",
	}

	var s Server
	for _, n := range expectNames {
		_, digest := createBinFile(t, nil, nil)

		createRequest(t, s.CreateHandler, api.CreateRequest{
			Name:  n,
			Files: map[string]string{"test.gguf": digest},
		})
	}

	w := createRequest(t, s.ListHandler, nil)
	if w.Code != http.StatusOK {
		t.Fatalf("expected status code 200, actual %d", w.Code)
	}

	var resp api.ListResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}

	if len(resp.Models) != len(expectNames) {
		t.Fatalf("expected %d models, actual %d", len(expectNames), len(resp.Models))
	}

	actualNames := make([]string, len(resp.Models))
	for i, m := range resp.Models {
		actualNames[i] = m.Name
	}

	slices.Sort(actualNames)
	slices.Sort(expectNames)

	if !slices.Equal(actualNames, expectNames) {
		t.Fatalf("expected slices to be equal %v", actualNames)
	}

	// all test models are basic GGUF without pooling_type, so they should have "completion"
	for _, m := range resp.Models {
		if !slices.Contains(m.Capabilities, model.CapabilityCompletion) {
			t.Errorf("expected model %s to have completion capability, got %v", m.Name, m.Capabilities)
		}
	}
}

func TestListCapabilities(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Setenv("OLLAMA_MODELS", t.TempDir())

	var s Server

	// Create a completion model (no pooling_type)
	_, completionDigest := createBinFile(t, nil, nil)
	createRequest(t, s.CreateHandler, api.CreateRequest{
		Name:  "completion-model",
		Files: map[string]string{"test.gguf": completionDigest},
	})

	// Create an embedding model (with pooling_type)
	_, embeddingDigest := createBinFile(t, map[string]any{"pooling_type": uint32(1)}, nil)
	createRequest(t, s.CreateHandler, api.CreateRequest{
		Name:  "embedding-model",
		Files: map[string]string{"test.gguf": embeddingDigest},
	})

	w := createRequest(t, s.ListHandler, nil)
	if w.Code != http.StatusOK {
		t.Fatalf("expected status code 200, actual %d", w.Code)
	}

	var resp api.ListResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}

	for _, m := range resp.Models {
		switch m.Name {
		case "completion-model:latest":
			if !slices.Contains(m.Capabilities, model.CapabilityCompletion) {
				t.Errorf("expected completion capability, got %v", m.Capabilities)
			}
			if slices.Contains(m.Capabilities, model.CapabilityEmbedding) {
				t.Errorf("unexpected embedding capability for completion model")
			}
		case "embedding-model:latest":
			if !slices.Contains(m.Capabilities, model.CapabilityEmbedding) {
				t.Errorf("expected embedding capability, got %v", m.Capabilities)
			}
			if slices.Contains(m.Capabilities, model.CapabilityCompletion) {
				t.Errorf("unexpected completion capability for embedding model")
			}
		default:
			t.Errorf("unexpected model %s", m.Name)
		}
	}
}
