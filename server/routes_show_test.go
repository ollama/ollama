package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/llm"
	"github.com/stretchr/testify/assert"
)

func TestShow(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	envconfig.LoadConfig()

	var s Server

	createRequest(t, s.CreateModelHandler, api.CreateRequest{
		Name: "show-model",
		Modelfile: fmt.Sprintf(
			"FROM %s\nFROM %s",
			createBinFile(t, llm.KV{"general.architecture": "test"}, nil),
			createBinFile(t, llm.KV{"general.architecture": "clip"}, nil),
		),
	})

	w := createRequest(t, s.ShowModelHandler, api.ShowRequest{
		Name: "show-model",
	})

	if w.Code != http.StatusOK {
		t.Fatalf("expected status code 200, actual %d", w.Code)
	}

	var resp api.ShowResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}

	assert.Equal(t, "test", resp.ModelInfo["general.architecture"])
	assert.Equal(t, "clip", resp.ProjectorInfo["general.architecture"])
}
