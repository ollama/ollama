package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"slices"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
)

func TestList(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	envconfig.LoadConfig()

	expectNames := []string{
		"mistral:7b-instruct-q4_0",
		"zephyr:7b-beta-q5_K_M",
		"apple/OpenELM",
		"boreas:2b-code-v1.5-q6_K",
		"notus:7b-v1-IQ2_S",
		// TODO: host:port currently fails on windows (#4107)
		// "localhost:5000/library/eurus:700b-v0.5-iq3_XXS",
		"mynamespace/apeliotes",
		"myhost/mynamespace/lips:code",
	}

	var s Server
	for _, n := range expectNames {
		createRequest(t, s.CreateModelHandler, api.CreateRequest{
			Name:      n,
			Modelfile: fmt.Sprintf("FROM %s", createBinFile(t, nil, nil)),
		})
	}

	w := createRequest(t, s.ListModelsHandler, nil)
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
}
