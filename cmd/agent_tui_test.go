package cmd

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/agent/skills"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/tui"
	modelpkg "github.com/ollama/ollama/types/model"
)

func TestAgentSystemPromptIncludesModel(t *testing.T) {
	prompt := agentSystemPromptAt(time.Date(2026, time.June, 12, 9, 30, 0, 0, time.UTC), "llama3.2", nil, false, "")
	for _, want := range []string{
		"You are running in Ollama as part of the Ollama agent, and the model is llama3.2.",
		"Current date: Friday, June 12, 2026.",
		"Be concise, practical, and action-oriented.",
		"Use bash carefully.",
		"Tell the user about meaningful changes",
	} {
		if !strings.Contains(prompt, want) {
			t.Fatalf("prompt missing %q:\n%s", want, prompt)
		}
	}
}

func TestAgentToolsRegistryNoCloudDisablesWebTools(t *testing.T) {
	t.Setenv("OLLAMA_NO_CLOUD", "1")
	t.Setenv("OLLAMA_AGENT_DISABLE_BASH", "")
	t.Setenv("OLLAMA_AGENT_DISABLE_WEBSEARCH", "")

	statusCalls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/show":
			_ = json.NewEncoder(w).Encode(api.ShowResponse{
				Capabilities: []modelpkg.Capability{modelpkg.CapabilityTools},
			})
		case "/api/status":
			statusCalls++
			_ = json.NewEncoder(w).Encode(api.StatusResponse{})
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	baseURL, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	registry := agentToolsRegistry(context.Background(), api.NewClient(baseURL, srv.Client()), "test-model", nil)
	if registry == nil {
		t.Fatal("registry = nil, want local tools")
	}
	if !registry.Has("bash") || !registry.Has("read") || !registry.Has("edit") {
		t.Fatalf("local tools missing: %v", registry.Names())
	}
	if registry.Has("list") {
		t.Fatalf("list tool should not be registered; got %v", registry.Names())
	}
	if registry.Has("web_search") || registry.Has("web_fetch") {
		t.Fatalf("web tools should be disabled when OLLAMA_NO_CLOUD is set: %v", registry.Names())
	}
	if statusCalls != 0 {
		t.Fatalf("/api/status calls = %d, want local no-cloud short-circuit", statusCalls)
	}
}

func TestAgentToolsRegistryRegistersSkillTool(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/show":
			_ = json.NewEncoder(w).Encode(api.ShowResponse{
				Capabilities: []modelpkg.Capability{modelpkg.CapabilityTools},
			})
		case "/api/status":
			_ = json.NewEncoder(w).Encode(api.StatusResponse{})
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	baseURL, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	catalog := &skills.Catalog{Skills: []skills.Skill{{Name: "go-code", Description: "Write Go code."}}}
	registry := agentToolsRegistry(context.Background(), api.NewClient(baseURL, srv.Client()), "test-model", catalog)
	if registry == nil || !registry.Has("skill") {
		t.Fatalf("registry missing skill tool: %#v", registry)
	}
}

func TestAgentModelOptionsIncludesCloudRecommendationsAndLocalModels(t *testing.T) {
	var recommendationsCalled bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/tags":
			_ = json.NewEncoder(w).Encode(api.ListResponse{
				Models: []api.ListModelResponse{{
					Name: "llama3.2:latest",
					Details: api.ModelDetails{
						Family:            "llama",
						ParameterSize:     "3B",
						QuantizationLevel: "Q4_K_M",
						ContextLength:     131072,
					},
					Size: 2_000_000_000,
				}},
			})
		case "/api/status":
			_ = json.NewEncoder(w).Encode(api.StatusResponse{})
		case "/api/experimental/model-recommendations":
			recommendationsCalled = true
			_ = json.NewEncoder(w).Encode(api.ModelRecommendationsResponse{
				Recommendations: []api.ModelRecommendation{
					{Model: "qwen3.5:cloud", Description: "cloud reasoning", ContextLength: 262144, RequiredPlan: "pro"},
					{Model: "gemma4", Description: "local recommendation should be ignored"},
				},
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	baseURL, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	options, err := agentModelOptions(context.Background(), api.NewClient(baseURL, srv.Client()))
	if err != nil {
		t.Fatal(err)
	}
	if !recommendationsCalled {
		t.Fatal("expected recommendations endpoint to be called")
	}
	if got, want := modelOptionNames(options), []string{"qwen3.5:cloud", "llama3.2"}; !slices.Equal(got, want) {
		t.Fatalf("model options = %#v, want %#v", got, want)
	}
	if options[0].Description == "" || options[1].Description == "" {
		t.Fatalf("expected descriptions for model options: %#v", options)
	}
	if !options[0].Recommended || options[1].Recommended {
		t.Fatalf("recommended flags = %#v, want only cloud recommendation marked", options)
	}
	if strings.Contains(options[0].Description, "plan") || strings.Contains(options[0].Description, "pro") {
		t.Fatalf("recommendation description should not include plan type: %q", options[0].Description)
	}
}

func TestAgentModelOptionsNoCloudSkipsCloudRecommendations(t *testing.T) {
	t.Setenv("OLLAMA_NO_CLOUD", "1")

	var recommendationsCalled bool
	var statusCalled bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/tags":
			_ = json.NewEncoder(w).Encode(api.ListResponse{
				Models: []api.ListModelResponse{{Name: "llama3.2:latest"}},
			})
		case "/api/status":
			statusCalled = true
			_ = json.NewEncoder(w).Encode(api.StatusResponse{})
		case "/api/experimental/model-recommendations":
			recommendationsCalled = true
			_ = json.NewEncoder(w).Encode(api.ModelRecommendationsResponse{
				Recommendations: []api.ModelRecommendation{{Model: "qwen3.5:cloud"}},
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	baseURL, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	options, err := agentModelOptions(context.Background(), api.NewClient(baseURL, srv.Client()))
	if err != nil {
		t.Fatal(err)
	}
	if recommendationsCalled {
		t.Fatal("recommendations endpoint should not be called when no-cloud is set")
	}
	if statusCalled {
		t.Fatal("status endpoint should not be called when local no-cloud short-circuits")
	}
	if got, want := modelOptionNames(options), []string{"llama3.2"}; !slices.Equal(got, want) {
		t.Fatalf("model options = %#v, want %#v", got, want)
	}
}

func modelOptionNames(options []tui.ChatModelOption) []string {
	names := make([]string, 0, len(options))
	for _, option := range options {
		names = append(names, option.Name)
	}
	return names
}
