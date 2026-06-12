package cmd

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"

	"github.com/ollama/ollama/agent/skills"
	"github.com/ollama/ollama/api"
	modelpkg "github.com/ollama/ollama/types/model"
)

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
	if !registry.Has("bash") || !registry.Has("read") || !registry.Has("list") || !registry.Has("edit") {
		t.Fatalf("local tools missing: %v", registry.Names())
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
