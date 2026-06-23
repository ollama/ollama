package cmd

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"slices"
	"strings"
	"testing"
	"time"

	coreagent "github.com/ollama/ollama/agent"
	"github.com/ollama/ollama/agent/skills"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/tui"
	"github.com/ollama/ollama/envconfig"
	modelpkg "github.com/ollama/ollama/types/model"
)

func setAgentTUITestCloudEnabled(t *testing.T) {
	t.Helper()
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	t.Setenv("OLLAMA_NO_CLOUD", "")
	envconfig.ReloadServerConfig()
}

func TestAgentSystemPromptIncludesModel(t *testing.T) {
	prompt := agentSystemPromptAt(time.Date(2026, time.June, 12, 9, 30, 0, 0, time.UTC), "llama3.2", nil, false, "")
	for _, want := range []string{
		"You are running in Ollama, in a harness to help the user accomplish tasks, and the model is llama3.2.",
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

func TestAgentHeadlessEventSinkPrintsThinkingUnlessHidden(t *testing.T) {
	t.Run("visible", func(t *testing.T) {
		output := captureStdout(t, func() {
			sink := &agentHeadlessEventSink{}
			if err := sink.Emit(coreagent.Event{Type: coreagent.EventThinkingDelta, Thinking: "thinking"}); err != nil {
				t.Fatal(err)
			}
			if err := sink.Emit(coreagent.Event{Type: coreagent.EventMessageDelta, Content: "answer"}); err != nil {
				t.Fatal(err)
			}
		})
		if output != "thinking\nanswer" {
			t.Fatalf("output = %q, want thinking followed by answer", output)
		}
	})

	t.Run("hidden", func(t *testing.T) {
		output := captureStdout(t, func() {
			sink := &agentHeadlessEventSink{hideThinking: true}
			if err := sink.Emit(coreagent.Event{Type: coreagent.EventThinkingDelta, Thinking: "thinking"}); err != nil {
				t.Fatal(err)
			}
			if err := sink.Emit(coreagent.Event{Type: coreagent.EventMessageDelta, Content: "answer"}); err != nil {
				t.Fatal(err)
			}
		})
		if output != "answer" {
			t.Fatalf("output = %q, want answer only", output)
		}
	})
}

func TestAgentHeadlessEventSinkPrintsOnlyFinishedToolEvents(t *testing.T) {
	output := captureStderr(t, func() {
		sink := &agentHeadlessEventSink{}
		if err := sink.Emit(coreagent.Event{
			Type:     coreagent.EventToolStarted,
			ToolName: "bash",
			Args:     map[string]any{"command": "pwd"},
		}); err != nil {
			t.Fatal(err)
		}
		if err := sink.Emit(coreagent.Event{
			Type:     coreagent.EventToolFinished,
			Status:   "done",
			ToolName: "bash",
			Args:     map[string]any{"command": "pwd"},
		}); err != nil {
			t.Fatal(err)
		}
		if err := sink.Emit(coreagent.Event{
			Type:     coreagent.EventToolFinished,
			Status:   "denied",
			ToolName: "edit",
			Args:     map[string]any{"path": "main.go"},
		}); err != nil {
			t.Fatal(err)
		}
		if err := sink.Emit(coreagent.Event{
			Type:     coreagent.EventToolFinished,
			Status:   "done",
			ToolName: "web_fetch",
			Args:     map[string]any{"url": "https://example.com"},
			Error:    "timeout",
		}); err != nil {
			t.Fatal(err)
		}
	})
	if strings.Contains(output, "in progress") {
		t.Fatalf("headless output should not include in-progress tool events:\n%s", output)
	}
	for _, want := range []string{
		`• Bash("pwd") done`,
		`• Edit("main.go") failed`,
		`• Web Fetch("https://example.com") failed`,
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("headless output missing %q:\n%s", want, output)
		}
	}
}

func captureStdout(t *testing.T, fn func()) string {
	t.Helper()

	oldStdout := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	os.Stdout = w
	t.Cleanup(func() {
		os.Stdout = oldStdout
	})

	fn()

	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	out, err := io.ReadAll(r)
	if err != nil {
		t.Fatal(err)
	}
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	os.Stdout = oldStdout
	return string(out)
}

func captureStderr(t *testing.T, fn func()) string {
	t.Helper()

	oldStderr := os.Stderr
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	os.Stderr = w
	t.Cleanup(func() {
		os.Stderr = oldStderr
	})

	fn()

	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	out, err := io.ReadAll(r)
	if err != nil {
		t.Fatal(err)
	}
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	os.Stderr = oldStderr
	return string(out)
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
	setAgentTUITestCloudEnabled(t)

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

func TestPreloadAgentModelIfLocalLoadsLocalModel(t *testing.T) {
	var generateReq api.GenerateRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/show":
			_ = json.NewEncoder(w).Encode(api.ShowResponse{})
		case "/api/generate":
			if err := json.NewDecoder(r.Body).Decode(&generateReq); err != nil {
				t.Fatal(err)
			}
			_ = json.NewEncoder(w).Encode(api.GenerateResponse{Done: true})
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	baseURL, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	keepAlive := api.Duration{Duration: 5 * time.Minute}
	think := &api.ThinkValue{Value: "low"}
	err = preloadAgentModelIfLocal(context.Background(), api.NewClient(baseURL, srv.Client()), AgentTUIOptions{
		KeepAlive: &keepAlive,
		Think:     think,
	}, "llama3.2")
	if err != nil {
		t.Fatal(err)
	}
	if generateReq.Model != "llama3.2" {
		t.Fatalf("generate model = %q, want llama3.2", generateReq.Model)
	}
	if generateReq.KeepAlive == nil || generateReq.KeepAlive.Duration != 5*time.Minute {
		t.Fatalf("generate keepalive = %#v, want 5m", generateReq.KeepAlive)
	}
	if generateReq.Think == nil || generateReq.Think.String() != "low" {
		t.Fatalf("generate think = %#v, want low", generateReq.Think)
	}
}

func TestPreloadAgentModelIfLocalSkipsCloudModel(t *testing.T) {
	generateCalled := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/show":
			_ = json.NewEncoder(w).Encode(api.ShowResponse{RemoteHost: "https://ollama.com"})
		case "/api/generate":
			generateCalled = true
			t.Fatal("cloud model should not be preloaded with generate")
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	baseURL, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	if err := preloadAgentModelIfLocal(context.Background(), api.NewClient(baseURL, srv.Client()), AgentTUIOptions{}, "kimi-k2:cloud"); err != nil {
		t.Fatal(err)
	}
	if generateCalled {
		t.Fatal("generate was called for cloud model")
	}
}

func modelOptionNames(options []tui.ChatModelOption) []string {
	names := make([]string, 0, len(options))
	for _, option := range options {
		names = append(names, option.Name)
	}
	return names
}
