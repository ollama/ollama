package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/klauspost/compress/zstd"
)

func TestCodexDesktopRoutesCatalogModelToOllamaAndStripsCredentials(t *testing.T) {
	var gotPath, gotQuery, gotAuthorization, gotEncoding, gotMetadata string
	var gotBody []byte
	ollama := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotQuery = r.URL.RawQuery
		gotAuthorization = r.Header.Get("Authorization")
		gotEncoding = r.Header.Get("Content-Encoding")
		gotMetadata = r.Header.Get("X-Codex-Turn-Metadata")
		gotBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: routed\n\n")
	}))
	defer ollama.Close()

	chatGPT := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Fatal("catalog model should not reach ChatGPT")
	}))
	defer chatGPT.Close()

	catalogPath := writeCatalog(t, "glm-5.2:cloud")
	handler := newTestCodexDesktop(t, ollama.URL, chatGPT.URL+"/backend-api/codex", catalogPath)
	proxy := httptest.NewServer(handler)
	defer proxy.Close()

	payload := []byte(`{"model":"glm-5.2:cloud","stream":true,"reasoning":{"effort":"high"},"tools":[{"type":"web_search","external_web_access":false}]}`)
	encoder, err := zstd.NewWriter(nil)
	if err != nil {
		t.Fatal(err)
	}
	compressed := encoder.EncodeAll(payload, nil)
	encoder.Close()

	for _, authorization := range []string{"Bearer chatgpt-secret", "Bearer " + CodexDesktopManagedAPIKey} {
		req, err := http.NewRequest(http.MethodPost, proxy.URL+CodexDesktopPathPrefix+"/v1/responses?trace=1", bytes.NewReader(compressed))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Content-Encoding", "zstd")
		req.Header.Set("ChatGPT-Account-ID", "account-123")
		req.Header.Set("Authorization", authorization)
		req.Header.Set("X-Codex-Turn-Metadata", `{"thread":"secret"}`)

		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		responseBody, _ := io.ReadAll(resp.Body)
		resp.Body.Close()

		if resp.StatusCode != http.StatusOK || string(responseBody) != "data: routed\n\n" {
			t.Fatalf("response with %q = %d %q", authorization, resp.StatusCode, responseBody)
		}
		if gotPath != "/v1/responses" || gotQuery != "trace=1" {
			t.Fatalf("Ollama target = %s?%s", gotPath, gotQuery)
		}
		if gotAuthorization != "" || gotEncoding != "" || gotMetadata != "" {
			t.Fatalf("credentials leaked to Ollama: authorization=%q encoding=%q metadata=%q", gotAuthorization, gotEncoding, gotMetadata)
		}
		if string(gotBody) != string(payload) {
			t.Fatalf("Ollama body = %q, want decompressed %q", gotBody, payload)
		}
	}
}

func TestCodexDesktopRoutesOllamaAndNativeModelsThroughOneEndpoint(t *testing.T) {
	var ollamaCalls, chatGPTCalls int
	ollama := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		ollamaCalls++
		w.WriteHeader(http.StatusNoContent)
	}))
	defer ollama.Close()
	chatGPT := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		chatGPTCalls++
		w.WriteHeader(http.StatusNoContent)
	}))
	defer chatGPT.Close()

	handler := newTestCodexDesktop(t, ollama.URL, chatGPT.URL+"/backend-api/codex", writeCatalog(t, "glm-5.2:cloud"))
	endpoint := httptest.NewServer(handler)
	defer endpoint.Close()

	for _, model := range []string{"glm-5.2:cloud", "gpt-5.6-sol"} {
		req, err := http.NewRequest(
			http.MethodPost,
			endpoint.URL+CodexDesktopPathPrefix+"/v1/responses",
			strings.NewReader(fmt.Sprintf(`{"model":%q}`, model)),
		)
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Authorization", "Bearer chatgpt-secret")
		req.Header.Set("ChatGPT-Account-ID", "account-123")
		resp, err := endpoint.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		_ = resp.Body.Close()
		if resp.StatusCode != http.StatusNoContent {
			t.Fatalf("model %q status = %d", model, resp.StatusCode)
		}
	}

	if ollamaCalls != 1 || chatGPTCalls != 1 {
		t.Fatalf("routes through one endpoint: Ollama calls=%d ChatGPT calls=%d", ollamaCalls, chatGPTCalls)
	}
}

func TestNormalizeOllamaThinkingUsesRoutedModelContract(t *testing.T) {
	tests := []struct {
		name          string
		model         routingModel
		effort        string
		wantReasoning bool
		wantEffort    string
		omitEffort    bool
		wantThink     any
		hasThink      bool
	}{
		{
			name:          "legacy catalog preserves request",
			effort:        "high",
			wantReasoning: true,
			wantEffort:    "high",
		},
		{
			name:          "non-thinking model drops stale reasoning",
			model:         routingModel{Thinking: &routingThinkingMetadata{}},
			effort:        "high",
			wantReasoning: false,
		},
		{
			name: "binary thinking stays off",
			model: routingModel{Thinking: &routingThinkingMetadata{
				Supported: true,
				Levels:    []string{"none", "medium"},
				Values:    map[string]json.RawMessage{"none": json.RawMessage("false"), "medium": json.RawMessage("true")},
			}},
			effort:        "none",
			wantReasoning: true,
			wantEffort:    "none",
			wantThink:     false,
			hasThink:      true,
		},
		{
			name: "binary thinking uses medium for its enabled choice",
			model: routingModel{Thinking: &routingThinkingMetadata{
				Supported: true,
				Levels:    []string{"none", "medium"},
				Values:    map[string]json.RawMessage{"none": json.RawMessage("false"), "medium": json.RawMessage("true")},
			}},
			effort:        "medium",
			wantReasoning: true,
			wantEffort:    "medium",
			wantThink:     true,
			hasThink:      true,
		},
		{
			name: "binary thinking maps a stale enabled level to medium",
			model: routingModel{Thinking: &routingThinkingMetadata{
				Supported: true,
				Levels:    []string{"none", "medium"},
				Values:    map[string]json.RawMessage{"none": json.RawMessage("false"), "medium": json.RawMessage("true")},
			}},
			effort:        "high",
			wantReasoning: true,
			wantEffort:    "medium",
			wantThink:     true,
			hasThink:      true,
		},
		{
			name: "exact GLM ladder clamps xhigh to max",
			model: routingModel{Thinking: &routingThinkingMetadata{
				Supported: true,
				Levels:    []string{"low", "high", "max"},
				Values:    map[string]json.RawMessage{"low": json.RawMessage(`"low"`), "high": json.RawMessage(`"high"`), "max": json.RawMessage(`"max"`)},
			}},
			effort:        "xhigh",
			wantReasoning: true,
			wantEffort:    "max",
			wantThink:     "max",
			hasThink:      true,
		},
		{
			name: "exact GLM ladder omits unsupported medium",
			model: routingModel{Thinking: &routingThinkingMetadata{
				Supported: true,
				Levels:    []string{"low", "high", "max"},
			}},
			effort:        "medium",
			wantReasoning: true,
			omitEffort:    true,
		},
		{
			name: "always-thinking GLM omits stale off",
			model: routingModel{Thinking: &routingThinkingMetadata{
				Supported: true,
				Levels:    []string{"low", "high", "max"},
			}},
			effort:        "none",
			wantReasoning: true,
			omitEffort:    true,
		},
		{
			name: "minimal maps to Ollama low",
			model: routingModel{Thinking: &routingThinkingMetadata{
				Supported: true,
				Levels:    []string{"low", "medium", "high"},
				Values:    map[string]json.RawMessage{"low": json.RawMessage(`"low"`), "medium": json.RawMessage(`"medium"`), "high": json.RawMessage(`"high"`)},
			}},
			effort:        "minimal",
			wantReasoning: true,
			wantEffort:    "low",
			wantThink:     "low",
			hasThink:      true,
		},
		{
			name: "stale xhigh clamps to the strongest supported effort",
			model: routingModel{Thinking: &routingThinkingMetadata{
				Supported: true,
				Levels:    []string{"low", "medium", "high"},
				Values:    map[string]json.RawMessage{"low": json.RawMessage(`"low"`), "medium": json.RawMessage(`"medium"`), "high": json.RawMessage(`"high"`)},
			}},
			effort:        "xhigh",
			wantReasoning: true,
			wantEffort:    "high",
			wantThink:     "high",
			hasThink:      true,
		},
		{
			name: "unknown effort is omitted instead of enabling binary thinking",
			model: routingModel{Thinking: &routingThinkingMetadata{
				Supported: true,
				Levels:    []string{"none", "medium"},
			}},
			effort:        "unexpected",
			wantReasoning: true,
			omitEffort:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body, err := json.Marshal(map[string]any{
				"model": "test-model",
				"reasoning": map[string]any{
					"effort":  tt.effort,
					"summary": "auto",
				},
			})
			if err != nil {
				t.Fatal(err)
			}
			normalized, err := normalizeOllamaRequestBody(body, tt.model)
			if err != nil {
				t.Fatal(err)
			}
			var payload map[string]any
			if err := json.Unmarshal(normalized, &payload); err != nil {
				t.Fatal(err)
			}
			reasoning, ok := payload["reasoning"].(map[string]any)
			if ok != tt.wantReasoning {
				t.Fatalf("reasoning present = %v, want %v in %s", ok, tt.wantReasoning, normalized)
			}
			gotThink, hasThink := payload["think"]
			if hasThink != tt.hasThink || (hasThink && gotThink != tt.wantThink) {
				t.Fatalf("think = %#v, %v; want %#v, %v in %s", gotThink, hasThink, tt.wantThink, tt.hasThink, normalized)
			}
			if !tt.wantReasoning {
				return
			}
			gotEffort, hasEffort := reasoning["effort"].(string)
			if tt.omitEffort {
				if hasEffort {
					t.Fatalf("reasoning effort = %q, want model default with no explicit effort in %s", gotEffort, normalized)
				}
			} else if !hasEffort || gotEffort != tt.wantEffort {
				t.Fatalf("reasoning effort = %q, %v; want %q in %s", gotEffort, hasEffort, tt.wantEffort, normalized)
			}
			if got, _ := reasoning["summary"].(string); got != "auto" {
				t.Fatalf("reasoning summary = %q, want preserved", got)
			}
		})
	}
}

func TestNormalizeOllamaThinkingRejectsMalformedReasoning(t *testing.T) {
	model := routingModel{Thinking: &routingThinkingMetadata{
		Supported: true,
		Levels:    []string{"none", "medium"},
	}}
	_, err := normalizeOllamaRequestBody([]byte(`{"model":"test-model","reasoning":"high"}`), model)
	if err == nil || !strings.Contains(err.Error(), "decode reasoning") {
		t.Fatalf("normalize error = %v, want malformed reasoning error", err)
	}
}

func TestNormalizeFullAccessExecToolRemovesEscalationArguments(t *testing.T) {
	turnMetadata := `{"sandbox_mode":"danger-full-access"}`
	body := []byte(fmt.Sprintf(`{
		"model":"glm-5.3-flash:cloud",
		"client_metadata":{"x-codex-turn-metadata":%q},
		"tools":[
			{"type":"function","name":"exec_command","parameters":{"type":"object","properties":{"cmd":{"type":"string"},"sandbox_permissions":{"type":"string"},"justification":{"type":"string"},"prefix_rule":{"type":"array"}},"required":["cmd","sandbox_permissions","justification","prefix_rule"],"additionalProperties":false}},
			{"type":"function","name":"other_tool","parameters":{"type":"object","properties":{"sandbox_permissions":{"type":"string"}}}}
		]
	}`, turnMetadata))

	normalized, err := normalizeFullAccessExecTool(body)
	if err != nil {
		t.Fatal(err)
	}
	var payload struct {
		Tools []struct {
			Name       string `json:"name"`
			Parameters struct {
				Properties map[string]json.RawMessage `json:"properties"`
				Required   []string                   `json:"required"`
			} `json:"parameters"`
		} `json:"tools"`
	}
	if err := json.Unmarshal(normalized, &payload); err != nil {
		t.Fatal(err)
	}
	if len(payload.Tools) != 2 {
		t.Fatalf("tools = %d, want 2", len(payload.Tools))
	}
	execTool := payload.Tools[0]
	if _, ok := execTool.Parameters.Properties["cmd"]; !ok {
		t.Fatal("exec_command cmd property was removed")
	}
	for _, property := range []string{"sandbox_permissions", "justification", "prefix_rule"} {
		if _, ok := execTool.Parameters.Properties[property]; ok {
			t.Fatalf("exec_command retained %q property", property)
		}
		if slices.Contains(execTool.Parameters.Required, property) {
			t.Fatalf("exec_command retained required %q", property)
		}
	}
	if !slices.Equal(execTool.Parameters.Required, []string{"cmd"}) {
		t.Fatalf("exec_command required = %v, want [cmd]", execTool.Parameters.Required)
	}
	if _, ok := payload.Tools[1].Parameters.Properties["sandbox_permissions"]; !ok {
		t.Fatal("non-exec tool was changed")
	}
}

func TestNormalizeFullAccessExecToolLeavesSandboxedTurnUnchanged(t *testing.T) {
	body := []byte(`{"client_metadata":{"x-codex-turn-metadata":"{\"sandbox_mode\":\"workspace-write\"}"},"tools":[{"name":"exec_command","parameters":{"properties":{"sandbox_permissions":{"type":"string"}}}}]}`)
	normalized, err := normalizeFullAccessExecTool(body)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(normalized, body) {
		t.Fatalf("sandboxed request changed:\n%s", normalized)
	}
}

func TestLoadCatalogModelsReadsOptionalThinkingMetadata(t *testing.T) {
	path := filepath.Join(t.TempDir(), CodexDesktopModelCatalogFilename)
	data := []byte(`{"models":[{"slug":"legacy"},{"slug":"binary","thinking":{"supported":true,"levels":["none","medium"],"values":{"none":false,"medium":true}}}]}`)
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	catalog, err := loadRoutingCatalog(path)
	if err != nil {
		t.Fatal(err)
	}
	models := catalog.models
	if models["legacy"].Thinking != nil {
		t.Fatalf("legacy model thinking metadata = %+v, want absent", models["legacy"].Thinking)
	}
	binary := models["binary"].Thinking
	if binary == nil || !binary.Supported || strings.Join(binary.Levels, ",") != "none,medium" {
		t.Fatalf("binary model thinking metadata = %+v, want off/medium contract", binary)
	}
	if string(binary.Values["none"]) != "false" || string(binary.Values["medium"]) != "true" {
		t.Fatalf("binary model thinking values = %#v, want exact false/true values", binary.Values)
	}
}

func TestCodexDesktopRoutesAutoReviewToSelectedNativeModel(t *testing.T) {
	ollama := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Fatal("native selected model should not reach Ollama")
	}))
	defer ollama.Close()

	var models []string
	var encodings []string
	var guardianBody map[string]json.RawMessage
	chatGPT := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload map[string]json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatal(err)
		}
		var model string
		if err := json.Unmarshal(payload["model"], &model); err != nil {
			t.Fatal(err)
		}
		models = append(models, model)
		encodings = append(encodings, r.Header.Get("Content-Encoding"))
		if len(models) == 2 {
			guardianBody = payload
		}
		w.WriteHeader(http.StatusNoContent)
	}))
	defer chatGPT.Close()

	handler := newTestCodexDesktop(t, ollama.URL, chatGPT.URL, writeCatalogWithSelectedAutoReview(t, "glm-5.3-flash:cloud", "glm-5.3-flash:cloud"))
	proxy := httptest.NewServer(handler)
	defer proxy.Close()

	postCodexRequest(t, proxy.URL, `{"model":"gpt-5.6-sol","client_metadata":{"turn_id":"turn-native"}}`, true)
	postCompressedCodexRequest(t, proxy.URL, `{"model":"codex-auto-review","client_metadata":{"parent_turn_id":"turn-native"},"text":{"format":{"type":"json_schema"}}}`)

	if strings.Join(models, ",") != "gpt-5.6-sol,gpt-5.6-sol" {
		t.Fatalf("forwarded models = %q, want selected native model for Guardian", models)
	}
	if _, ok := guardianBody["text"]; !ok {
		t.Fatalf("native Guardian structured output was removed: %#v", guardianBody)
	}
	if _, ok := guardianBody["tools"]; ok {
		t.Fatalf("proxy decision tool was added to native Guardian: %#v", guardianBody)
	}
	if encodings[1] != "" {
		t.Fatalf("rewritten native Guardian retained content encoding %q", encodings[1])
	}
}

func TestCodexDesktopRoutesAutoReviewToSelectedOllamaModel(t *testing.T) {
	var models []string
	var guardianBody map[string]json.RawMessage
	ollama := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload map[string]json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatal(err)
		}
		var model string
		if err := json.Unmarshal(payload["model"], &model); err != nil {
			t.Fatal(err)
		}
		models = append(models, model)
		if len(models) == 2 {
			guardianBody = payload
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, autoReviewJSONResponse(`{"risk_level":"low","user_authorization":"high","outcome":"allow","rationale":"Allowed."}`))
			return
		}
		w.WriteHeader(http.StatusNoContent)
	}))
	defer ollama.Close()
	chatGPT := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Fatal("Ollama selected model should not reach ChatGPT")
	}))
	defer chatGPT.Close()

	handler := newTestCodexDesktop(t, ollama.URL, chatGPT.URL, writeCatalogWithSelectedAutoReview(t, "glm-5.3-flash:cloud", "glm-5.3-flash:cloud", "deepseek-v3.1:671b-cloud"))
	proxy := httptest.NewServer(handler)
	defer proxy.Close()

	postCodexRequest(t, proxy.URL, `{"model":"deepseek-v3.1:671b-cloud","client_metadata":{"turn_id":"turn-ollama"}}`, false)
	postCodexRequest(t, proxy.URL, `{"model":"codex-auto-review","client_metadata":{"parent_turn_id":"turn-ollama"},"input":[{"role":"user","content":"review"}],"text":{"format":{"type":"json_schema"}}}`, false)

	if strings.Join(models, ",") != "deepseek-v3.1:671b-cloud,deepseek-v3.1:671b-cloud" {
		t.Fatalf("forwarded models = %q, want selected Ollama model for Guardian", models)
	}
	if _, ok := guardianBody["text"]; ok {
		t.Fatalf("Ollama Guardian retained competing structured output: %#v", guardianBody)
	}
	if _, ok := guardianBody["tools"]; !ok {
		t.Fatalf("Ollama Guardian decision tool is missing: %#v", guardianBody)
	}
}

func TestCodexDesktopSelectedAutoReviewFallsBackWithoutParentTurn(t *testing.T) {
	var gotModel string
	ollama := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload struct {
			Model string `json:"model"`
		}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatal(err)
		}
		gotModel = payload.Model
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, autoReviewJSONResponse(`{"risk_level":"low","user_authorization":"high","outcome":"allow","rationale":"Allowed."}`))
	}))
	defer ollama.Close()
	chatGPT := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Fatal("selected Auto-review fallback should not reach ChatGPT")
	}))
	defer chatGPT.Close()

	handler := newTestCodexDesktop(t, ollama.URL, chatGPT.URL, writeCatalogWithSelectedAutoReview(t, "glm-5.3-flash:cloud", "glm-5.3-flash:cloud"))
	proxy := httptest.NewServer(handler)
	defer proxy.Close()

	postCodexRequest(t, proxy.URL, `{"model":"codex-auto-review","input":[{"role":"user","content":"review"}]}`, false)
	if gotModel != "glm-5.3-flash:cloud" {
		t.Fatalf("fallback model = %q, want launch primary", gotModel)
	}
}

func TestTurnModelCacheIsBounded(t *testing.T) {
	var cache turnModelCache
	for i := 0; i <= maxTrackedTurnModels; i++ {
		cache.remember(fmt.Sprintf("turn-%d", i), fmt.Sprintf("model-%d", i))
	}
	if _, ok := cache.lookup("turn-0"); ok {
		t.Fatal("oldest turn was not evicted")
	}
	if model, ok := cache.lookup(fmt.Sprintf("turn-%d", maxTrackedTurnModels)); !ok || model != fmt.Sprintf("model-%d", maxTrackedTurnModels) {
		t.Fatalf("newest turn = %q, %v", model, ok)
	}
}

func TestCodexDesktopRoutesAutoReviewToConfiguredOllamaModel(t *testing.T) {
	var gotBody []byte
	ollama := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, autoReviewJSONResponse(`{"risk_level":"low","user_authorization":"high","outcome":"allow","rationale":"The user requested this action."}`))
	}))
	defer ollama.Close()

	chatGPT := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Fatal("configured Auto-review request should not reach ChatGPT")
	}))
	defer chatGPT.Close()

	handler := newTestCodexDesktop(t, ollama.URL, chatGPT.URL, writeCatalogWithAutoReview(t, "glm-5.3-flash:cloud", "glm-5.3-flash:cloud"))
	proxy := httptest.NewServer(handler)
	defer proxy.Close()

	resp, err := http.Post(
		proxy.URL+CodexDesktopPathPrefix+"/v1/responses",
		"application/json",
		strings.NewReader(`{"model":"codex-auto-review","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"review"}]}],"text":{"format":{"type":"json_schema"}},"response_format":{"type":"json_object"}}`),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", resp.StatusCode)
	}

	var forwarded struct {
		Model string `json:"model"`
		Tools []struct {
			Type       string `json:"type"`
			Name       string `json:"name"`
			Strict     bool   `json:"strict"`
			Parameters struct {
				Required             []string `json:"required"`
				AdditionalProperties bool     `json:"additionalProperties"`
			} `json:"parameters"`
		} `json:"tools"`
		Input []struct {
			Content []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"content"`
		} `json:"input"`
	}
	if err := json.Unmarshal(gotBody, &forwarded); err != nil {
		t.Fatal(err)
	}
	if forwarded.Model != "glm-5.3-flash:cloud" {
		t.Fatalf("forwarded model = %q, want selected Ollama model", forwarded.Model)
	}
	if len(forwarded.Input) != 1 || len(forwarded.Input[0].Content) != 2 {
		t.Fatalf("forwarded input = %#v", forwarded.Input)
	}
	instruction := forwarded.Input[0].Content[1]
	if instruction.Type != "input_text" || !strings.Contains(instruction.Text, "call submit_guardian_decision exactly once") {
		t.Fatalf("Auto-review decision instruction = %#v", instruction)
	}
	if len(forwarded.Tools) != 1 || forwarded.Tools[0].Type != "function" || forwarded.Tools[0].Name != guardianDecisionToolName || !forwarded.Tools[0].Strict {
		t.Fatalf("Auto-review tools = %#v", forwarded.Tools)
	}
	if len(forwarded.Tools[0].Parameters.Required) != 4 || forwarded.Tools[0].Parameters.AdditionalProperties {
		t.Fatalf("Auto-review tool schema = %#v", forwarded.Tools[0].Parameters)
	}

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(gotBody, &raw); err != nil {
		t.Fatal(err)
	}
	if _, ok := raw["response_format"]; ok {
		t.Fatalf("Auto-review request unexpectedly used structured outputs: %s", gotBody)
	}
	if _, ok := raw["text"]; ok {
		t.Fatalf("Auto-review request unexpectedly added a structured text format: %s", gotBody)
	}

	var response struct {
		Output []struct {
			Type    string `json:"type"`
			Content []struct {
				Text string `json:"text"`
			} `json:"content"`
		} `json:"output"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		t.Fatal(err)
	}
	if len(response.Output) != 1 || response.Output[0].Type != "message" || len(response.Output[0].Content) != 1 || response.Output[0].Content[0].Text != `{"risk_level":"low","user_authorization":"high","outcome":"allow","rationale":"The user requested this action."}` {
		t.Fatalf("translated Auto-review response = %#v", response.Output)
	}
}

func TestPrepareAutoReviewRequestSupportsStringContentAndPreservesTools(t *testing.T) {
	body := []byte(`{"model":"codex-auto-review","tools":[{"type":"function","name":"inspect","description":"Inspect","strict":true,"parameters":{"type":"object"}}],"input":[{"role":"user","content":"review"}]}`)
	updated, err := prepareAutoReviewRequest(body)
	if err != nil {
		t.Fatal(err)
	}
	var payload struct {
		Tools []struct {
			Name string `json:"name"`
		} `json:"tools"`
		Input []struct {
			Content string `json:"content"`
		} `json:"input"`
	}
	if err := json.Unmarshal(updated, &payload); err != nil {
		t.Fatal(err)
	}
	if len(payload.Input) != 1 || !strings.HasPrefix(payload.Input[0].Content, "review") || !strings.Contains(payload.Input[0].Content, guardianDecisionToolName) {
		t.Fatalf("updated input = %#v", payload.Input)
	}
	if len(payload.Tools) != 2 || payload.Tools[0].Name != "inspect" || payload.Tools[1].Name != guardianDecisionToolName {
		t.Fatalf("updated tools = %#v", payload.Tools)
	}
}

func TestPrepareAutoReviewRequestRejectsRequestsWithoutUserMessages(t *testing.T) {
	body := []byte(`{"model":"codex-auto-review","input":[{"role":"developer","content":"policy"}]}`)
	if _, err := prepareAutoReviewRequest(body); err == nil || !strings.Contains(err.Error(), "no user message") {
		t.Fatalf("error = %v, want missing user message", err)
	}
}

func TestPrepareAutoReviewRequestUsesLastUserMessage(t *testing.T) {
	body := []byte(`{"input":[{"role":"user","content":"earlier"},{"role":"assistant","content":"reviewed"},{"role":"user","content":"latest"}]}`)
	updated, err := prepareAutoReviewRequest(body)
	if err != nil {
		t.Fatal(err)
	}
	var payload struct {
		Input []struct {
			Content string `json:"content"`
		} `json:"input"`
	}
	if err := json.Unmarshal(updated, &payload); err != nil {
		t.Fatal(err)
	}
	if payload.Input[0].Content != "earlier" {
		t.Fatalf("earlier user message was changed: %q", payload.Input[0].Content)
	}
	if !strings.HasPrefix(payload.Input[2].Content, "latest") || !strings.Contains(payload.Input[2].Content, guardianDecisionToolName) {
		t.Fatalf("latest user message = %q", payload.Input[2].Content)
	}
}

func TestTransformAutoReviewEventStreamConvertsDecisionToolCall(t *testing.T) {
	arguments := `{"risk_level":"medium","user_authorization":"low","outcome":"deny","rationale":"The action exceeds the user's authorization."}`
	body := autoReviewEventStream(t, guardianDecisionToolName, arguments)
	transformed, changed, err := transformAutoReviewResponse(body, "text/event-stream; charset=utf-8")
	if err != nil {
		t.Fatal(err)
	}
	if !changed {
		t.Fatal("decision tool call was not transformed")
	}
	text := string(transformed)
	if strings.Contains(text, "response.function_call_arguments") || strings.Contains(text, `"type":"function_call"`) {
		t.Fatalf("transformed stream still contains the decision tool call:\n%s", text)
	}
	for _, want := range []string{
		"event: response.output_text.delta",
		"event: response.output_text.done",
		`\"outcome\":\"deny\"`,
		`"type":"message"`,
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("transformed stream missing %q:\n%s", want, text)
		}
	}

	events, err := parseServerSentEvents(transformed)
	if err != nil {
		t.Fatal(err)
	}
	for i, event := range events {
		if bytes.Equal(bytes.TrimSpace(event.data), []byte("[DONE]")) {
			continue
		}
		var payload struct {
			Sequence int `json:"sequence_number"`
		}
		if err := json.Unmarshal(event.data, &payload); err != nil {
			t.Fatal(err)
		}
		if payload.Sequence != i {
			t.Fatalf("event %d sequence_number = %d", i, payload.Sequence)
		}
	}
}

func TestTransformAutoReviewResponsePassesThroughInvestigativeToolCall(t *testing.T) {
	body := []byte(autoReviewJSONResponseForTool("inspect_command", `{"command":"rm -rf tmp"}`))
	transformed, changed, err := transformAutoReviewResponse(body, "application/json")
	if err != nil {
		t.Fatal(err)
	}
	if changed || !bytes.Equal(transformed, body) {
		t.Fatalf("investigative tool call changed: %s", transformed)
	}
}

func TestTransformAutoReviewResponseRejectsInvalidDecision(t *testing.T) {
	body := []byte(autoReviewJSONResponse(`{"risk_level":"low","user_authorization":"high","outcome":"maybe","rationale":"Invalid outcome."}`))
	if _, _, err := transformAutoReviewResponse(body, "application/json"); err == nil || !strings.Contains(err.Error(), "invalid Guardian outcome") {
		t.Fatalf("error = %v, want invalid outcome", err)
	}
}

func TestTransformAutoReviewResponseRejectsTextDecision(t *testing.T) {
	body := []byte(`{"id":"resp_1","status":"completed","output":[{"id":"msg_1","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"{\\\"outcome\\\":\\\"deny\\\"}"}]}]}`)
	if _, _, err := transformAutoReviewResponse(body, "application/json"); err == nil || !strings.Contains(err.Error(), "did not call") {
		t.Fatalf("error = %v, want missing decision tool call", err)
	}
}

func TestTransformAutoReviewResponseAcceptsValidatedTextFallback(t *testing.T) {
	decision := `{"risk_level":"low","user_authorization":"high","outcome":"allow","rationale":"The user requested this benign local action."}`
	jsonBody := []byte(fmt.Sprintf(`{"id":"resp_1","status":"completed","output":[{"id":"msg_1","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":%q}]}]}`, decision))
	for _, test := range []struct {
		name        string
		contentType string
		body        []byte
	}{
		{name: "json", contentType: "application/json", body: jsonBody},
		{name: "event stream", contentType: "text/event-stream", body: autoReviewTextEventStream(t, decision)},
	} {
		t.Run(test.name, func(t *testing.T) {
			transformed, changed, err := transformAutoReviewResponse(test.body, test.contentType)
			if err != nil {
				t.Fatal(err)
			}
			if changed || !bytes.Equal(transformed, test.body) {
				t.Fatalf("validated text fallback changed:\n%s", transformed)
			}
		})
	}
}

func TestTransformAutoReviewResponseAcceptsCompactAllowTextFallback(t *testing.T) {
	decision := `{"outcome":"allow"}`
	jsonBody := []byte(fmt.Sprintf(`{"id":"resp_1","status":"completed","output":[{"id":"msg_1","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":%q}]}]}`, decision))
	for _, test := range []struct {
		name        string
		contentType string
		body        []byte
	}{
		{name: "json", contentType: "application/json", body: jsonBody},
		{name: "event stream", contentType: "text/event-stream", body: autoReviewTextEventStream(t, decision)},
	} {
		t.Run(test.name, func(t *testing.T) {
			transformed, changed, err := transformAutoReviewResponse(test.body, test.contentType)
			if err != nil {
				t.Fatal(err)
			}
			if changed || !bytes.Equal(transformed, test.body) {
				t.Fatalf("compact allow fallback changed:\n%s", transformed)
			}
		})
	}
}

func TestTransformAutoReviewJSONDiscardsProseWithValidDecision(t *testing.T) {
	arguments := `{"risk_level":"low","user_authorization":"high","outcome":"allow","rationale":"The command is read-only and explicitly requested."}`
	body := []byte(fmt.Sprintf(`{"id":"resp_1","status":"completed","output":[{"id":"fc_1","type":"function_call","status":"completed","call_id":"call_1","name":%q,"arguments":%q},{"id":"msg_1","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"Submitting approval."}]}]}`, guardianDecisionToolName, arguments))
	transformed, changed, err := transformAutoReviewResponse(body, "application/json")
	if err != nil {
		t.Fatal(err)
	}
	if !changed {
		t.Fatal("mixed decision response was not transformed")
	}
	var response struct {
		Output []struct {
			ID      string `json:"id"`
			Type    string `json:"type"`
			Content []struct {
				Text string `json:"text"`
			} `json:"content"`
		} `json:"output"`
	}
	if err := json.Unmarshal(transformed, &response); err != nil {
		t.Fatal(err)
	}
	if len(response.Output) != 1 || response.Output[0].ID != "fc_1" || response.Output[0].Type != "message" || len(response.Output[0].Content) != 1 || response.Output[0].Content[0].Text != arguments {
		t.Fatalf("transformed output = %#v", response.Output)
	}
}

func TestTransformAutoReviewEventStreamDiscardsProseWithValidDecision(t *testing.T) {
	arguments := `{"risk_level":"low","user_authorization":"high","outcome":"allow","rationale":"The command is read-only and explicitly requested."}`
	extraMessage := map[string]any{
		"id": "msg_1", "type": "message", "status": "completed", "role": "assistant",
		"content": []any{map[string]any{"type": "output_text", "text": "Submitting approval.", "annotations": []any{}, "logprobs": []any{}}},
	}
	body := autoReviewEventStream(t, guardianDecisionToolName, arguments, extraMessage)
	transformed, changed, err := transformAutoReviewResponse(body, "text/event-stream")
	if err != nil {
		t.Fatal(err)
	}
	if !changed {
		t.Fatal("mixed decision stream was not transformed")
	}
	text := string(transformed)
	if strings.Contains(text, "Submitting approval") || strings.Contains(text, "msg_1") {
		t.Fatalf("transformed stream retained assistant prose:\n%s", text)
	}
	events, err := parseServerSentEvents(transformed)
	if err != nil {
		t.Fatal(err)
	}
	var completed map[string]any
	for _, event := range events {
		if event.event == "response.completed" {
			if err := json.Unmarshal(event.data, &completed); err != nil {
				t.Fatal(err)
			}
		}
	}
	response := completed["response"].(map[string]any)
	output := response["output"].([]any)
	if len(output) != 1 || itemString(output[0].(map[string]any), "id") != "fc_1" || itemString(output[0].(map[string]any), "type") != "message" {
		t.Fatalf("completed output = %#v", output)
	}
}

func TestTransformAutoReviewResponseRejectsDecisionWithAnotherToolCall(t *testing.T) {
	decision := `{"risk_level":"low","user_authorization":"high","outcome":"allow","rationale":"Allowed."}`
	body := []byte(fmt.Sprintf(`{"id":"resp_1","status":"completed","output":[{"id":"fc_1","type":"function_call","name":%q,"arguments":%q},{"id":"fc_2","type":"function_call","name":"inspect_command","arguments":"{}"}]}`, guardianDecisionToolName, decision))
	if _, _, err := transformAutoReviewResponse(body, "application/json"); err == nil || !strings.Contains(err.Error(), "other terminal output") {
		t.Fatalf("error = %v, want mixed terminal output", err)
	}
}

func TestTransformAutoReviewResponsePreservesProviderFailure(t *testing.T) {
	body := []byte(`{"id":"resp_1","status":"failed","output":[],"error":{"code":"provider_error","message":"unavailable"}}`)
	transformed, changed, err := transformAutoReviewResponse(body, "application/json")
	if err != nil {
		t.Fatal(err)
	}
	if changed || !bytes.Equal(transformed, body) {
		t.Fatalf("provider failure changed: %s", transformed)
	}
}

func TestCodexDesktopKeepsAutoReviewOnChatGPTByDefault(t *testing.T) {
	ollama := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Fatal("native Auto-review request should not reach Ollama")
	}))
	defer ollama.Close()

	var gotModel string
	chatGPT := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload struct {
			Model string `json:"model"`
		}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatal(err)
		}
		gotModel = payload.Model
		w.WriteHeader(http.StatusOK)
	}))
	defer chatGPT.Close()

	handler := newTestCodexDesktop(t, ollama.URL, chatGPT.URL, writeCatalog(t, "glm-5.3-flash:cloud"))
	proxy := httptest.NewServer(handler)
	defer proxy.Close()

	req, err := http.NewRequest(http.MethodPost, proxy.URL+CodexDesktopPathPrefix+"/v1/responses", strings.NewReader(`{"model":"codex-auto-review"}`))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("ChatGPT-Account-ID", "account-123")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	if gotModel != autoReviewModel {
		t.Fatalf("forwarded model = %q, want native reviewer alias", gotModel)
	}
}

func TestCodexDesktopRejectsAutoReviewModelOutsideRoutingCatalog(t *testing.T) {
	called := false
	upstream := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		called = true
	}))
	defer upstream.Close()

	handler := newTestCodexDesktop(t, upstream.URL, upstream.URL, writeCatalogWithAutoReview(t, "qwen3:8b", "glm-5.3-flash:cloud"))
	req := httptest.NewRequest(http.MethodPost, "http://localhost"+CodexDesktopPathPrefix+"/v1/responses", strings.NewReader(`{"model":"codex-auto-review"}`))
	req.RemoteAddr = "127.0.0.1:1234"
	recorder := httptest.NewRecorder()

	handler.ServeHTTP(recorder, req)
	if recorder.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503: %s", recorder.Code, recorder.Body.String())
	}
	if called {
		t.Fatal("invalid Auto-review configuration reached an upstream")
	}
}

func TestCodexDesktopNormalizesCodexOnlyHistoryForOllama(t *testing.T) {
	var gotBody []byte
	ollama := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotBody, _ = io.ReadAll(r.Body)
		w.WriteHeader(http.StatusOK)
	}))
	defer ollama.Close()
	chatGPT := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Fatal("Ollama model should not reach ChatGPT")
	}))
	defer chatGPT.Close()

	handler := newTestCodexDesktop(t, ollama.URL, chatGPT.URL, writeCatalog(t, "glm-5.2:cloud"))
	proxy := httptest.NewServer(handler)
	defer proxy.Close()

	payload := `{
		"model":"glm-5.2:cloud",
		"stream":true,
		"input":[
			{"type":"compaction","encrypted_content":"opaque"},
			{"type":"message","role":"developer","content":[{"type":"input_text","text":"<collaboration_mode>Plan Mode</collaboration_mode>"}]},
			{"type":"custom_tool_call","id":"ctc_1","status":"completed","call_id":"call_1","name":"apply_patch","input":"*** Begin Patch"},
			{"type":"custom_tool_call_output","call_id":"call_1","output":"Success"},
			{"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}]},
			{"type":"future_codex_item","secret":"ignored"}
		]
	}`
	resp, err := http.Post(proxy.URL+CodexDesktopPathPrefix+"/v1/responses", "application/json", strings.NewReader(payload))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", resp.StatusCode)
	}

	var forwarded struct {
		Input []map[string]any `json:"input"`
	}
	if err := json.Unmarshal(gotBody, &forwarded); err != nil {
		t.Fatal(err)
	}
	if len(forwarded.Input) != 4 {
		t.Fatalf("forwarded input = %#v", forwarded.Input)
	}
	developer := forwarded.Input[0]
	if developer["type"] != "message" || developer["role"] != "system" {
		t.Fatalf("developer instructions were not promoted for Ollama: %#v", developer)
	}
	call := forwarded.Input[1]
	if call["type"] != "function_call" || call["call_id"] != "call_1" || call["name"] != "apply_patch" {
		t.Fatalf("converted call = %#v", call)
	}
	if call["arguments"] != `{"input":"*** Begin Patch"}` {
		t.Fatalf("converted arguments = %#v", call["arguments"])
	}
	output := forwarded.Input[2]
	if output["type"] != "function_call_output" || output["call_id"] != "call_1" || output["output"] != "Success" {
		t.Fatalf("converted output = %#v", output)
	}
	if forwarded.Input[3]["type"] != "message" || forwarded.Input[3]["role"] != "user" {
		t.Fatalf("user message was not preserved: %#v", forwarded.Input[3])
	}
}

func TestCodexDesktopPreservesToolSearchAndOllamaCompactionForOllama(t *testing.T) {
	var gotBody []byte
	ollama := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotBody, _ = io.ReadAll(r.Body)
		w.WriteHeader(http.StatusOK)
	}))
	defer ollama.Close()
	chatGPT := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Fatal("Ollama model should not reach ChatGPT")
	}))
	defer chatGPT.Close()

	handler := newTestCodexDesktop(t, ollama.URL, chatGPT.URL, writeCatalog(t, "glm-5.3-flash:cloud"))
	proxy := httptest.NewServer(handler)
	defer proxy.Close()

	payload := `{
		"model":"glm-5.3-flash:cloud",
		"input":[
			{"type":"compaction","encrypted_content":"native-opaque-state"},
			{"type":"compaction","encrypted_content":"{\"type\":\"ollama_compaction\",\"version\":1,\"summary\":\"summary\",\"retained\":[]}"},
			{"type":"tool_search_call","id":"ts_1","call_id":"call_search","execution":"client","status":"completed","arguments":{"query":"notion"}},
			{"type":"tool_search_output","id":"tso_1","call_id":"call_search","execution":"client","status":"completed","tools":[{"type":"function","name":"notion.search"}]},
			{"type":"message","role":"user","content":"continue"},
			{"type":"compaction_trigger"}
		]
	}`
	resp, err := http.Post(proxy.URL+CodexDesktopPathPrefix+"/v1/responses", "application/json", strings.NewReader(payload))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", resp.StatusCode)
	}

	var forwarded struct {
		Input []map[string]any `json:"input"`
	}
	if err := json.Unmarshal(gotBody, &forwarded); err != nil {
		t.Fatal(err)
	}
	if len(forwarded.Input) != 5 {
		t.Fatalf("forwarded input = %#v", forwarded.Input)
	}
	wantTypes := []string{"compaction", "tool_search_call", "tool_search_output", "message", "compaction_trigger"}
	for i, want := range wantTypes {
		if got := forwarded.Input[i]["type"]; got != want {
			t.Fatalf("forwarded input[%d] type = %#v, want %q", i, got, want)
		}
	}
}

func TestCodexDesktopFiltersNativeReasoningWhenSwitchingToOllama(t *testing.T) {
	var gotBody []byte
	ollama := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotBody, _ = io.ReadAll(r.Body)
		w.WriteHeader(http.StatusOK)
	}))
	defer ollama.Close()
	chatGPT := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Fatal("Ollama model should not reach ChatGPT")
	}))
	defer chatGPT.Close()

	handler := newTestCodexDesktop(t, ollama.URL, chatGPT.URL, writeCatalog(t, "glm-5.3-flash:cloud"))
	proxy := httptest.NewServer(handler)
	defer proxy.Close()

	payload := `{
		"model":"glm-5.3-flash:cloud",
		"input":[
			{"type":"reasoning","id":"rs_098c6fb068ce51bf016a9709ab7dcc87d185ecc21991f0f39c","encrypted_content":"gAAAAAB-native"},
			{"type":"reasoning","id":"rs_713083","encrypted_content":"Ollama plaintext thinking"},
			{"type":"reasoning","id":"rs_resp_123456","encrypted_content":"More Ollama thinking"},
			{"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}]}
		]
	}`
	resp, err := http.Post(proxy.URL+CodexDesktopPathPrefix+"/v1/responses", "application/json", strings.NewReader(payload))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", resp.StatusCode)
	}

	var forwarded struct {
		Input []map[string]any `json:"input"`
	}
	if err := json.Unmarshal(gotBody, &forwarded); err != nil {
		t.Fatal(err)
	}
	if len(forwarded.Input) != 3 {
		t.Fatalf("forwarded input = %#v", forwarded.Input)
	}
	if forwarded.Input[0]["id"] != "rs_713083" || forwarded.Input[1]["id"] != "rs_resp_123456" {
		t.Fatalf("Ollama reasoning was not preserved: %#v", forwarded.Input)
	}
	if forwarded.Input[2]["type"] != "message" {
		t.Fatalf("message was not preserved: %#v", forwarded.Input[2])
	}
}

func TestCodexDesktopPassesNativeModelToChatGPT(t *testing.T) {
	ollama := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Fatal("native model should not reach Ollama")
	}))
	defer ollama.Close()

	var gotPath, gotAuthorization, gotMetadata string
	var gotBody []byte
	chatGPT := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotAuthorization = r.Header.Get("Authorization")
		gotMetadata = r.Header.Get("X-Codex-Turn-Metadata")
		gotBody, _ = io.ReadAll(r.Body)
		w.WriteHeader(http.StatusAccepted)
		_, _ = io.WriteString(w, "native")
	}))
	defer chatGPT.Close()

	handler := newTestCodexDesktop(t, ollama.URL, chatGPT.URL+"/backend-api/codex", writeCatalog(t, "glm-5.2:cloud"))
	proxy := httptest.NewServer(handler)
	defer proxy.Close()

	payload := []byte(`{"model":"gpt-5.6-sol","stream":true,"input":[{"type":"message","role":"developer","content":[{"type":"input_text","text":"native instructions"}]}]}`)
	req, err := http.NewRequest(http.MethodPost, proxy.URL+CodexDesktopPathPrefix+"/v1/responses", bytes.NewReader(payload))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Authorization", "Bearer chatgpt-secret")
	req.Header.Set("ChatGPT-Account-ID", "account-123")
	req.Header.Set("X-Codex-Turn-Metadata", `{"thread":"kept"}`)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusAccepted {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	if gotPath != "/backend-api/codex/responses" {
		t.Fatalf("ChatGPT path = %q", gotPath)
	}
	if gotAuthorization != "Bearer chatgpt-secret" || gotMetadata != `{"thread":"kept"}` {
		t.Fatalf("native headers were not preserved: authorization=%q metadata=%q", gotAuthorization, gotMetadata)
	}
	if string(gotBody) != string(payload) {
		t.Fatalf("ChatGPT body = %q, want %q", gotBody, payload)
	}
}

func TestCodexDesktopPassesNativeModelToOpenAIAPIWithAPIKey(t *testing.T) {
	ollama := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Fatal("native model should not reach Ollama")
	}))
	defer ollama.Close()
	chatGPT := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Fatal("API-key request should not reach the ChatGPT subscription endpoint")
	}))
	defer chatGPT.Close()

	var gotPath, gotAuthorization, gotOrganization string
	openAI := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotAuthorization = r.Header.Get("Authorization")
		gotOrganization = r.Header.Get("OpenAI-Organization")
		w.WriteHeader(http.StatusAccepted)
	}))
	defer openAI.Close()

	handler, err := NewCodexDesktop(CodexDesktopConfig{
		OllamaURL:          ollama.URL,
		ChatGPTURL:         chatGPT.URL + "/backend-api/codex",
		OpenAIURL:          openAI.URL + "/v1",
		RoutingCatalogPath: writeCatalog(t, "glm-5.2:cloud"),
	})
	if err != nil {
		t.Fatal(err)
	}
	proxy := httptest.NewServer(handler)
	defer proxy.Close()

	req, err := http.NewRequest(
		http.MethodPost,
		proxy.URL+CodexDesktopPathPrefix+"/v1/responses",
		strings.NewReader(`{"model":"gpt-5.6-sol"}`),
	)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Authorization", "Bearer sk-test")
	req.Header.Set("OpenAI-Organization", "org-test")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusAccepted {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	if gotPath != "/v1/responses" {
		t.Fatalf("OpenAI API path = %q", gotPath)
	}
	if gotAuthorization != "Bearer sk-test" || gotOrganization != "org-test" {
		t.Fatalf("OpenAI API headers were not preserved: authorization=%q organization=%q", gotAuthorization, gotOrganization)
	}
}

func TestCodexDesktopRejectsManagedAPIKeyForNativeModel(t *testing.T) {
	ollamaCalled := false
	ollama := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		ollamaCalled = true
	}))
	defer ollama.Close()
	chatGPTCalled := false
	chatGPT := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		chatGPTCalled = true
	}))
	defer chatGPT.Close()
	openAICalled := false
	openAI := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		openAICalled = true
	}))
	defer openAI.Close()

	handler, err := NewCodexDesktop(CodexDesktopConfig{
		OllamaURL:          ollama.URL,
		ChatGPTURL:         chatGPT.URL + "/backend-api/codex",
		OpenAIURL:          openAI.URL + "/v1",
		RoutingCatalogPath: writeCatalog(t, "glm-5.2:cloud"),
	})
	if err != nil {
		t.Fatal(err)
	}
	proxy := httptest.NewServer(handler)
	defer proxy.Close()

	req, err := http.NewRequest(
		http.MethodPost,
		proxy.URL+CodexDesktopPathPrefix+"/v1/responses",
		strings.NewReader(`{"model":"gpt-5.6-sol"}`),
	)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Authorization", "Bearer "+CodexDesktopManagedAPIKey)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("status = %d, want 401: %s", resp.StatusCode, body)
	}
	if !strings.Contains(string(body), "OpenAI models require signing in") {
		t.Fatalf("body = %q, want sign-in recovery", body)
	}
	if ollamaCalled || chatGPTCalled || openAICalled {
		t.Fatalf("managed API key escaped local rejection: ollama=%v chatgpt=%v openai=%v", ollamaCalled, chatGPTCalled, openAICalled)
	}
}

func TestCodexDesktopPreservesCompressedNativeRequestWithoutOllamaReasoning(t *testing.T) {
	ollama := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Fatal("native model should not reach Ollama")
	}))
	defer ollama.Close()

	var gotEncoding string
	var gotBody []byte
	chatGPT := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotEncoding = r.Header.Get("Content-Encoding")
		gotBody, _ = io.ReadAll(r.Body)
		w.WriteHeader(http.StatusOK)
	}))
	defer chatGPT.Close()

	handler := newTestCodexDesktop(t, ollama.URL, chatGPT.URL+"/backend-api/codex", writeCatalog(t, "glm-5.3-flash:cloud"))
	proxy := httptest.NewServer(handler)
	defer proxy.Close()

	payload := []byte(`{"model":"gpt-5.6-sol","input":[{"type":"reasoning","id":"rs_098c6fb068ce51bf016a9709ab7dcc87d185ecc21991f0f39c","encrypted_content":"gAAAAAB-native"}]}`)
	encoder, err := zstd.NewWriter(nil)
	if err != nil {
		t.Fatal(err)
	}
	compressed := encoder.EncodeAll(payload, nil)
	encoder.Close()

	req, err := http.NewRequest(http.MethodPost, proxy.URL+CodexDesktopPathPrefix+"/v1/responses", bytes.NewReader(compressed))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Content-Encoding", "zstd")
	req.Header.Set("ChatGPT-Account-ID", "account-123")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	if gotEncoding != "zstd" {
		t.Fatalf("content encoding = %q, want zstd", gotEncoding)
	}
	if !bytes.Equal(gotBody, compressed) {
		t.Fatal("native request body was rewritten")
	}
}

func TestCodexDesktopFiltersOllamaProviderStateWhenSwitchingToNativeModel(t *testing.T) {
	ollama := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Fatal("native model should not reach Ollama")
	}))
	defer ollama.Close()

	var gotAuthorization, gotEncoding string
	var gotBody []byte
	chatGPT := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuthorization = r.Header.Get("Authorization")
		gotEncoding = r.Header.Get("Content-Encoding")
		gotBody, _ = io.ReadAll(r.Body)
		w.WriteHeader(http.StatusOK)
	}))
	defer chatGPT.Close()

	handler := newTestCodexDesktop(t, ollama.URL, chatGPT.URL+"/backend-api/codex", writeCatalog(t, "glm-5.3-flash:cloud"))
	proxy := httptest.NewServer(handler)
	defer proxy.Close()

	payload := []byte(`{
		"model":"gpt-5.6-sol",
		"input":[
			{"type":"reasoning","id":"rs_713083","encrypted_content":"The user wants info about the repo"},
			{"type":"reasoning","id":"rs_resp_123456","encrypted_content":"More plaintext thinking"},
			{"type":"compaction","encrypted_content":"{\"type\":\"ollama_compaction\",\"version\":1,\"summary\":\"local summary\",\"retained\":[]}"},
			{"type":"compaction","encrypted_content":"gAAAAAB-native-compaction"},
			{"type":"reasoning","id":"rs_098c6fb068ce51bf016a9709ab7dcc87d185ecc21991f0f39c","encrypted_content":"gAAAAAB-native"},
			{"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}]}
		]
	}`)
	encoder, err := zstd.NewWriter(nil)
	if err != nil {
		t.Fatal(err)
	}
	compressed := encoder.EncodeAll(payload, nil)
	encoder.Close()

	req, err := http.NewRequest(http.MethodPost, proxy.URL+CodexDesktopPathPrefix+"/v1/responses", bytes.NewReader(compressed))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Content-Encoding", "zstd")
	req.Header.Set("Authorization", "Bearer chatgpt-secret")
	req.Header.Set("ChatGPT-Account-ID", "account-123")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	if gotAuthorization != "Bearer chatgpt-secret" {
		t.Fatalf("authorization = %q", gotAuthorization)
	}
	if gotEncoding != "" {
		t.Fatalf("normalized body retained content encoding %q", gotEncoding)
	}

	var forwarded struct {
		Input []map[string]any `json:"input"`
	}
	if err := json.Unmarshal(gotBody, &forwarded); err != nil {
		t.Fatal(err)
	}
	if len(forwarded.Input) != 3 {
		t.Fatalf("forwarded input = %#v", forwarded.Input)
	}
	if forwarded.Input[0]["type"] != "compaction" ||
		forwarded.Input[0]["encrypted_content"] != "gAAAAAB-native-compaction" {
		t.Fatalf("native encrypted compaction was not preserved: %#v", forwarded.Input[0])
	}
	if forwarded.Input[1]["id"] != "rs_098c6fb068ce51bf016a9709ab7dcc87d185ecc21991f0f39c" ||
		forwarded.Input[1]["encrypted_content"] != "gAAAAAB-native" {
		t.Fatalf("native encrypted reasoning was not preserved: %#v", forwarded.Input[1])
	}
	if forwarded.Input[2]["type"] != "message" {
		t.Fatalf("message was not preserved: %#v", forwarded.Input[2])
	}
}

func TestIsOllamaReasoningItemID(t *testing.T) {
	for _, test := range []struct {
		id   string
		want bool
	}{
		{id: "rs_0", want: true},
		{id: "rs_713083", want: true},
		{id: "rs_resp_123456", want: true},
		{id: "rs_1234567", want: false},
		{id: "rs_resp_1234567", want: false},
		{id: "rs_098c6fb068ce51bf016a9709ab7dcc87d185ecc21991f0f39c", want: false},
		{id: "reasoning_123", want: false},
	} {
		if got := isOllamaReasoningItemID(test.id); got != test.want {
			t.Errorf("isOllamaReasoningItemID(%q) = %v, want %v", test.id, got, test.want)
		}
	}
}

func TestCodexDesktopRequestsHTTPFallbackForWebSocketUpgrade(t *testing.T) {
	ollama := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Fatal("WebSocket fallback should not reach Ollama")
	}))
	defer ollama.Close()
	chatGPT := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Fatal("WebSocket fallback should not reach ChatGPT")
	}))
	defer chatGPT.Close()

	handler := newTestCodexDesktop(t, ollama.URL, chatGPT.URL+"/backend-api/codex", writeCatalog(t, "glm-5.2:cloud"))
	proxy := httptest.NewServer(handler)
	defer proxy.Close()

	req, err := http.NewRequest(http.MethodGet, proxy.URL+CodexDesktopPathPrefix+"/v1/responses", nil)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Connection", "keep-alive, Upgrade")
	req.Header.Set("Upgrade", "websocket")
	req.Header.Set("Sec-WebSocket-Key", "dGhlIHNhbXBsZSBub25jZQ==")
	req.Header.Set("Sec-WebSocket-Version", "13")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusUpgradeRequired {
		t.Fatalf("status = %d, want 426", resp.StatusCode)
	}
}

func TestCodexDesktopStatusReportsObservedRoutes(t *testing.T) {
	ollama := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer ollama.Close()

	chatGPT := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusAccepted)
	}))
	defer chatGPT.Close()

	handler := newTestCodexDesktop(t, ollama.URL, chatGPT.URL+"/backend-api/codex", writeCatalog(t, "glm-5.2:cloud"))
	proxy := httptest.NewServer(handler)
	defer proxy.Close()

	for _, payload := range []string{
		`{"model":"glm-5.2:cloud"}`,
		`{"model":"gpt-5.6-sol"}`,
	} {
		req, err := http.NewRequest(http.MethodPost, proxy.URL+CodexDesktopPathPrefix+"/v1/responses", strings.NewReader(payload))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("ChatGPT-Account-ID", "account-123")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		_ = resp.Body.Close()
	}

	resp, err := http.Get(proxy.URL + CodexDesktopPathPrefix + "/_status")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var status statusResponse
	if err := json.NewDecoder(resp.Body).Decode(&status); err != nil {
		t.Fatal(err)
	}
	if !status.OK || status.OllamaRequests != 1 || status.ChatGPTRequests != 1 || status.UpstreamErrors != 0 {
		t.Fatalf("status = %+v", status)
	}
	if status.LastModel != "gpt-5.6-sol" || status.LastRoute != "chatgpt" || status.LastUpstreamStatus != http.StatusAccepted {
		t.Fatalf("last route = %+v", status)
	}
}

func TestCodexDesktopCountersExcludeProbesAndFailedRetries(t *testing.T) {
	ollama := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer ollama.Close()
	chatGPT := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusMethodNotAllowed)
	}))
	defer chatGPT.Close()

	handler := newTestCodexDesktop(t, ollama.URL, chatGPT.URL+"/backend-api/codex", writeCatalog(t, "glm-5.2:cloud"))
	proxy := httptest.NewServer(handler)
	defer proxy.Close()

	probeReq, err := http.NewRequest(http.MethodGet, proxy.URL+CodexDesktopPathPrefix+"/v1/responses", nil)
	if err != nil {
		t.Fatal(err)
	}
	probeReq.Header.Set("ChatGPT-Account-ID", "account-123")
	probeResp, err := http.DefaultClient.Do(probeReq)
	if err != nil {
		t.Fatal(err)
	}
	_ = probeResp.Body.Close()
	failedResp, err := http.Post(
		proxy.URL+CodexDesktopPathPrefix+"/v1/responses",
		"application/json",
		strings.NewReader(`{"model":"glm-5.2:cloud"}`),
	)
	if err != nil {
		t.Fatal(err)
	}
	_ = failedResp.Body.Close()

	statusResp, err := http.Get(proxy.URL + CodexDesktopPathPrefix + "/_status")
	if err != nil {
		t.Fatal(err)
	}
	defer statusResp.Body.Close()
	var status statusResponse
	if err := json.NewDecoder(statusResp.Body).Decode(&status); err != nil {
		t.Fatal(err)
	}
	if status.OllamaRequests != 0 || status.ChatGPTRequests != 0 {
		t.Fatalf("failed requests were counted: %+v", status)
	}
	if status.UpstreamErrors != 1 {
		t.Fatalf("upstream errors = %d, want 1", status.UpstreamErrors)
	}
}

func TestCodexDesktopWritesSafeActivityLog(t *testing.T) {
	ollama := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer ollama.Close()

	activityLogPath := filepath.Join(t.TempDir(), "logs", "codex-proxy.log")
	handler, err := NewCodexDesktop(CodexDesktopConfig{
		OllamaURL:          ollama.URL,
		ChatGPTURL:         "https://chatgpt.com/backend-api/codex",
		RoutingCatalogPath: writeCatalog(t, "glm-5.2:cloud"),
		ActivityLogPath:    activityLogPath,
	})
	if err != nil {
		t.Fatal(err)
	}
	proxy := httptest.NewServer(handler)
	defer proxy.Close()

	prompt := "do-not-record-this-prompt"
	req, err := http.NewRequest(
		http.MethodPost,
		proxy.URL+CodexDesktopPathPrefix+"/v1/responses",
		strings.NewReader(`{"model":"glm-5.2:cloud","input":"`+prompt+`"}`),
	)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Authorization", "Bearer do-not-record-this-token")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	_ = resp.Body.Close()

	// logActivity runs when the server finishes the request, which can be a
	// scheduler beat after the client has received the full response. Poll
	// for the log line instead of assuming it is already on disk.
	var logText string
	deadline := time.Now().Add(2 * time.Second)
	for {
		data, readErr := os.ReadFile(activityLogPath)
		if readErr == nil {
			logText = string(data)
			if strings.Contains(logText, `route=ollama model="glm-5.2:cloud"`) {
				break
			}
		}
		if time.Now().After(deadline) {
			if readErr != nil {
				t.Fatal(readErr)
			}
			t.Fatalf("activity log missing %q:\n%s", `route=ollama model="glm-5.2:cloud"`, logText)
		}
		time.Sleep(time.Millisecond)
	}
	for _, want := range []string{
		`route=ollama model="glm-5.2:cloud"`,
		"method=POST path=/v1/responses status=200",
		"result=ok",
	} {
		if !strings.Contains(logText, want) {
			t.Fatalf("activity log missing %q:\n%s", want, logText)
		}
	}
	for _, secret := range []string{prompt, "do-not-record-this-token", "Authorization"} {
		if strings.Contains(logText, secret) {
			t.Fatalf("activity log recorded sensitive value %q:\n%s", secret, logText)
		}
	}
}

func TestCodexDesktopRecordsMidstreamAbortWithoutPanicking(t *testing.T) {
	activityLogPath := filepath.Join(t.TempDir(), "codex-proxy.log")
	streamErr := errors.New("upstream stream failed")
	handler, err := NewCodexDesktop(CodexDesktopConfig{
		OllamaURL:          "http://127.0.0.1:11434",
		ChatGPTURL:         "https://chatgpt.com/backend-api/codex",
		RoutingCatalogPath: writeCatalog(t, "glm-5.3-flash:cloud"),
		ActivityLogPath:    activityLogPath,
		Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			return &http.Response{
				Status:     "200 OK",
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
				Body: io.NopCloser(io.MultiReader(
					strings.NewReader("data: partial\n\n"),
					errorReader{err: streamErr},
				)),
				Request: req,
			}, nil
		}),
	})
	if err != nil {
		t.Fatal(err)
	}

	req := httptest.NewRequest(
		http.MethodPost,
		"http://localhost"+CodexDesktopPathPrefix+"/v1/responses",
		strings.NewReader(`{"model":"glm-5.3-flash:cloud"}`),
	)
	req.RemoteAddr = "127.0.0.1:1234"
	req = req.WithContext(context.WithValue(req.Context(), http.ServerContextKey, &http.Server{}))
	recorder := httptest.NewRecorder()

	handler.ServeHTTP(recorder, req)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", recorder.Code)
	}
	if got := recorder.Body.String(); got != "data: partial\n\n" {
		t.Fatalf("body = %q, want partial event", got)
	}
	if got := handler.upstreamErrors.Load(); got != 1 {
		t.Fatalf("upstream errors = %d, want 1", got)
	}
	data, err := os.ReadFile(activityLogPath)
	if err != nil {
		t.Fatal(err)
	}
	logText := string(data)
	for _, want := range []string{
		`route=ollama model="glm-5.3-flash:cloud"`,
		"status=200",
		"result=stream_error",
	} {
		if !strings.Contains(logText, want) {
			t.Fatalf("activity log missing %q:\n%s", want, logText)
		}
	}
}

func TestCodexDesktopRecordsClientCanceledStream(t *testing.T) {
	activityLogPath := filepath.Join(t.TempDir(), "codex-proxy.log")
	handler, err := NewCodexDesktop(CodexDesktopConfig{
		OllamaURL:          "http://127.0.0.1:11434",
		ChatGPTURL:         "https://chatgpt.com/backend-api/codex",
		RoutingCatalogPath: writeCatalog(t, "glm-5.3-flash:cloud"),
		ActivityLogPath:    activityLogPath,
		Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			return &http.Response{
				Status:     "200 OK",
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
				Body:       io.NopCloser(strings.NewReader("data: ignored\n\n")),
				Request:    req,
			}, nil
		}),
	})
	if err != nil {
		t.Fatal(err)
	}

	req := httptest.NewRequest(
		http.MethodPost,
		"http://localhost"+CodexDesktopPathPrefix+"/v1/responses",
		strings.NewReader(`{"model":"glm-5.3-flash:cloud"}`),
	)
	req.RemoteAddr = "127.0.0.1:1234"
	ctx, cancel := context.WithCancel(req.Context())
	cancel()
	req = req.WithContext(ctx)
	recorder := &writeErrorResponseWriter{err: context.Canceled}

	handler.ServeHTTP(recorder, req)

	if recorder.status != http.StatusOK {
		t.Fatalf("status = %d, want 200", recorder.status)
	}
	if got := handler.upstreamErrors.Load(); got != 0 {
		t.Fatalf("upstream errors = %d, want 0", got)
	}
	data, err := os.ReadFile(activityLogPath)
	if err != nil {
		t.Fatal(err)
	}
	logText := string(data)
	if !strings.Contains(logText, "status=200") || !strings.Contains(logText, "result=canceled") {
		t.Fatalf("activity log did not record cancellation:\n%s", logText)
	}
}

func TestCodexDesktopRejectsNonLoopbackClients(t *testing.T) {
	handler := newTestCodexDesktop(t, "http://127.0.0.1:11434", "https://chatgpt.com/backend-api/codex", writeCatalog(t, "glm"))
	req := httptest.NewRequest(http.MethodGet, "http://example.test"+CodexDesktopPathPrefix+"/_health", nil)
	req.RemoteAddr = "192.0.2.10:1234"
	recorder := httptest.NewRecorder()

	handler.ServeHTTP(recorder, req)
	if recorder.Code != http.StatusForbidden {
		t.Fatalf("status = %d, want 403", recorder.Code)
	}
}

func TestCodexDesktopFailsClosedWhenCatalogIsMissing(t *testing.T) {
	handler := newTestCodexDesktop(t, "http://127.0.0.1:11434", "https://chatgpt.com/backend-api/codex", filepath.Join(t.TempDir(), "missing.json"))
	req := httptest.NewRequest(http.MethodPost, "http://localhost"+CodexDesktopPathPrefix+"/v1/responses", strings.NewReader(`{"model":"glm"}`))
	req.RemoteAddr = "127.0.0.1:1234"
	recorder := httptest.NewRecorder()

	handler.ServeHTTP(recorder, req)
	if recorder.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503: %s", recorder.Code, recorder.Body.String())
	}
}

func postCodexRequest(t *testing.T, proxyURL, body string, chatGPT bool) {
	t.Helper()
	req, err := http.NewRequest(http.MethodPost, proxyURL+CodexDesktopPathPrefix+"/v1/responses", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	if chatGPT {
		req.Header.Set("Authorization", "Bearer chatgpt-secret")
		req.Header.Set("ChatGPT-Account-ID", "account-123")
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		responseBody, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d: %s", resp.StatusCode, responseBody)
	}
}

func postCompressedCodexRequest(t *testing.T, proxyURL, body string) {
	t.Helper()
	encoder, err := zstd.NewWriter(nil)
	if err != nil {
		t.Fatal(err)
	}
	compressed := encoder.EncodeAll([]byte(body), nil)
	encoder.Close()
	req, err := http.NewRequest(http.MethodPost, proxyURL+CodexDesktopPathPrefix+"/v1/responses", bytes.NewReader(compressed))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Content-Encoding", "zstd")
	req.Header.Set("Authorization", "Bearer chatgpt-secret")
	req.Header.Set("ChatGPT-Account-ID", "account-123")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		responseBody, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d: %s", resp.StatusCode, responseBody)
	}
}

func newTestCodexDesktop(t *testing.T, ollamaURL, chatGPTURL, catalogPath string) *CodexDesktop {
	t.Helper()
	handler, err := NewCodexDesktop(CodexDesktopConfig{
		OllamaURL:          ollamaURL,
		ChatGPTURL:         chatGPTURL,
		RoutingCatalogPath: catalogPath,
	})
	if err != nil {
		t.Fatal(err)
	}
	return handler
}

func writeCatalog(t *testing.T, models ...string) string {
	return writeCatalogWithAutoReview(t, "", models...)
}

func writeCatalogWithAutoReview(t *testing.T, autoReviewModel string, models ...string) string {
	return writeRoutingCatalog(t, autoReviewModel, "", models...)
}

func writeCatalogWithSelectedAutoReview(t *testing.T, fallbackModel string, models ...string) string {
	return writeRoutingCatalog(t, autoReviewSelectedModel, fallbackModel, models...)
}

func writeRoutingCatalog(t *testing.T, autoReviewModel, fallbackModel string, models ...string) string {
	t.Helper()
	entries := make([]map[string]string, 0, len(models))
	for _, model := range models {
		entries = append(entries, map[string]string{"slug": model})
	}
	data, err := json.Marshal(map[string]any{
		"models":                     entries,
		"auto_review_model":          autoReviewModel,
		"auto_review_fallback_model": fallbackModel,
	})
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), CodexDesktopModelCatalogFilename)
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func autoReviewJSONResponse(arguments string) string {
	return autoReviewJSONResponseForTool(guardianDecisionToolName, arguments)
}

type errorReader struct {
	err error
}

func (r errorReader) Read([]byte) (int, error) {
	return 0, r.err
}

type writeErrorResponseWriter struct {
	header http.Header
	status int
	err    error
}

func (w *writeErrorResponseWriter) Header() http.Header {
	if w.header == nil {
		w.header = make(http.Header)
	}
	return w.header
}

func (w *writeErrorResponseWriter) WriteHeader(status int) {
	w.status = status
}

func (w *writeErrorResponseWriter) Write([]byte) (int, error) {
	return 0, w.err
}

func autoReviewJSONResponseForTool(name, arguments string) string {
	return fmt.Sprintf(`{"id":"resp_1","status":"completed","output":[{"id":"fc_1","type":"function_call","status":"completed","call_id":"call_1","name":%q,"arguments":%q}]}`, name, arguments)
}

func autoReviewEventStream(t *testing.T, name, arguments string, extraOutput ...map[string]any) []byte {
	t.Helper()
	inProgressItem := map[string]any{
		"id": "fc_1", "type": "function_call", "status": "in_progress", "call_id": "call_1", "name": name, "arguments": "",
	}
	completedItem := map[string]any{
		"id": "fc_1", "type": "function_call", "status": "completed", "call_id": "call_1", "name": name, "arguments": arguments,
	}
	events := []serverSentEvent{
		newServerSentEvent("response.created", map[string]any{"response": map[string]any{"id": "resp_1", "status": "in_progress", "output": []any{}}}),
		newServerSentEvent("response.output_item.added", map[string]any{"output_index": 0, "item": inProgressItem}),
		newServerSentEvent("response.function_call_arguments.delta", map[string]any{"item_id": "fc_1", "output_index": 0, "delta": arguments}),
		newServerSentEvent("response.function_call_arguments.done", map[string]any{"item_id": "fc_1", "output_index": 0, "arguments": arguments}),
		newServerSentEvent("response.output_item.done", map[string]any{"output_index": 0, "item": completedItem}),
	}
	output := []any{completedItem}
	for i, item := range extraOutput {
		outputIndex := i + 1
		itemID := itemString(item, "id")
		content := item["content"].([]any)[0].(map[string]any)
		text := itemString(content, "text")
		events = append(events,
			newServerSentEvent("response.output_item.added", map[string]any{"output_index": outputIndex, "item": map[string]any{"id": itemID, "type": "message", "status": "in_progress", "role": "assistant", "content": []any{}}}),
			newServerSentEvent("response.content_part.added", map[string]any{"item_id": itemID, "output_index": outputIndex, "content_index": 0, "part": map[string]any{"type": "output_text", "text": "", "annotations": []any{}, "logprobs": []any{}}}),
			newServerSentEvent("response.output_text.delta", map[string]any{"item_id": itemID, "output_index": outputIndex, "content_index": 0, "delta": text, "logprobs": []any{}}),
			newServerSentEvent("response.output_text.done", map[string]any{"item_id": itemID, "output_index": outputIndex, "content_index": 0, "text": text, "logprobs": []any{}}),
			newServerSentEvent("response.content_part.done", map[string]any{"item_id": itemID, "output_index": outputIndex, "content_index": 0, "part": content}),
			newServerSentEvent("response.output_item.done", map[string]any{"output_index": outputIndex, "item": item}),
		)
		output = append(output, item)
	}
	events = append(events, newServerSentEvent("response.completed", map[string]any{"response": map[string]any{"id": "resp_1", "status": "completed", "output": output}}))
	encoded, err := encodeServerSentEvents(events)
	if err != nil {
		t.Fatal(err)
	}
	return encoded
}

func autoReviewTextEventStream(t *testing.T, text string) []byte {
	t.Helper()
	inProgressItem := map[string]any{
		"id": "msg_1", "type": "message", "status": "in_progress", "role": "assistant", "content": []any{},
	}
	content := map[string]any{"type": "output_text", "text": text, "annotations": []any{}, "logprobs": []any{}}
	completedItem := map[string]any{
		"id": "msg_1", "type": "message", "status": "completed", "role": "assistant", "content": []any{content},
	}
	events := []serverSentEvent{
		newServerSentEvent("response.created", map[string]any{"response": map[string]any{"id": "resp_1", "status": "in_progress", "output": []any{}}}),
		newServerSentEvent("response.output_item.added", map[string]any{"output_index": 0, "item": inProgressItem}),
		newServerSentEvent("response.content_part.added", map[string]any{"item_id": "msg_1", "output_index": 0, "content_index": 0, "part": map[string]any{"type": "output_text", "text": "", "annotations": []any{}, "logprobs": []any{}}}),
		newServerSentEvent("response.output_text.delta", map[string]any{"item_id": "msg_1", "output_index": 0, "content_index": 0, "delta": text, "logprobs": []any{}}),
		newServerSentEvent("response.output_text.done", map[string]any{"item_id": "msg_1", "output_index": 0, "content_index": 0, "text": text, "logprobs": []any{}}),
		newServerSentEvent("response.content_part.done", map[string]any{"item_id": "msg_1", "output_index": 0, "content_index": 0, "part": content}),
		newServerSentEvent("response.output_item.done", map[string]any{"output_index": 0, "item": completedItem}),
		newServerSentEvent("response.completed", map[string]any{"response": map[string]any{"id": "resp_1", "status": "completed", "output": []any{completedItem}}}),
	}
	encoded, err := encodeServerSentEvents(events)
	if err != nil {
		t.Fatal(err)
	}
	return encoded
}

func TestCodexDesktopDoesNotAddForwardedHeaders(t *testing.T) {
	forwardedHeaders := func(r *http.Request) []string {
		var forwarded []string
		for _, key := range []string{"X-Forwarded-For", "X-Forwarded-Host", "X-Forwarded-Proto"} {
			forwarded = append(forwarded, r.Header.Values(key)...)
		}
		return forwarded
	}
	var ollamaForwarded, nativeForwarded []string
	ollama := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ollamaForwarded = forwardedHeaders(r)
		w.WriteHeader(http.StatusOK)
	}))
	defer ollama.Close()

	native := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		nativeForwarded = forwardedHeaders(r)
		w.WriteHeader(http.StatusOK)
	}))
	defer native.Close()

	catalogPath := writeCatalog(t, "glm-5.2:cloud")
	handler := newTestCodexDesktop(t, ollama.URL, native.URL, catalogPath)
	proxy := httptest.NewServer(handler)
	defer proxy.Close()

	for _, model := range []string{"glm-5.2:cloud", "gpt-native"} {
		req, err := http.NewRequest(http.MethodPost, proxy.URL+CodexDesktopPathPrefix+"/v1/responses", strings.NewReader(`{"model":"`+model+`"}`))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("ChatGPT-Account-ID", "account-123")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
	}
	if len(ollamaForwarded) != 0 {
		t.Fatalf("Ollama received forwarding headers: %q", ollamaForwarded)
	}
	if len(nativeForwarded) != 0 {
		t.Fatalf("native upstream received forwarding headers: %q", nativeForwarded)
	}
}

func TestCodexDesktopPassesThroughRedirectResponse(t *testing.T) {
	native := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Location", "https://example.com/moved")
		w.WriteHeader(http.StatusFound)
		_, _ = io.WriteString(w, "moved")
	}))
	defer native.Close()

	ollama := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Fatal("native model should not reach Ollama")
	}))
	defer ollama.Close()

	catalogPath := writeCatalog(t, "glm-5.2:cloud")
	handler := newTestCodexDesktop(t, ollama.URL, native.URL, catalogPath)
	proxy := httptest.NewServer(handler)
	defer proxy.Close()

	req, err := http.NewRequest(http.MethodPost, proxy.URL+CodexDesktopPathPrefix+"/v1/responses", strings.NewReader(`{"model":"gpt-native"}`))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("ChatGPT-Account-ID", "account-123")
	client := &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error {
		return http.ErrUseLastResponse
	}}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusFound || resp.Header.Get("Location") != "https://example.com/moved" || string(body) != "moved" {
		t.Fatalf("redirect passthrough = %d %q %q", resp.StatusCode, resp.Header.Get("Location"), body)
	}
}
