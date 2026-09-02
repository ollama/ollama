package codexproxy

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/klauspost/compress/zstd"
)

func TestHandlerRoutesCatalogModelToOllamaAndStripsCredentials(t *testing.T) {
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
	handler := newTestHandler(t, ollama.URL, chatGPT.URL+"/backend-api/codex", catalogPath)
	proxy := httptest.NewServer(handler)
	defer proxy.Close()

	payload := []byte(`{"model":"glm-5.2:cloud","stream":true,"reasoning":{"effort":"high"},"tools":[{"type":"web_search","external_web_access":false}]}`)
	encoder, err := zstd.NewWriter(nil)
	if err != nil {
		t.Fatal(err)
	}
	compressed := encoder.EncodeAll(payload, nil)
	encoder.Close()

	req, err := http.NewRequest(http.MethodPost, proxy.URL+PathPrefix+"/v1/responses?trace=1", bytes.NewReader(compressed))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Content-Encoding", "zstd")
	req.Header.Set("ChatGPT-Account-ID", "account-123")
	req.Header.Set("Authorization", "Bearer chatgpt-secret")
	req.Header.Set("X-Codex-Turn-Metadata", `{"thread":"secret"}`)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	responseBody, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusOK || string(responseBody) != "data: routed\n\n" {
		t.Fatalf("response = %d %q", resp.StatusCode, responseBody)
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

func TestNormalizeOllamaThinkingUsesRoutedModelContract(t *testing.T) {
	tests := []struct {
		name          string
		model         routingModel
		effort        string
		wantReasoning bool
		wantEffort    string
		omitEffort    bool
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
			}},
			effort:        "none",
			wantReasoning: true,
			wantEffort:    "none",
		},
		{
			name: "binary thinking uses medium for its enabled choice",
			model: routingModel{Thinking: &routingThinkingMetadata{
				Supported: true,
				Levels:    []string{"none", "medium"},
			}},
			effort:        "medium",
			wantReasoning: true,
			wantEffort:    "medium",
		},
		{
			name: "binary thinking maps a stale enabled level to medium",
			model: routingModel{Thinking: &routingThinkingMetadata{
				Supported: true,
				Levels:    []string{"none", "medium"},
			}},
			effort:        "high",
			wantReasoning: true,
			wantEffort:    "medium",
		},
		{
			name: "exact GLM ladder clamps xhigh to max",
			model: routingModel{Thinking: &routingThinkingMetadata{
				Supported: true,
				Levels:    []string{"low", "high", "max"},
			}},
			effort:        "xhigh",
			wantReasoning: true,
			wantEffort:    "max",
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
			}},
			effort:        "minimal",
			wantReasoning: true,
			wantEffort:    "low",
		},
		{
			name: "stale xhigh clamps to the strongest supported effort",
			model: routingModel{Thinking: &routingThinkingMetadata{
				Supported: true,
				Levels:    []string{"low", "medium", "high"},
			}},
			effort:        "xhigh",
			wantReasoning: true,
			wantEffort:    "high",
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

func TestLoadCatalogModelsReadsOptionalThinkingMetadata(t *testing.T) {
	path := filepath.Join(t.TempDir(), ModelCatalogFilename)
	data := []byte(`{"models":[{"slug":"legacy"},{"slug":"binary","thinking":{"supported":true,"levels":["none","medium"]}}]}`)
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	models, err := loadCatalogModels(path)
	if err != nil {
		t.Fatal(err)
	}
	if models["legacy"].Thinking != nil {
		t.Fatalf("legacy model thinking metadata = %+v, want absent", models["legacy"].Thinking)
	}
	binary := models["binary"].Thinking
	if binary == nil || !binary.Supported || strings.Join(binary.Levels, ",") != "none,medium" {
		t.Fatalf("binary model thinking metadata = %+v, want off/medium contract", binary)
	}
}

func TestHandlerNormalizesCodexOnlyHistoryForOllama(t *testing.T) {
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

	handler := newTestHandler(t, ollama.URL, chatGPT.URL, writeCatalog(t, "glm-5.2:cloud"))
	proxy := httptest.NewServer(handler)
	defer proxy.Close()

	payload := `{
		"model":"glm-5.2:cloud",
		"stream":true,
		"input":[
			{"type":"compaction","encrypted_content":"opaque"},
			{"type":"custom_tool_call","id":"ctc_1","status":"completed","call_id":"call_1","name":"apply_patch","input":"*** Begin Patch"},
			{"type":"custom_tool_call_output","call_id":"call_1","output":"Success"},
			{"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}]},
			{"type":"future_codex_item","secret":"ignored"}
		]
	}`
	resp, err := http.Post(proxy.URL+PathPrefix+"/v1/responses", "application/json", strings.NewReader(payload))
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
	call := forwarded.Input[0]
	if call["type"] != "function_call" || call["call_id"] != "call_1" || call["name"] != "apply_patch" {
		t.Fatalf("converted call = %#v", call)
	}
	if call["arguments"] != `{"input":"*** Begin Patch"}` {
		t.Fatalf("converted arguments = %#v", call["arguments"])
	}
	output := forwarded.Input[1]
	if output["type"] != "function_call_output" || output["call_id"] != "call_1" || output["output"] != "Success" {
		t.Fatalf("converted output = %#v", output)
	}
	if forwarded.Input[2]["type"] != "message" {
		t.Fatalf("message was not preserved: %#v", forwarded.Input[2])
	}
}

func TestHandlerFiltersNativeReasoningWhenSwitchingToOllama(t *testing.T) {
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

	handler := newTestHandler(t, ollama.URL, chatGPT.URL, writeCatalog(t, "glm-5.3-flash:cloud"))
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
	resp, err := http.Post(proxy.URL+PathPrefix+"/v1/responses", "application/json", strings.NewReader(payload))
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

func TestHandlerPassesNativeModelToChatGPT(t *testing.T) {
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

	handler := newTestHandler(t, ollama.URL, chatGPT.URL+"/backend-api/codex", writeCatalog(t, "glm-5.2:cloud"))
	proxy := httptest.NewServer(handler)
	defer proxy.Close()

	payload := []byte(`{"model":"gpt-5.6-sol","stream":true}`)
	req, err := http.NewRequest(http.MethodPost, proxy.URL+PathPrefix+"/v1/responses", bytes.NewReader(payload))
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

func TestHandlerPassesNativeModelToOpenAIAPIWithAPIKey(t *testing.T) {
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

	handler, err := New(Config{
		PathPrefix:         PathPrefix,
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
		proxy.URL+PathPrefix+"/v1/responses",
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

func TestHandlerPreservesCompressedNativeRequestWithoutOllamaReasoning(t *testing.T) {
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

	handler := newTestHandler(t, ollama.URL, chatGPT.URL+"/backend-api/codex", writeCatalog(t, "glm-5.3-flash:cloud"))
	proxy := httptest.NewServer(handler)
	defer proxy.Close()

	payload := []byte(`{"model":"gpt-5.6-sol","input":[{"type":"reasoning","id":"rs_098c6fb068ce51bf016a9709ab7dcc87d185ecc21991f0f39c","encrypted_content":"gAAAAAB-native"}]}`)
	encoder, err := zstd.NewWriter(nil)
	if err != nil {
		t.Fatal(err)
	}
	compressed := encoder.EncodeAll(payload, nil)
	encoder.Close()

	req, err := http.NewRequest(http.MethodPost, proxy.URL+PathPrefix+"/v1/responses", bytes.NewReader(compressed))
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

func TestHandlerFiltersOllamaReasoningWhenSwitchingToNativeModel(t *testing.T) {
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

	handler := newTestHandler(t, ollama.URL, chatGPT.URL+"/backend-api/codex", writeCatalog(t, "glm-5.3-flash:cloud"))
	proxy := httptest.NewServer(handler)
	defer proxy.Close()

	payload := []byte(`{
		"model":"gpt-5.6-sol",
		"input":[
			{"type":"reasoning","id":"rs_713083","encrypted_content":"The user wants info about the repo"},
			{"type":"reasoning","id":"rs_resp_123456","encrypted_content":"More plaintext thinking"},
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

	req, err := http.NewRequest(http.MethodPost, proxy.URL+PathPrefix+"/v1/responses", bytes.NewReader(compressed))
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
	if len(forwarded.Input) != 2 {
		t.Fatalf("forwarded input = %#v", forwarded.Input)
	}
	if forwarded.Input[0]["id"] != "rs_098c6fb068ce51bf016a9709ab7dcc87d185ecc21991f0f39c" ||
		forwarded.Input[0]["encrypted_content"] != "gAAAAAB-native" {
		t.Fatalf("native encrypted reasoning was not preserved: %#v", forwarded.Input[0])
	}
	if forwarded.Input[1]["type"] != "message" {
		t.Fatalf("message was not preserved: %#v", forwarded.Input[1])
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

func TestHandlerRequestsHTTPFallbackForWebSocketUpgrade(t *testing.T) {
	ollama := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Fatal("WebSocket fallback should not reach Ollama")
	}))
	defer ollama.Close()
	chatGPT := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Fatal("WebSocket fallback should not reach ChatGPT")
	}))
	defer chatGPT.Close()

	handler := newTestHandler(t, ollama.URL, chatGPT.URL+"/backend-api/codex", writeCatalog(t, "glm-5.2:cloud"))
	proxy := httptest.NewServer(handler)
	defer proxy.Close()

	req, err := http.NewRequest(http.MethodGet, proxy.URL+PathPrefix+"/v1/responses", nil)
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

func TestHandlerStatusReportsObservedRoutes(t *testing.T) {
	ollama := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer ollama.Close()

	chatGPT := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusAccepted)
	}))
	defer chatGPT.Close()

	handler := newTestHandler(t, ollama.URL, chatGPT.URL+"/backend-api/codex", writeCatalog(t, "glm-5.2:cloud"))
	proxy := httptest.NewServer(handler)
	defer proxy.Close()

	for _, payload := range []string{
		`{"model":"glm-5.2:cloud"}`,
		`{"model":"gpt-5.6-sol"}`,
	} {
		req, err := http.NewRequest(http.MethodPost, proxy.URL+PathPrefix+"/v1/responses", strings.NewReader(payload))
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

	resp, err := http.Get(proxy.URL + PathPrefix + "/_status")
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

func TestHandlerCountersExcludeProbesAndFailedRetries(t *testing.T) {
	ollama := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer ollama.Close()
	chatGPT := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusMethodNotAllowed)
	}))
	defer chatGPT.Close()

	handler := newTestHandler(t, ollama.URL, chatGPT.URL+"/backend-api/codex", writeCatalog(t, "glm-5.2:cloud"))
	proxy := httptest.NewServer(handler)
	defer proxy.Close()

	probeReq, err := http.NewRequest(http.MethodGet, proxy.URL+PathPrefix+"/v1/responses", nil)
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
		proxy.URL+PathPrefix+"/v1/responses",
		"application/json",
		strings.NewReader(`{"model":"glm-5.2:cloud"}`),
	)
	if err != nil {
		t.Fatal(err)
	}
	_ = failedResp.Body.Close()

	statusResp, err := http.Get(proxy.URL + PathPrefix + "/_status")
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

func TestHandlerWritesSafeActivityLog(t *testing.T) {
	ollama := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer ollama.Close()

	activityLogPath := filepath.Join(t.TempDir(), "logs", "codex-proxy.log")
	handler, err := New(Config{
		PathPrefix:         PathPrefix,
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
		proxy.URL+PathPrefix+"/v1/responses",
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

	data, err := os.ReadFile(activityLogPath)
	if err != nil {
		t.Fatal(err)
	}
	logText := string(data)
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

func TestHandlerRejectsNonLoopbackClients(t *testing.T) {
	handler := newTestHandler(t, "http://127.0.0.1:11434", "https://chatgpt.com/backend-api/codex", writeCatalog(t, "glm"))
	req := httptest.NewRequest(http.MethodGet, "http://example.test"+PathPrefix+"/_health", nil)
	req.RemoteAddr = "192.0.2.10:1234"
	recorder := httptest.NewRecorder()

	handler.ServeHTTP(recorder, req)
	if recorder.Code != http.StatusForbidden {
		t.Fatalf("status = %d, want 403", recorder.Code)
	}
}

func TestHandlerFailsClosedWhenCatalogIsMissing(t *testing.T) {
	handler := newTestHandler(t, "http://127.0.0.1:11434", "https://chatgpt.com/backend-api/codex", filepath.Join(t.TempDir(), "missing.json"))
	req := httptest.NewRequest(http.MethodPost, "http://localhost"+PathPrefix+"/v1/responses", strings.NewReader(`{"model":"glm"}`))
	req.RemoteAddr = "127.0.0.1:1234"
	recorder := httptest.NewRecorder()

	handler.ServeHTTP(recorder, req)
	if recorder.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503: %s", recorder.Code, recorder.Body.String())
	}
}

func newTestHandler(t *testing.T, ollamaURL, chatGPTURL, catalogPath string) *Handler {
	t.Helper()
	handler, err := New(Config{
		PathPrefix:         PathPrefix,
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
	t.Helper()
	entries := make([]map[string]string, 0, len(models))
	for _, model := range models {
		entries = append(entries, map[string]string{"slug": model})
	}
	data, err := json.Marshal(map[string]any{"models": entries})
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), ModelCatalogFilename)
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}
