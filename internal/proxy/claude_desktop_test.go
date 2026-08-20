package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/anthropic"
)

func TestGatewayRoutesClaudeProtocolToOllama(t *testing.T) {
	var paths []string
	var messageModels []string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		paths = append(paths, r.URL.RequestURI())
		if r.Header.Get("Authorization") != "" || r.Header.Get("Cookie") != "" {
			t.Fatal("gateway credential leaked upstream")
		}
		if r.Header.Get("X-Api-Key") != "ollama" {
			t.Fatalf("X-Api-Key = %q", r.Header.Get("X-Api-Key"))
		}
		if r.URL.Path == "/v1/messages" {
			messageModels = append(messageModels, requestModel(t, r))
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"ok":true}`)
	}))
	defer upstream.Close()

	p := startTestGateway(t, upstream.URL)
	client := &http.Client{Timeout: 2 * time.Second}
	for _, request := range []struct {
		method string
		path   string
		body   string
	}{
		{http.MethodGet, "/v1/models?limit=1000", ""},
		{http.MethodPost, "/v1/messages", `{"model":"claude-fable-5","messages":[]}`},
		{http.MethodPost, "/v1/messages", `{"model":"claude-opus-5","messages":[]}`},
		{http.MethodPost, "/v1/messages", `{"model":"claude-sonnet-5","messages":[]}`},
		{http.MethodPost, "/v1/messages", `{"model":"claude-haiku-4-5-20251001","messages":[]}`},
		{http.MethodPost, "/v1/messages", `{"model":"claude-sonnet-5-20251001","messages":[]}`},
		{http.MethodPost, "/v1/messages", `{"model":"claude-opus-4-6","messages":[]}`},
	} {
		req, err := http.NewRequest(request.method, "http://"+p.Addr()+request.path, strings.NewReader(request.body))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Authorization", "Bearer local-placeholder")
		req.Header.Set("Cookie", "session=secret")
		resp, err := client.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		if resp.StatusCode != http.StatusOK {
			_ = resp.Body.Close()
			t.Fatalf("%s = %d", request.path, resp.StatusCode)
		}
		if request.path == "/v1/models?limit=1000" {
			body, err := io.ReadAll(resp.Body)
			if err != nil {
				t.Fatal(err)
			}
			var catalog struct {
				Data    []gatewayModel `json:"data"`
				FirstID string         `json:"first_id"`
				LastID  string         `json:"last_id"`
				HasMore bool           `json:"has_more"`
			}
			if err := json.Unmarshal(body, &catalog); err != nil {
				t.Fatal(err)
			}
			var rawCatalog struct {
				Data []map[string]json.RawMessage `json:"data"`
			}
			if err := json.Unmarshal(body, &rawCatalog); err != nil {
				t.Fatal(err)
			}
			for _, model := range rawCatalog.Data {
				if _, ok := model["max_input_tokens"]; ok {
					t.Fatalf("gateway model should not advertise context size: %s", body)
				}
			}
			gotIDs := make([]string, len(catalog.Data))
			gotNames := make([]string, len(catalog.Data))
			familyDefaults := make(map[string]int)
			for i, model := range catalog.Data {
				gotIDs[i] = model.ID
				gotNames[i] = model.DisplayName
				if model.Type != "model" || model.AnthropicFamilyTier == "" {
					t.Fatalf("incomplete gateway model metadata: %+v", model)
				}
				if model.IsFamilyDefault {
					familyDefaults[model.AnthropicFamilyTier]++
				}
				if model.ID == "claude-sonnet-5-20251001" && model.AnthropicFamilyTier != "sonnet" {
					t.Fatalf("Qwen family tier = %q, want sonnet", model.AnthropicFamilyTier)
				}
			}
			for _, family := range []string{"opus", "fable", "sonnet", "haiku"} {
				if familyDefaults[family] != 1 {
					t.Fatalf("%s family defaults = %d, want 1", family, familyDefaults[family])
				}
			}
			wantIDs := "claude-fable-5,claude-opus-5,claude-sonnet-5,claude-haiku-4-5-20251001,claude-sonnet-5-20251001"
			if strings.Join(gotIDs, ",") != wantIDs || catalog.FirstID != gotIDs[0] || catalog.LastID != gotIDs[len(gotIDs)-1] || catalog.HasMore {
				t.Fatalf("gateway catalog = %+v", catalog.Data)
			}
			wantNames := "GLM 5.2,Kimi K3,DeepSeek V4 Pro,DeepSeek V4 Flash,Qwen3.8 MLX"
			if strings.Join(gotNames, ",") != wantNames {
				t.Fatalf("gateway display names = %v, want %s", gotNames, wantNames)
			}
		}
		_ = resp.Body.Close()
	}

	wantModels := []string{
		"glm-5.2:cloud",
		"kimi-k3:cloud",
		"deepseek-v4-pro:cloud",
		"deepseek-v4-flash:0731:cloud",
		"qwen3.8:27b-mlx",
		"glm-5.2:cloud",
	}
	if strings.Join(messageModels, ",") != strings.Join(wantModels, ",") {
		t.Fatalf("rewritten message models = %v", messageModels)
	}
	if len(paths) != 6 {
		t.Fatalf("upstream paths = %v", paths)
	}
	if got := p.Counts().Routed; got != 6 {
		t.Fatalf("routed requests = %d, want 6", got)
	}
}

func TestGatewayPrunesCloudModelsWhenSignedOut(t *testing.T) {
	upstream := httptest.NewServer(http.NotFoundHandler())
	defer upstream.Close()

	var cloudModelsAvailable atomic.Bool
	p, err := NewClaudeDesktop(ClaudeDesktopConfig{
		ListenAddr:           "127.0.0.1:0",
		OllamaURL:            upstream.URL,
		Model:                "kimi-k3:cloud",
		CloudModelsAvailable: func(context.Context) bool { return cloudModelsAvailable.Load() },
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := p.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()
		if err := p.Close(ctx); err != nil {
			t.Fatal(err)
		}
	})

	fetchCatalog := func() struct {
		Data    []gatewayModel `json:"data"`
		FirstID string         `json:"first_id"`
		LastID  string         `json:"last_id"`
	} {
		resp, err := http.Get("http://" + p.Addr() + "/v1/models")
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("status = %d, want 200", resp.StatusCode)
		}
		var catalog struct {
			Data    []gatewayModel `json:"data"`
			FirstID string         `json:"first_id"`
			LastID  string         `json:"last_id"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&catalog); err != nil {
			t.Fatal(err)
		}
		return catalog
	}

	catalog := fetchCatalog()
	if len(catalog.Data) != 1 || catalog.Data[0].DisplayName != "Qwen3.8 MLX" {
		t.Fatalf("signed-out catalog = %+v, want local model only", catalog.Data)
	}
	if catalog.FirstID != catalog.Data[0].ID || catalog.LastID != catalog.Data[0].ID {
		t.Fatalf("signed-out catalog bounds = %q, %q", catalog.FirstID, catalog.LastID)
	}

	cloudModelsAvailable.Store(true)
	catalog = fetchCatalog()
	if len(catalog.Data) != len(claudeModels) {
		t.Fatalf("signed-in catalog has %d models, want %d", len(catalog.Data), len(claudeModels))
	}
}

func TestGatewayCountsTokensLocally(t *testing.T) {
	upstreamCalls := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamCalls++
	}))
	defer upstream.Close()
	p := startTestGateway(t, upstream.URL)

	resp, err := http.Post(
		"http://"+p.Addr()+"/v1/messages/count_tokens",
		"application/json",
		strings.NewReader(`{"model":"claude-sonnet","messages":[{"role":"user","content":"hello world"}]}`),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	var result struct {
		InputTokens int `json:"input_tokens"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatal(err)
	}
	if result.InputTokens <= 0 {
		t.Fatalf("input_tokens = %d, want positive estimate", result.InputTokens)
	}
	if upstreamCalls != 0 {
		t.Fatalf("upstream calls = %d, want 0", upstreamCalls)
	}
}

func TestGatewayRejectsNonLoopbackHost(t *testing.T) {
	var upstreamCalls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		upstreamCalls.Add(1)
	}))
	defer upstream.Close()
	p := startTestGateway(t, upstream.URL)

	_, port, err := net.SplitHostPort(p.Addr())
	if err != nil {
		t.Fatal(err)
	}
	req, err := http.NewRequest(http.MethodPost, "http://"+p.Addr()+"/v1/messages", strings.NewReader(`{"model":"claude-sonnet-5","messages":[]}`))
	if err != nil {
		t.Fatal(err)
	}
	req.Host = net.JoinHostPort("attacker.example", port)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusForbidden {
		t.Fatalf("status = %d, want 403", resp.StatusCode)
	}
	if got := upstreamCalls.Load(); got != 0 {
		t.Fatalf("upstream calls = %d, want 0", got)
	}
}

func TestGatewayRejectsHostWithWrongPort(t *testing.T) {
	var upstreamCalls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		upstreamCalls.Add(1)
	}))
	defer upstream.Close()
	p := startTestGateway(t, upstream.URL)

	_, port, err := net.SplitHostPort(p.Addr())
	if err != nil {
		t.Fatal(err)
	}
	wrongPort := "1"
	if port == wrongPort {
		wrongPort = "2"
	}
	req, err := http.NewRequest(http.MethodGet, "http://"+p.Addr()+"/v1/models", nil)
	if err != nil {
		t.Fatal(err)
	}
	req.Host = net.JoinHostPort("127.0.0.1", wrongPort)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusForbidden {
		t.Fatalf("status = %d, want 403", resp.StatusCode)
	}
	if got := upstreamCalls.Load(); got != 0 {
		t.Fatalf("upstream calls = %d, want 0", got)
	}
}

func TestGatewayAllowsLoopbackHosts(t *testing.T) {
	upstream := httptest.NewServer(http.NotFoundHandler())
	defer upstream.Close()
	p := startTestGateway(t, upstream.URL)

	_, port, err := net.SplitHostPort(p.Addr())
	if err != nil {
		t.Fatal(err)
	}
	for _, host := range []string{"localhost", "127.0.0.1", "::1"} {
		t.Run(host, func(t *testing.T) {
			req, err := http.NewRequest(http.MethodGet, "http://"+p.Addr()+"/v1/models", nil)
			if err != nil {
				t.Fatal(err)
			}
			req.Host = net.JoinHostPort(host, port)
			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				t.Fatal(err)
			}
			defer resp.Body.Close()
			if resp.StatusCode != http.StatusOK {
				t.Fatalf("status = %d, want 200", resp.StatusCode)
			}
		})
	}
}

func TestGatewayDoesNotForwardUnsupportedRoutes(t *testing.T) {
	var upstreamCalls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		upstreamCalls.Add(1)
	}))
	defer upstream.Close()
	p := startTestGateway(t, upstream.URL)

	for _, test := range []struct {
		method string
		path   string
		status int
		allow  string
	}{
		{http.MethodPost, "/api/delete", http.StatusNotFound, ""},
		{http.MethodPost, "/v1/chat/completions", http.StatusNotFound, ""},
		{http.MethodGet, "/v1/messages", http.StatusMethodNotAllowed, http.MethodPost},
		{http.MethodGet, "/v1/messages/count_tokens", http.StatusMethodNotAllowed, http.MethodPost},
	} {
		t.Run(test.method+" "+test.path, func(t *testing.T) {
			req, err := http.NewRequest(test.method, "http://"+p.Addr()+test.path, nil)
			if err != nil {
				t.Fatal(err)
			}
			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				t.Fatal(err)
			}
			defer resp.Body.Close()
			if resp.StatusCode != test.status {
				t.Fatalf("status = %d, want %d", resp.StatusCode, test.status)
			}
			if resp.Header.Get("Allow") != test.allow {
				t.Fatalf("Allow = %q, want %q", resp.Header.Get("Allow"), test.allow)
			}
		})
	}
	if got := upstreamCalls.Load(); got != 0 {
		t.Fatalf("upstream calls = %d, want 0", got)
	}
}

func TestGatewayLazilyRetriesUnsupportedVision(t *testing.T) {
	for _, test := range []struct {
		name     string
		body     string
		preserve string
	}{
		{
			name:     "message image",
			body:     `{"model":"claude-sonnet-5","messages":[{"role":"user","content":[{"type":"text","text":"Describe this"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"aW1hZ2U="}}]}],"stream":true}`,
			preserve: "Describe this",
		},
		{
			name:     "tool result image",
			body:     `{"model":"claude-fable-5","messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"tool_1","content":[{"type":"text","text":"Screenshot"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"aW1hZ2U="}}]}]}]}`,
			preserve: "tool_1",
		},
		{
			name:     "historical image with later text",
			body:     `{"model":"claude-fable-5","messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"aW1hZ2U="}},{"type":"text","text":"what dis"}]},{"role":"assistant","content":[{"type":"text","text":"I cannot inspect it"}]},{"role":"user","content":[{"type":"text","text":"kk"}]}]}`,
			preserve: "kk",
		},
		{
			name:     "unknown ID uses non-vision fallback",
			body:     `{"model":"claude-unknown","messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"aW1hZ2U="}}]}]}`,
			preserve: unsupportedImageNotice,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			var upstreamBodies [][]byte
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, err := io.ReadAll(r.Body)
				if err != nil {
					t.Fatal(err)
				}
				upstreamBodies = append(upstreamBodies, body)
				w.Header().Set("Content-Type", "application/json")
				if len(upstreamBodies) == 1 {
					w.WriteHeader(http.StatusBadRequest)
					_, _ = io.WriteString(w, `{"type":"error","error":{"type":"invalid_request_error","message":"model does not support images"}}`)
					return
				}
				_, _ = io.WriteString(w, `{"ok":true}`)
			}))
			defer upstream.Close()
			p := startTestGateway(t, upstream.URL)

			resp, err := http.Post("http://"+p.Addr()+"/v1/messages", "application/json", strings.NewReader(test.body))
			if err != nil {
				t.Fatal(err)
			}
			defer resp.Body.Close()
			if resp.StatusCode != http.StatusOK {
				t.Fatalf("status = %d, want 200", resp.StatusCode)
			}
			if len(upstreamBodies) != 2 {
				t.Fatalf("upstream calls = %d, want 2", len(upstreamBodies))
			}
			if !strings.Contains(string(upstreamBodies[0]), `"type":"image"`) || !strings.Contains(string(upstreamBodies[0]), "aW1hZ2U=") {
				t.Fatalf("first request was unexpectedly sanitized: %s", upstreamBodies[0])
			}
			if strings.Contains(string(upstreamBodies[1]), `"type":"image"`) || strings.Contains(string(upstreamBodies[1]), "aW1hZ2U=") {
				t.Fatalf("image reached fallback request: %s", upstreamBodies[1])
			}
			if !strings.Contains(string(upstreamBodies[1]), unsupportedImageNotice) || !strings.Contains(string(upstreamBodies[1]), test.preserve) {
				t.Fatalf("sanitized fallback body = %s", upstreamBodies[1])
			}
		})
	}
}

func TestGatewayDoesNotSanitizeImagesOnSuccessfulRequest(t *testing.T) {
	var upstreamBodies [][]byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatal(err)
		}
		upstreamBodies = append(upstreamBodies, body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"ok":true}`)
	}))
	defer upstream.Close()
	p := startTestGateway(t, upstream.URL)

	body := `{"model":"claude-fable-5","messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"aW1hZ2U="}},{"type":"text","text":"what dis"}]}]}`
	resp, err := http.Post("http://"+p.Addr()+"/v1/messages", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if len(upstreamBodies) != 1 {
		t.Fatalf("upstream calls = %d, want 1", len(upstreamBodies))
	}
	if !strings.Contains(string(upstreamBodies[0]), `"type":"image"`) || !strings.Contains(string(upstreamBodies[0]), "aW1hZ2U=") {
		t.Fatalf("successful request was unexpectedly sanitized: %s", upstreamBodies[0])
	}
}

func TestGatewayDoesNotRetryImageFreeBadRequest(t *testing.T) {
	upstreamCalls := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamCalls++
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, `{"type":"error","error":{"type":"invalid_request_error","message":"bad tool schema"}}`)
	}))
	defer upstream.Close()
	p := startTestGateway(t, upstream.URL)

	body := `{"model":"claude-fable-5","messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}]}`
	resp, err := http.Post("http://"+p.Addr()+"/v1/messages", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", resp.StatusCode)
	}
	if upstreamCalls != 1 {
		t.Fatalf("upstream calls = %d, want 1", upstreamCalls)
	}
}

func TestGatewayDoesNotRetryUnrelatedBadRequestWithImage(t *testing.T) {
	upstreamCalls := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamCalls++
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, `{"type":"error","error":{"type":"invalid_request_error","message":"bad tool schema"}}`)
	}))
	defer upstream.Close()
	p := startTestGateway(t, upstream.URL)

	body := `{"model":"claude-fable-5","messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"aW1hZ2U="}}]}]}`
	resp, err := http.Post("http://"+p.Addr()+"/v1/messages", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	responseBody, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusBadRequest || upstreamCalls != 1 {
		t.Fatalf("status/calls = %d/%d, want 400/1", resp.StatusCode, upstreamCalls)
	}
	if !strings.Contains(string(responseBody), "bad tool schema") {
		t.Fatalf("original upstream error was not preserved: %s", responseBody)
	}
}

func TestGatewayDoesNotRetryUnstructuredImageError(t *testing.T) {
	upstreamCalls := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamCalls++
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, "model does not support images")
	}))
	defer upstream.Close()
	p := startTestGateway(t, upstream.URL)

	body := `{"model":"claude-fable-5","messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"aW1hZ2U="}}]}]}`
	resp, err := http.Post("http://"+p.Addr()+"/v1/messages", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	responseBody, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusBadRequest || upstreamCalls != 1 {
		t.Fatalf("status/calls = %d/%d, want 400/1", resp.StatusCode, upstreamCalls)
	}
	if string(responseBody) != "model does not support images" {
		t.Fatalf("response body = %q", responseBody)
	}
}

func TestGatewayLogsUnknownModelFallback(t *testing.T) {
	var logs bytes.Buffer
	p := &ClaudeDesktop{
		logger: slog.New(slog.NewTextHandler(&logs, &slog.HandlerOptions{Level: slog.LevelDebug})),
	}

	model := p.modelForClaudeID("claude-future", "kimi-k3:cloud")
	if model.OllamaModel != "kimi-k3:cloud" {
		t.Fatalf("fallback model = %q", model.OllamaModel)
	}
	if output := logs.String(); !strings.Contains(output, "requested_model=claude-future") || !strings.Contains(output, "default_model=kimi-k3:cloud") {
		t.Fatalf("fallback log = %q", output)
	}
}

func TestGatewayAllowsVisionForSupportedModel(t *testing.T) {
	upstreamCalls := 0
	upstreamModel := ""
	var upstreamBody []byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamCalls++
		var err error
		upstreamBody, err = io.ReadAll(r.Body)
		if err != nil {
			t.Fatal(err)
		}
		var request struct {
			Model string `json:"model"`
		}
		if err := json.Unmarshal(upstreamBody, &request); err != nil {
			t.Fatal(err)
		}
		upstreamModel = request.Model
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"ok":true}`)
	}))
	defer upstream.Close()
	p := startTestGateway(t, upstream.URL)

	body := `{"model":"claude-opus-5","messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"aW1hZ2U="}}]}]}`
	resp, err := http.Post("http://"+p.Addr()+"/v1/messages", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if upstreamCalls != 1 || upstreamModel != "kimi-k3:cloud" {
		t.Fatalf("upstream calls/model = %d/%q", upstreamCalls, upstreamModel)
	}
	if !strings.Contains(string(upstreamBody), `"type":"image"`) || !strings.Contains(string(upstreamBody), "aW1hZ2U=") {
		t.Fatalf("vision request was unexpectedly sanitized: %s", upstreamBody)
	}
}

func TestGatewayDoesNotRetryVisionModelBadRequest(t *testing.T) {
	upstreamCalls := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamCalls++
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, `{"type":"error","error":{"type":"invalid_request_error","message":"bad request"}}`)
	}))
	defer upstream.Close()
	p := startTestGateway(t, upstream.URL)

	body := `{"model":"claude-opus-5","messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"aW1hZ2U="}}]}]}`
	resp, err := http.Post("http://"+p.Addr()+"/v1/messages", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", resp.StatusCode)
	}
	if upstreamCalls != 1 {
		t.Fatalf("upstream calls = %d, want 1", upstreamCalls)
	}
}

func TestGatewayCountsUnsupportedVisionWithoutUpstream(t *testing.T) {
	upstreamCalls := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		upstreamCalls++
	}))
	defer upstream.Close()
	p := startTestGateway(t, upstream.URL)

	body := `{"model":"claude-haiku-4-5-20251001","messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"aW1hZ2U="}}]}]}`
	resp, err := http.Post("http://"+p.Addr()+"/v1/messages/count_tokens", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	var response anthropic.CountTokensResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		t.Fatal(err)
	}
	if response.InputTokens <= 0 {
		t.Fatalf("input_tokens = %d, want positive estimate", response.InputTokens)
	}
	if upstreamCalls != 0 {
		t.Fatalf("upstream calls = %d, want 0", upstreamCalls)
	}
}

func TestGatewayWaitsForColdUpstream(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	address := listener.Addr().String()
	if err := listener.Close(); err != nil {
		t.Fatal(err)
	}

	p := startTestGateway(t, "http://"+address)
	response := make(chan int, 1)
	errs := make(chan error, 1)
	go func() {
		resp, err := http.Post(
			"http://"+p.Addr()+"/v1/messages",
			"application/json",
			strings.NewReader(`{"model":"claude-sonnet-5","messages":[]}`),
		)
		if err != nil {
			errs <- err
			return
		}
		defer resp.Body.Close()
		response <- resp.StatusCode
	}()

	time.Sleep(250 * time.Millisecond)
	upstreamListener, err := net.Listen("tcp", address)
	if err != nil {
		t.Fatal(err)
	}
	upstream := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"ok":true}`)
	})}
	go func() { _ = upstream.Serve(upstreamListener) }()
	t.Cleanup(func() { _ = upstream.Close() })

	select {
	case err := <-errs:
		t.Fatal(err)
	case status := <-response:
		if status != http.StatusOK {
			t.Fatalf("status = %d, want 200", status)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("gateway did not resume after upstream became ready")
	}
}

func TestProbeIdentifiesOllamaGateway(t *testing.T) {
	upstream := httptest.NewServer(http.NotFoundHandler())
	defer upstream.Close()
	p := startTestGateway(t, upstream.URL)

	if err := ProbeClaudeDesktop(context.Background(), "http://"+p.Addr()); err != nil {
		t.Fatalf("ProbeClaudeDesktop returned error: %v", err)
	}

	other := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))
	defer other.Close()
	if err := ProbeClaudeDesktop(context.Background(), other.URL); err == nil || !strings.Contains(err.Error(), "another service is using") {
		t.Fatalf("ProbeClaudeDesktop error = %v, want port ownership error", err)
	}
}

func TestGatewayCachesSuccessfulUpstreamReadiness(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	p, err := NewClaudeDesktop(ClaudeDesktopConfig{
		ListenAddr: "127.0.0.1:0",
		OllamaURL:  "http://" + listener.Addr().String(),
		Model:      "glm-5.2:cloud",
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := p.waitForUpstream(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := listener.Close(); err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	if err := p.waitForUpstream(ctx); err != nil {
		t.Fatalf("cached readiness returned error: %v", err)
	}

	p.markUpstreamNotReady()
	ctx, cancel = context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	if err := p.waitForUpstream(ctx); err == nil {
		t.Fatal("expected invalidated readiness to probe the closed upstream")
	}
}

func TestGatewaySharesFailedUpstreamReadinessCheck(t *testing.T) {
	p, err := NewClaudeDesktop(ClaudeDesktopConfig{
		ListenAddr: "127.0.0.1:0",
		OllamaURL:  "http://127.0.0.1:1",
		Model:      "glm-5.2:cloud",
	})
	if err != nil {
		t.Fatal(err)
	}
	p.readyWait = 100 * time.Millisecond
	p.readyPoll = 5 * time.Millisecond
	dialCalls := 0
	p.dial = func(context.Context, string, string) (net.Conn, error) {
		dialCalls++
		return nil, errors.New("upstream unavailable")
	}

	const requests = 8
	start := make(chan struct{})
	results := make(chan error, requests)
	for range requests {
		go func() {
			<-start
			results <- p.waitForUpstream(context.Background())
		}()
	}
	close(start)
	for range requests {
		if err := <-results; err == nil {
			t.Fatal("expected readiness check to fail")
		}
	}
	if dialCalls >= requests*10 {
		t.Fatalf("dial calls = %d, readiness checks were serialized", dialCalls)
	}
}

func TestGatewayCloseCancelsUpstreamReadinessCheck(t *testing.T) {
	p, err := NewClaudeDesktop(ClaudeDesktopConfig{
		ListenAddr: "127.0.0.1:0",
		OllamaURL:  "http://127.0.0.1:1",
		Model:      "glm-5.2:cloud",
	})
	if err != nil {
		t.Fatal(err)
	}
	started := make(chan struct{}, 1)
	p.dial = func(context.Context, string, string) (net.Conn, error) {
		select {
		case started <- struct{}{}:
		default:
		}
		return nil, errors.New("upstream unavailable")
	}
	result := make(chan error, 1)
	go func() { result <- p.waitForUpstream(context.Background()) }()
	<-started
	if err := p.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-result:
		if err == nil {
			t.Fatal("expected canceled readiness check to fail")
		}
	case <-time.After(time.Second):
		t.Fatal("readiness check remained blocked after gateway close")
	}
}

func startTestGateway(t *testing.T, upstreamURL string) *ClaudeDesktop {
	t.Helper()
	p, err := NewClaudeDesktop(ClaudeDesktopConfig{ListenAddr: "127.0.0.1:0", OllamaURL: upstreamURL, Model: "glm-5.2:cloud"})
	if err != nil {
		t.Fatal(err)
	}
	if err := p.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()
		if err := p.Close(ctx); err != nil {
			t.Fatal(err)
		}
	})
	return p
}

func requestModel(t *testing.T, r *http.Request) string {
	t.Helper()
	var payload struct {
		Model string `json:"model"`
	}
	if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
		t.Fatal(err)
	}
	return payload.Model
}
