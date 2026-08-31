package proxy

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/anthropic"
	"github.com/ollama/ollama/api"
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
		{http.MethodPost, "/v1/messages", `{"model":"glm-5.2:cloud","messages":[]}`},
		{http.MethodPost, "/v1/messages", `{"model":"kimi-k3:cloud","messages":[]}`},
		{http.MethodPost, "/v1/messages", `{"model":"deepseek-v4-pro:cloud","messages":[]}`},
		{http.MethodPost, "/v1/messages", `{"model":"deepseek-v4-flash:0731:cloud","messages":[]}`},
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
			for i, model := range catalog.Data {
				gotIDs[i] = model.ID
				gotNames[i] = model.DisplayName
				if model.Type != "model" {
					t.Fatalf("incomplete gateway model metadata: %+v", model)
				}
			}
			wantIDs := "claude-fable-5,claude-opus-5,claude-sonnet-5,claude-haiku-4-5-20251001,claude-sonnet-4-6"
			if strings.Join(gotIDs, ",") != wantIDs || catalog.FirstID != gotIDs[0] || catalog.LastID != gotIDs[len(gotIDs)-1] || catalog.HasMore {
				t.Fatalf("gateway catalog = %+v", catalog.Data)
			}
			wantNames := "glm-5.2:cloud,kimi-k3:cloud,deepseek-v4-pro:cloud,deepseek-v4-flash:0731:cloud,gemma4:31b-cloud"
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
	}
	if strings.Join(messageModels, ",") != strings.Join(wantModels, ",") {
		t.Fatalf("rewritten message models = %v", messageModels)
	}
	if len(paths) != 4 {
		t.Fatalf("upstream paths = %v", paths)
	}
	if got := p.Counts().Routed; got != 4 {
		t.Fatalf("routed requests = %d, want 4", got)
	}
}

func TestGatewayRoutesSelectedKimiByExactRoute(t *testing.T) {
	var routedModel string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		routedModel = requestModel(t, r)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"ok":true}`)
	}))
	defer upstream.Close()

	selected := SelectClaudeDesktopModels(DefaultClaudeDesktopModels(), []string{"kimi-k3:cloud"})
	if len(selected) != 1 || selected[0].GatewayID() != "claude-fable-5" || selected[0].OllamaModel != "kimi-k3:cloud" {
		t.Fatalf("selected kimi = %+v, want validated ID routing to exact model", selected)
	}
	p, err := NewClaudeDesktop(ClaudeDesktopConfig{
		ListenAddr: "127.0.0.1:0",
		OllamaURL:  upstream.URL,
		Model:      selected[0].OllamaModel,
		Models:     selected,
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

	resp, err := http.Get("http://" + p.Addr() + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var catalog struct {
		Data []gatewayModel `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&catalog); err != nil {
		t.Fatal(err)
	}
	if len(catalog.Data) != 1 || catalog.Data[0].ID != "claude-fable-5" || catalog.Data[0].DisplayName != "kimi-k3:cloud" {
		t.Fatalf("catalog = %+v, want claude-fable-5 displaying Kimi", catalog.Data)
	}

	message, err := http.Post(
		"http://"+p.Addr()+"/v1/messages",
		"application/json",
		strings.NewReader(`{"model":"claude-fable-5","messages":[{"role":"user","content":"hi"}]}`),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer message.Body.Close()
	if message.StatusCode != http.StatusOK || routedModel != "kimi-k3:cloud" {
		t.Fatalf("status/model = %d/%q, want 200 routing kimi-k3:cloud", message.StatusCode, routedModel)
	}
}

func TestGatewayRoutesEveryMappedModelIDToExactOllamaRoute(t *testing.T) {
	var routed []string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		routed = append(routed, requestModel(t, r))
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"ok":true}`)
	}))
	defer upstream.Close()

	models := ClaudeDesktopModelsFromRecommendations([]api.ModelRecommendation{
		{Model: "glm-5.2:cloud", RequiredPlan: "pro"},
		{Model: "kimi-k3:cloud", RequiredPlan: "pro"},
		{Model: "deepseek-v4-pro", RequiredPlan: "pro"},
		{Model: "deepseek-v4-flash", RequiredPlan: "pro"},
		{Model: "gemma4:26b:cloud", RequiredPlan: "pro"},
	})
	p, err := NewClaudeDesktop(ClaudeDesktopConfig{
		ListenAddr: "127.0.0.1:0",
		OllamaURL:  upstream.URL,
		Model:      models[0].OllamaModel,
		Models:     models,
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

	routes := []struct {
		id     string
		ollama string
	}{
		{"claude-fable-5", "glm-5.2:cloud"},
		{"claude-opus-5", "kimi-k3:cloud"},
		{"claude-sonnet-5", "deepseek-v4-pro:cloud"},
		{"claude-haiku-4-5-20251001", "deepseek-v4-flash:cloud"},
		{"claude-sonnet-4-6", "gemma4:26b:cloud"},
	}
	for _, route := range routes {
		resp, err := http.Post(
			"http://"+p.Addr()+"/v1/messages",
			"application/json",
			strings.NewReader(`{"model":"`+route.id+`","messages":[{"role":"user","content":"hi"}]}`),
		)
		if err != nil {
			t.Fatal(err)
		}
		if resp.StatusCode != http.StatusOK {
			_ = resp.Body.Close()
			t.Fatalf("%s status = %d, want 200", route.id, resp.StatusCode)
		}
		_ = resp.Body.Close()
		if got := routed[len(routed)-1]; got != route.ollama {
			t.Fatalf("%s routed to %q, want exact Ollama route %q", route.id, got, route.ollama)
		}
	}

	resp, err := http.Get("http://" + p.Addr() + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var catalog struct {
		Data []gatewayModel `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&catalog); err != nil {
		t.Fatal(err)
	}
	gotIDs := make([]string, len(catalog.Data))
	for i, model := range catalog.Data {
		gotIDs[i] = model.ID
	}
	wantIDs := []string{
		"claude-fable-5",
		"claude-opus-5",
		"claude-sonnet-5",
		"claude-haiku-4-5-20251001",
		"claude-sonnet-4-6",
	}
	if strings.Join(gotIDs, ",") != strings.Join(wantIDs, ",") {
		t.Fatalf("catalog IDs = %v, want validated Claude IDs %v", gotIDs, wantIDs)
	}
}

func TestGatewayRoutesClaudeAutoClassifierAsOrdinaryMessages(t *testing.T) {
	local := SelectClaudeDesktopModels(nil, []string{"qwen3.5:latest"})
	cloud := ClaudeDesktopModelsFromRecommendations([]api.ModelRecommendation{{Model: "glm-5.2:cloud", RequiredPlan: "pro"}})
	tests := []struct {
		name          string
		models        []ClaudeDesktopModel
		ollamaModel   string
		maxTokens     int
		stopSequences []string
		responseText  string
		configure     func(*ClaudeDesktopConfig)
	}{
		{
			name:          "stage one selected local model",
			models:        local,
			ollamaModel:   "qwen3.5:latest",
			maxTokens:     2112,
			stopSequences: []string{"</block>"},
			responseText:  "malformed classifier output",
			configure: func(config *ClaudeDesktopConfig) {
				config.ListLocalModels = func(context.Context) ([]string, error) {
					return []string{"qwen3.5:latest"}, nil
				}
			},
		},
		{
			name:         "stage two selected cloud model",
			models:       cloud,
			ollamaModel:  "glm-5.2:cloud",
			maxTokens:    10240,
			responseText: "<block>yes</block><category>Synthetic risk</category><reason>Denied by the synthetic fixture.</reason>",
			configure: func(config *ClaudeDesktopConfig) {
				config.ResolveAccessState = func(context.Context) (ClaudeDesktopAccessState, error) {
					return ClaudeDesktopAccessState{Cloud: ClaudeDesktopCloudOn, Account: ClaudeDesktopAccountSignedIn, Plan: "pro"}, nil
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var upstreamPayload map[string]json.RawMessage
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.RequestURI() != "/v1/messages?beta=true" {
					t.Fatalf("request URI = %q", r.URL.RequestURI())
				}
				if r.Header.Get("Authorization") != "" || r.Header.Get("Cookie") != "" || r.Header.Get("Proxy-Authorization") != "" {
					t.Fatal("Claude credentials leaked to the Ollama server")
				}
				if err := json.NewDecoder(r.Body).Decode(&upstreamPayload); err != nil {
					t.Fatal(err)
				}
				w.Header().Set("Content-Type", "application/json")
				_ = json.NewEncoder(w).Encode(anthropic.MessagesResponse{
					ID:    "msg_classifier",
					Type:  "message",
					Role:  "assistant",
					Model: tt.ollamaModel,
					Content: []anthropic.ContentBlock{{
						Type: "text",
						Text: &tt.responseText,
					}},
					StopReason: "end_turn",
					Usage:      anthropic.Usage{InputTokens: 24644, OutputTokens: 300},
				})
			}))
			defer upstream.Close()

			config := ClaudeDesktopConfig{
				ListenAddr: "127.0.0.1:0",
				OllamaURL:  upstream.URL,
				Model:      tt.ollamaModel,
				Models:     tt.models,
			}
			tt.configure(&config)
			p, err := NewClaudeDesktop(config)
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

			requestBody := map[string]any{
				"model":      p.Models()[0].GatewayID(),
				"max_tokens": tt.maxTokens,
				"system": []any{
					map[string]any{"type": "text", "text": "Synthetic policy fixture.", "cache_control": map[string]any{"type": "ephemeral"}},
					map[string]any{"type": "text", "text": "Synthetic session context."},
				},
				"messages": []any{map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{"type": "text", "text": "<transcript>\n"},
						map[string]any{"type": "text", "text": "User: synthetic task\n"},
						map[string]any{"type": "text", "text": "Bash synthetic action\n"},
						map[string]any{"type": "text", "text": "</transcript>\n"},
						map[string]any{"type": "text", "text": "Return the synthetic verdict."},
					},
				}},
			}
			if len(tt.stopSequences) > 0 {
				requestBody["stop_sequences"] = tt.stopSequences
			}
			body, err := json.Marshal(requestBody)
			if err != nil {
				t.Fatal(err)
			}
			req, err := http.NewRequest(http.MethodPost, "http://"+p.Addr()+"/v1/messages?beta=true", strings.NewReader(string(body)))
			if err != nil {
				t.Fatal(err)
			}
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("Authorization", "Bearer local-placeholder")
			req.Header.Set("Cookie", "session=secret")
			req.Header.Set("Proxy-Authorization", "Basic secret")
			response, err := http.DefaultClient.Do(req)
			if err != nil {
				t.Fatal(err)
			}
			defer response.Body.Close()
			var message anthropic.MessagesResponse
			if err := json.NewDecoder(response.Body).Decode(&message); err != nil {
				t.Fatal(err)
			}
			if response.StatusCode != http.StatusOK || len(message.Content) != 1 || message.Content[0].Text == nil || *message.Content[0].Text != tt.responseText {
				t.Fatalf("response = %d %+v, want opaque upstream classifier output", response.StatusCode, message)
			}

			var routedModel string
			if err := json.Unmarshal(upstreamPayload["model"], &routedModel); err != nil {
				t.Fatal(err)
			}
			if routedModel != tt.ollamaModel {
				t.Fatalf("classifier routed to %q, want exact selected model %q", routedModel, tt.ollamaModel)
			}
			if _, ok := upstreamPayload["stream"]; ok {
				t.Fatalf("classifier request unexpectedly gained stream: %s", upstreamPayload["stream"])
			}
			if _, ok := upstreamPayload["tools"]; ok {
				t.Fatalf("classifier request unexpectedly gained tools: %s", upstreamPayload["tools"])
			}
			var maxTokens int
			if err := json.Unmarshal(upstreamPayload["max_tokens"], &maxTokens); err != nil || maxTokens != tt.maxTokens {
				t.Fatalf("max_tokens = %d, error = %v, want %d", maxTokens, err, tt.maxTokens)
			}
			var stopSequences []string
			if raw := upstreamPayload["stop_sequences"]; raw != nil {
				if err := json.Unmarshal(raw, &stopSequences); err != nil {
					t.Fatal(err)
				}
			}
			if strings.Join(stopSequences, ",") != strings.Join(tt.stopSequences, ",") {
				t.Fatalf("stop_sequences = %v, want %v", stopSequences, tt.stopSequences)
			}
		})
	}
}

func TestGatewayPropagatesClaudeAutoClassifierCancellationAndTimeout(t *testing.T) {
	for _, tt := range []struct {
		name       string
		newContext func() (context.Context, context.CancelFunc)
		cancel     bool
		wantErr    error
	}{
		{
			name: "caller cancellation",
			newContext: func() (context.Context, context.CancelFunc) {
				return context.WithCancel(context.Background())
			},
			cancel:  true,
			wantErr: context.Canceled,
		},
		{
			name: "caller timeout",
			newContext: func() (context.Context, context.CancelFunc) {
				return context.WithTimeout(context.Background(), 100*time.Millisecond)
			},
			wantErr: context.DeadlineExceeded,
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			started := make(chan struct{})
			canceled := make(chan struct{})

			models := SelectClaudeDesktopModels(nil, []string{"qwen3.5:latest"})
			p, err := NewClaudeDesktop(ClaudeDesktopConfig{
				ListenAddr: "127.0.0.1:0",
				OllamaURL:  "http://127.0.0.1:1",
				Model:      "qwen3.5:latest",
				Models:     models,
				ListLocalModels: func(context.Context) ([]string, error) {
					return []string{"qwen3.5:latest"}, nil
				},
			})
			if err != nil {
				t.Fatal(err)
			}
			p.readyUntil = time.Now().Add(time.Minute)
			p.ollamaProxy.Transport = roundTripFunc(func(r *http.Request) (*http.Response, error) {
				close(started)
				<-r.Context().Done()
				close(canceled)
				return nil, r.Context().Err()
			})
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

			ctx, cancel := tt.newContext()
			defer cancel()
			req, err := http.NewRequestWithContext(ctx, http.MethodPost, "http://"+p.Addr()+"/v1/messages?beta=true", strings.NewReader(`{"model":"claude-fable-5","max_tokens":2112,"messages":[{"role":"user","content":"synthetic classifier"}]}`))
			if err != nil {
				t.Fatal(err)
			}
			result := make(chan error, 1)
			go func() {
				resp, err := http.DefaultClient.Do(req)
				if resp != nil {
					_ = resp.Body.Close()
				}
				result <- err
			}()
			<-started
			if tt.cancel {
				cancel()
			}
			if err := <-result; !errors.Is(err, tt.wantErr) {
				t.Fatalf("request error = %v, want %v", err, tt.wantErr)
			}
			select {
			case <-canceled:
			case <-time.After(time.Second):
				t.Fatal("classifier cancellation did not reach the selected-model request")
			}
		})
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) {
	return f(r)
}

func TestGatewayPrunesCloudModelsWhenSignedOut(t *testing.T) {
	upstream := httptest.NewServer(http.NotFoundHandler())
	defer upstream.Close()

	var cloudModelsAvailable atomic.Bool
	var localModelAvailable atomic.Bool
	models := ClaudeDesktopModelsFromRecommendations([]api.ModelRecommendation{
		{Model: "kimi-k3:cloud", RequiredPlan: "pro"},
		{Model: "qwen3.8:27b"},
	})
	p, err := NewClaudeDesktop(ClaudeDesktopConfig{
		ListenAddr: "127.0.0.1:0",
		OllamaURL:  upstream.URL,
		Model:      "kimi-k3:cloud",
		Models:     models,
		ResolveAccessState: func(context.Context) (ClaudeDesktopAccessState, error) {
			state := ClaudeDesktopAccessState{Cloud: ClaudeDesktopCloudOn, Account: ClaudeDesktopAccountSignedOut}
			if cloudModelsAvailable.Load() {
				state.Account = ClaudeDesktopAccountSignedIn
				state.Plan = "pro"
			}
			return state, nil
		},
		ListLocalModels: func(context.Context) ([]string, error) {
			if localModelAvailable.Load() {
				return []string{"qwen3.8:27b"}, nil
			}
			return nil, nil
		},
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
	if len(catalog.Data) != 0 || catalog.FirstID != "" || catalog.LastID != "" {
		t.Fatalf("signed-out empty catalog = %+v", catalog)
	}

	localModelAvailable.Store(true)
	catalog = fetchCatalog()
	if len(catalog.Data) != 1 || catalog.Data[0].DisplayName != "qwen3.8:27b" {
		t.Fatalf("signed-out catalog = %+v, want local model only", catalog.Data)
	}
	if catalog.FirstID != catalog.Data[0].ID || catalog.LastID != catalog.Data[0].ID {
		t.Fatalf("signed-out catalog bounds = %q, %q", catalog.FirstID, catalog.LastID)
	}

	cloudModelsAvailable.Store(true)
	catalog = fetchCatalog()
	if len(catalog.Data) != len(models) {
		t.Fatalf("signed-in catalog has %d models, want %d", len(catalog.Data), len(models))
	}
}

func TestGatewayReevaluatesModelAccessBeforeRouting(t *testing.T) {
	var routed atomic.Int64
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		routed.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"ok":true}`)
	}))
	defer upstream.Close()

	state := ClaudeDesktopAccessState{
		Cloud:   ClaudeDesktopCloudOn,
		Account: ClaudeDesktopAccountSignedIn,
		Plan:    "free",
	}
	models := ClaudeDesktopModelsFromRecommendations([]api.ModelRecommendation{
		{Model: "glm-5.2:cloud", RequiredPlan: "pro"},
		{Model: "gemma4:31b-cloud", RequiredPlan: "free"},
	})
	p, err := NewClaudeDesktop(ClaudeDesktopConfig{
		ListenAddr: "127.0.0.1:0",
		OllamaURL:  upstream.URL,
		Model:      models[0].OllamaModel,
		Models:     models,
		ResolveAccessState: func(context.Context) (ClaudeDesktopAccessState, error) {
			return state, nil
		},
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

	postMessage := func(model string) (int, string) {
		t.Helper()
		resp, err := http.Post(
			"http://"+p.Addr()+"/v1/messages",
			"application/json",
			strings.NewReader(`{"model":"`+model+`","messages":[{"role":"user","content":"hi"}]}`),
		)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			t.Fatal(err)
		}
		return resp.StatusCode, string(body)
	}
	assertMessage := func(model string, status int, want string) {
		t.Helper()
		gotStatus, body := postMessage(model)
		if gotStatus != status || !strings.Contains(body, want) {
			t.Fatalf("status/body = %d/%q, want %d containing %q", gotStatus, body, status, want)
		}
	}

	assertMessage("claude-fable-5", http.StatusForbidden, "requires an Ollama pro plan")
	if routed.Load() != 0 {
		t.Fatalf("ineligible Pro request reached upstream")
	}

	assertMessage("claude-opus-5", http.StatusOK, `"ok":true`)
	if routed.Load() != 1 {
		t.Fatalf("free model routed %d times, want 1", routed.Load())
	}

	// Account and Cloud settings can change while Claude is open. Each request
	// resolves them again so an old conversation cannot retain stale access.
	state.Cloud = ClaudeDesktopCloudOff
	assertMessage("claude-opus-5", http.StatusForbidden, "Turn on Cloud in Ollama Settings")
	if routed.Load() != 1 {
		t.Fatalf("Cloud-disabled request reached upstream")
	}
}

func assertResponseContains(t *testing.T, response *http.Response, status int, want string) {
	t.Helper()
	defer response.Body.Close()
	body, err := io.ReadAll(response.Body)
	if err != nil {
		t.Fatal(err)
	}
	if response.StatusCode != status || !strings.Contains(string(body), want) {
		t.Fatalf("response = %d %s, want %d containing %q", response.StatusCode, body, status, want)
	}
}

func TestGatewaySetModelsReplacesCatalogAndRoutesExactModel(t *testing.T) {
	var routedModel string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		routedModel = requestModel(t, r)
		_, _ = io.WriteString(w, `{"ok":true}`)
	}))
	defer upstream.Close()
	p := startTestGateway(t, upstream.URL)

	available := ClaudeDesktopModelsFromRecommendations([]api.ModelRecommendation{
		{Model: "glm-5.2:cloud", RequiredPlan: "pro"},
		{Model: "qwen3.8:27b"},
	})
	selected := SelectClaudeDesktopModels(available, []string{"qwen3.8:27b"})
	if err := p.SetModels(selected); err != nil {
		t.Fatal(err)
	}

	resp, err := http.Get("http://" + p.Addr() + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var catalog struct {
		Data []gatewayModel `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&catalog); err != nil {
		t.Fatal(err)
	}
	if len(catalog.Data) != 1 || catalog.Data[0].DisplayName != "qwen3.8:27b" {
		t.Fatalf("catalog = %+v", catalog.Data)
	}

	message, err := http.Post(
		"http://"+p.Addr()+"/v1/messages",
		"application/json",
		strings.NewReader(`{"model":"qwen3.8:27b","messages":[]}`),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer message.Body.Close()
	if message.StatusCode != http.StatusOK || routedModel != "qwen3.8:27b" {
		t.Fatalf("status/model = %d/%q", message.StatusCode, routedModel)
	}
}

func TestGatewayRefreshesEntitlementsWithoutRestart(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{"ok":true}`)
	}))
	defer upstream.Close()
	initial := ClaudeDesktopModelsFromRecommendations([]api.ModelRecommendation{{Model: "changing-model:cloud", RequiredPlan: "free"}})
	updated := ClaudeDesktopModelsFromRecommendations([]api.ModelRecommendation{{Model: "changing-model:cloud", RequiredPlan: "pro"}})
	p, err := NewClaudeDesktop(ClaudeDesktopConfig{
		ListenAddr: "127.0.0.1:0",
		OllamaURL:  upstream.URL,
		Model:      initial[0].OllamaModel,
		Models:     initial,
		RefreshModels: func(context.Context, []ClaudeDesktopModel) ([]ClaudeDesktopModel, error) {
			return updated, nil
		},
		ResolveAccessState: func(context.Context) (ClaudeDesktopAccessState, error) {
			return ClaudeDesktopAccessState{Cloud: ClaudeDesktopCloudOn, Account: ClaudeDesktopAccountSignedIn, Plan: "free"}, nil
		},
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
		_ = p.Close(ctx)
	})

	resp, err := http.Post(
		"http://"+p.Addr()+"/v1/messages",
		"application/json",
		strings.NewReader(`{"model":"claude-fable-5","messages":[]}`),
	)
	if err != nil {
		t.Fatal(err)
	}
	assertResponseContains(t, resp, http.StatusOK, `"ok":true`)

	resp, err = http.Post(
		"http://"+p.Addr()+"/v1/messages",
		"application/json",
		strings.NewReader(`{"model":"claude-fable-5","messages":[]}`),
	)
	if err != nil {
		t.Fatal(err)
	}
	assertResponseContains(t, resp, http.StatusForbidden, "requires an Ollama pro plan")
}

func TestGatewayDiscardsRefreshThatRacesWithModelUpdate(t *testing.T) {
	initial := ClaudeDesktopModelsFromRecommendations([]api.ModelRecommendation{{Model: "model-a:cloud", RequiredPlan: "free"}})
	updated := ClaudeDesktopModelsFromRecommendations([]api.ModelRecommendation{{Model: "model-b:cloud", RequiredPlan: "free"}})
	started := make(chan struct{})
	release := make(chan struct{})
	p, err := NewClaudeDesktop(ClaudeDesktopConfig{
		ListenAddr: "127.0.0.1:0",
		OllamaURL:  "http://127.0.0.1:11434",
		Model:      initial[0].OllamaModel,
		Models:     initial,
		RefreshModels: func(_ context.Context, current []ClaudeDesktopModel) ([]ClaudeDesktopModel, error) {
			close(started)
			<-release
			return current, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { done <- p.refreshModelCatalog(context.Background()) }()
	<-started
	if err := p.SetModels(updated); err != nil {
		t.Fatal(err)
	}
	close(release)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	models := p.Models()
	if len(models) != 1 || models[0].OllamaModel != "model-b:cloud" {
		t.Fatalf("models after racing refresh = %+v, want model B", models)
	}
}

func TestGatewayContinuesRequestWhenSlotChangesDuringRefresh(t *testing.T) {
	var upstreamCalls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		upstreamCalls.Add(1)
	}))
	defer upstream.Close()
	initial := ClaudeDesktopModelsFromRecommendations([]api.ModelRecommendation{{Model: "model-a:cloud", RequiredPlan: "free"}})
	updated := ClaudeDesktopModelsFromRecommendations([]api.ModelRecommendation{{Model: "model-b:cloud", RequiredPlan: "free"}})
	started := make(chan struct{})
	release := make(chan struct{})
	p, err := NewClaudeDesktop(ClaudeDesktopConfig{
		ListenAddr: "127.0.0.1:0",
		OllamaURL:  upstream.URL,
		Model:      initial[0].OllamaModel,
		Models:     initial,
		RefreshModels: func(_ context.Context, current []ClaudeDesktopModel) ([]ClaudeDesktopModel, error) {
			close(started)
			<-release
			return current, nil
		},
		ResolveAccessState: func(context.Context) (ClaudeDesktopAccessState, error) {
			return ClaudeDesktopAccessState{Cloud: ClaudeDesktopCloudOn, Account: ClaudeDesktopAccountSignedIn, Plan: "free"}, nil
		},
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
		_ = p.Close(ctx)
	})

	type messageResponse struct {
		status int
		body   string
	}
	response := make(chan messageResponse, 1)
	requestErr := make(chan error, 1)
	go func() {
		resp, err := http.Post(
			"http://"+p.Addr()+"/v1/messages",
			"application/json",
			strings.NewReader(`{"model":"claude-fable-5","messages":[]}`),
		)
		if err != nil {
			requestErr <- err
			return
		}
		defer resp.Body.Close()
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			requestErr <- err
			return
		}
		response <- messageResponse{status: resp.StatusCode, body: string(body)}
	}()
	<-started
	if err := p.SetModels(updated); err != nil {
		t.Fatal(err)
	}
	close(release)
	select {
	case err := <-requestErr:
		t.Fatal(err)
	case resp := <-response:
		if resp.status != http.StatusOK {
			t.Fatalf("response = (%d, %q), want status %d", resp.status, resp.body, http.StatusOK)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for request")
	}
	if got := upstreamCalls.Load(); got != 1 {
		t.Fatalf("upstream calls = %d, want 1 after a slot change", got)
	}
}

func TestGatewayContinuesTokenCountWhenSlotChangesDuringRefresh(t *testing.T) {
	initial := ClaudeDesktopModelsFromRecommendations([]api.ModelRecommendation{{Model: "model-a:cloud", RequiredPlan: "free"}})
	updated := ClaudeDesktopModelsFromRecommendations([]api.ModelRecommendation{{Model: "model-b:cloud", RequiredPlan: "free"}})
	started := make(chan struct{})
	release := make(chan struct{})
	p, err := NewClaudeDesktop(ClaudeDesktopConfig{
		ListenAddr: "127.0.0.1:0",
		OllamaURL:  "http://127.0.0.1:11434",
		Model:      initial[0].OllamaModel,
		Models:     initial,
		RefreshModels: func(_ context.Context, current []ClaudeDesktopModel) ([]ClaudeDesktopModel, error) {
			close(started)
			<-release
			return current, nil
		},
		ResolveAccessState: func(context.Context) (ClaudeDesktopAccessState, error) {
			return ClaudeDesktopAccessState{Cloud: ClaudeDesktopCloudOn, Account: ClaudeDesktopAccountSignedIn, Plan: "free"}, nil
		},
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
		_ = p.Close(ctx)
	})

	type tokenCountResponse struct {
		status int
		body   []byte
	}
	response := make(chan tokenCountResponse, 1)
	requestErr := make(chan error, 1)
	go func() {
		resp, err := http.Post(
			"http://"+p.Addr()+"/v1/messages/count_tokens",
			"application/json",
			strings.NewReader(`{"model":"claude-fable-5","messages":[{"role":"user","content":"hello"}]}`),
		)
		if err != nil {
			requestErr <- err
			return
		}
		body, err := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if err != nil {
			requestErr <- err
			return
		}
		response <- tokenCountResponse{status: resp.StatusCode, body: body}
	}()
	<-started
	if err := p.SetModels(updated); err != nil {
		t.Fatal(err)
	}
	close(release)

	select {
	case err := <-requestErr:
		t.Fatal(err)
	case resp := <-response:
		if resp.status != http.StatusOK {
			t.Fatalf("response = (%d, %q), want status %d", resp.status, resp.body, http.StatusOK)
		}
		var result anthropic.CountTokensResponse
		if err := json.Unmarshal(resp.body, &result); err != nil {
			t.Fatal(err)
		}
		if result.InputTokens <= 0 {
			t.Fatalf("input_tokens = %d, want positive estimate", result.InputTokens)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for token-count response")
	}
}

func TestGatewayRoutesAdmittedRequestAfterSlotReassignment(t *testing.T) {
	for _, tt := range []struct {
		name    string
		initial []ClaudeDesktopModel
		updated []ClaudeDesktopModel
	}{
		{
			name:    "cloud",
			initial: ClaudeDesktopModelsFromRecommendations([]api.ModelRecommendation{{Model: "cloud-a:cloud", RequiredPlan: "free"}}),
			updated: ClaudeDesktopModelsFromRecommendations([]api.ModelRecommendation{{Model: "cloud-b:cloud", RequiredPlan: "free"}}),
		},
		{
			name:    "local",
			initial: SelectClaudeDesktopModels(nil, []string{"local-a"}),
			updated: SelectClaudeDesktopModels(nil, []string{"local-b"}),
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			p, err := NewClaudeDesktop(ClaudeDesktopConfig{
				ListenAddr: "127.0.0.1:0",
				OllamaURL:  "http://127.0.0.1:11434",
				Model:      tt.initial[0].OllamaModel,
				Models:     tt.initial,
			})
			if err != nil {
				t.Fatal(err)
			}
			_, admitted := p.modelSnapshot()
			if err := p.SetModels(tt.updated); err != nil {
				t.Fatal(err)
			}
			request := httptest.NewRequest(
				http.MethodPost,
				"http://127.0.0.1/v1/messages",
				strings.NewReader(`{"model":"claude-fable-5","messages":[]}`),
			)
			if err := p.routeModel(request, admitted); err != nil {
				t.Fatal(err)
			}
			var payload struct {
				Model string `json:"model"`
			}
			if err := json.NewDecoder(request.Body).Decode(&payload); err != nil {
				t.Fatal(err)
			}
			if payload.Model != tt.initial[0].OllamaModel {
				t.Fatalf("routed model = %q, want admitted model %q", payload.Model, tt.initial[0].OllamaModel)
			}
		})
	}
}

func TestGatewayLocalRequestSkipsAccountResolution(t *testing.T) {
	var accessCalls atomic.Int32
	var refreshCalls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{"ok":true}`)
	}))
	defer upstream.Close()
	models := SelectClaudeDesktopModels(nil, []string{"qwen3:8b"})
	p, err := NewClaudeDesktop(ClaudeDesktopConfig{
		ListenAddr: "127.0.0.1:0",
		OllamaURL:  upstream.URL,
		Model:      models[0].OllamaModel,
		Models:     models,
		RefreshModels: func(context.Context, []ClaudeDesktopModel) ([]ClaudeDesktopModel, error) {
			refreshCalls.Add(1)
			return models, nil
		},
		ResolveAccessState: func(context.Context) (ClaudeDesktopAccessState, error) {
			accessCalls.Add(1)
			return ClaudeDesktopAccessState{}, errors.New("offline")
		},
		ListLocalModels: func(context.Context) ([]string, error) {
			return []string{"qwen3:8b"}, nil
		},
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
		_ = p.Close(ctx)
	})
	discovery, err := http.Get("http://" + p.Addr() + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	assertResponseContains(t, discovery, http.StatusOK, "qwen3:8b")

	resp, err := http.Post(
		"http://"+p.Addr()+"/v1/messages",
		"application/json",
		strings.NewReader(`{"model":"claude-fable-5","messages":[]}`),
	)
	if err != nil {
		t.Fatal(err)
	}
	assertResponseContains(t, resp, http.StatusOK, `"ok":true`)
	if got := accessCalls.Load(); got != 0 {
		t.Fatalf("account resolution calls = %d, want 0 for a local model", got)
	}
	if got := refreshCalls.Load(); got != 0 {
		t.Fatalf("recommendation refresh calls = %d, want 0 for an all-local catalog", got)
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
		strings.NewReader(`{"model":"glm-5.2:cloud","messages":[{"role":"user","content":"hello world"}]}`),
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

func TestGatewayRejectsUnknownModelRequests(t *testing.T) {
	var upstreamCalls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		upstreamCalls.Add(1)
	}))
	defer upstream.Close()
	p := startTestGateway(t, upstream.URL)

	for _, path := range []string{"/v1/messages", "/v1/messages/count_tokens"} {
		resp, err := http.Post(
			"http://"+p.Addr()+path,
			"application/json",
			strings.NewReader(`{"model":"claude-stale","messages":[]}`),
		)
		if err != nil {
			t.Fatal(err)
		}
		body, err := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if err != nil {
			t.Fatal(err)
		}
		if resp.StatusCode != http.StatusBadRequest || !strings.Contains(string(body), "unknown Claude model") {
			t.Fatalf("%s status/body = %d/%q", path, resp.StatusCode, body)
		}
	}
	if got := upstreamCalls.Load(); got != 0 {
		t.Fatalf("upstream calls = %d, want 0", got)
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

func TestGatewayRejectsRequestsWithOrigin(t *testing.T) {
	var upstreamCalls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		upstreamCalls.Add(1)
		w.Header().Set("Access-Control-Allow-Origin", "*")
	}))
	defer upstream.Close()
	p := startTestGateway(t, upstream.URL)

	messageBody, err := json.Marshal(map[string]any{
		"model":    p.Models()[0].GatewayID(),
		"messages": []any{},
	})
	if err != nil {
		t.Fatal(err)
	}

	for _, test := range []struct {
		name   string
		method string
		path   string
		origin string
		body   string
	}{
		{"messages from public site", http.MethodPost, "/v1/messages", "https://attacker.example", string(messageBody)},
		{"messages from localhost site", http.MethodPost, "/v1/messages", "http://localhost:3000", string(messageBody)},
		{"messages with opaque origin", http.MethodPost, "/v1/messages", "null", string(messageBody)},
		{"models", http.MethodGet, "/v1/models", "https://attacker.example", ""},
		{"token count", http.MethodPost, "/v1/messages/count_tokens", "https://attacker.example", string(messageBody)},
		{"health", http.MethodGet, healthPath, "https://attacker.example", ""},
		{"preflight", http.MethodOptions, "/v1/messages", "https://attacker.example", ""},
	} {
		t.Run(test.name, func(t *testing.T) {
			req, err := http.NewRequest(test.method, "http://"+p.Addr()+test.path, strings.NewReader(test.body))
			if err != nil {
				t.Fatal(err)
			}
			req.Header.Set("Origin", test.origin)
			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				t.Fatal(err)
			}
			defer resp.Body.Close()
			if resp.StatusCode != http.StatusForbidden {
				t.Fatalf("status = %d, want 403", resp.StatusCode)
			}
			if got := resp.Header.Get("Access-Control-Allow-Origin"); got != "" {
				t.Fatalf("Access-Control-Allow-Origin = %q, want empty", got)
			}
		})
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
			body:     `{"model":"deepseek-v4-pro:cloud","messages":[{"role":"user","content":[{"type":"text","text":"Describe this"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"aW1hZ2U="}}]}],"stream":true}`,
			preserve: "Describe this",
		},
		{
			name:     "tool result image",
			body:     `{"model":"glm-5.2:cloud","messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"tool_1","content":[{"type":"text","text":"Screenshot"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"aW1hZ2U="}}]}]}]}`,
			preserve: "tool_1",
		},
		{
			name:     "historical image with later text",
			body:     `{"model":"glm-5.2:cloud","messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"aW1hZ2U="}},{"type":"text","text":"what dis"}]},{"role":"assistant","content":[{"type":"text","text":"I cannot inspect it"}]},{"role":"user","content":[{"type":"text","text":"kk"}]}]}`,
			preserve: "kk",
		},
		{
			name:     "string message history",
			body:     `{"model":"glm-5.2:cloud","messages":[{"role":"user","content":"plain history"},{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"aW1hZ2U="}},{"type":"text","text":"what dis"}]}]}`,
			preserve: "plain history",
		},
		{
			name:     "string tool result history",
			body:     `{"model":"glm-5.2:cloud","messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"tool_1","content":"plain tool output"}]},{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"aW1hZ2U="}},{"type":"text","text":"what dis"}]}]}`,
			preserve: "plain tool output",
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

func TestReplaceImagesInContentRejectsUnsupportedShape(t *testing.T) {
	_, _, err := replaceImagesInContent(json.RawMessage(`{"type":"text","text":"not a content array"}`))
	if err == nil {
		t.Fatal("expected unsupported content shape to fail")
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

	body := `{"model":"glm-5.2:cloud","messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"aW1hZ2U="}},{"type":"text","text":"what dis"}]}]}`
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

	body := `{"model":"glm-5.2:cloud","messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}]}`
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

	body := `{"model":"glm-5.2:cloud","messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"aW1hZ2U="}}]}]}`
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

	body := `{"model":"glm-5.2:cloud","messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"aW1hZ2U="}}]}]}`
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

func TestGatewayRejectsUnknownModel(t *testing.T) {
	p := &ClaudeDesktop{models: DefaultClaudeDesktopModels()}
	if _, err := p.modelForClaudeID("claude-future"); err == nil || !strings.Contains(err.Error(), "unknown Claude model") {
		t.Fatalf("error = %v, want unknown Claude model", err)
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

	body := `{"model":"kimi-k3:cloud","messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"aW1hZ2U="}}]}]}`
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

	body := `{"model":"kimi-k3:cloud","messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"aW1hZ2U="}}]}]}`
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

	body := `{"model":"deepseek-v4-flash:0731:cloud","messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"aW1hZ2U="}}]}]}`
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
			strings.NewReader(`{"model":"deepseek-v4-pro:cloud","messages":[]}`),
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
