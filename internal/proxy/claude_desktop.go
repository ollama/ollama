package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/ollama/ollama/anthropic"
	"github.com/ollama/ollama/internal/modelref"
)

const (
	// DefaultClaudeDesktopListenAddr is the loopback address used by the Ollama
	// app and the Claude Desktop profile it manages.
	DefaultClaudeDesktopListenAddr = "127.0.0.1:11435"
	maxRequestBodyBytes            = 64 << 20
	maxErrorBodyBytes              = 64 << 10
	upstreamReadyTimeout           = 10 * time.Second
	upstreamReadyPoll              = 100 * time.Millisecond
	upstreamReadyTTL               = 5 * time.Second
	healthPath                     = "/_ollama/health"
	healthHeader                   = "X-Ollama-Claude-Gateway"
	unsupportedImageNotice         = "[Image omitted by Ollama because the selected model does not support image recognition.]"
)

// claudeModels are Claude-compatible IDs for Ollama models
var claudeModels = []gatewayModel{
	{
		ID:                  "claude-opus-5",
		Type:                "model",
		DisplayName:         "Kimi K3",
		CreatedAt:           "2026-07-24T00:00:00Z",
		MaxTokens:           262_144,
		AnthropicFamilyTier: "opus",
		IsFamilyDefault:     true,
		OllamaModel:         "kimi-k3:cloud",
		SupportsVision:      true,
	},
	{
		ID:                  "claude-fable-5",
		Type:                "model",
		DisplayName:         "GLM 5.2",
		CreatedAt:           "2026-06-09T00:00:00Z",
		MaxTokens:           128_000,
		AnthropicFamilyTier: "fable",
		IsFamilyDefault:     true,
		OllamaModel:         "glm-5.2:cloud",
	},
	{
		ID:                  "claude-sonnet-5",
		Type:                "model",
		DisplayName:         "DeepSeek V4 Pro",
		CreatedAt:           "2026-06-30T00:00:00Z",
		MaxTokens:           128_000,
		AnthropicFamilyTier: "sonnet",
		IsFamilyDefault:     true,
		OllamaModel:         "deepseek-v4-pro:cloud",
	},
	{
		ID:                  "claude-haiku-4-5-20251001",
		Type:                "model",
		DisplayName:         "DeepSeek V4 Flash",
		CreatedAt:           "2025-10-01T00:00:00Z",
		MaxTokens:           64_000,
		AnthropicFamilyTier: "haiku",
		IsFamilyDefault:     true,
		OllamaModel:         "deepseek-v4-flash:0731:cloud",
	},
	{
		ID:                  "claude-sonnet-5-20251001",
		Type:                "model",
		DisplayName:         "Qwen3.8 MLX",
		CreatedAt:           "2025-10-01T00:00:00Z",
		MaxTokens:           64_000,
		AnthropicFamilyTier: "sonnet",
		IsFamilyDefault:     false,
		OllamaModel:         "qwen3.8:27b-mlx",
	},
}

type gatewayModel struct {
	ID                  string `json:"id"`
	Type                string `json:"type"`
	DisplayName         string `json:"display_name"`
	CreatedAt           string `json:"created_at"`
	MaxTokens           int    `json:"max_tokens"`
	AnthropicFamilyTier string `json:"anthropic_family_tier"`
	IsFamilyDefault     bool   `json:"is_family_default"`
	OllamaModel         string `json:"-"`
	SupportsVision      bool   `json:"-"`
}

// ClaudeDesktopConfig configures a Claude Desktop proxy instance.
type ClaudeDesktopConfig struct {
	ListenAddr           string
	OllamaURL            string
	Model                string
	Logger               *slog.Logger
	OnCountsChanged      func(ClaudeDesktopCounts)
	CloudModelsAvailable func(context.Context) bool
}

// ClaudeDesktopCounts reports requests routed through the proxy.
type ClaudeDesktopCounts struct {
	Routed uint64
}

type upstreamReadinessCheck struct {
	done chan struct{}
	err  error
}

// ClaudeDesktop is the local gateway that Claude Desktop reaches through its
// native third-party inference mode. It is deliberately an ordinary HTTP reverse
// proxy: Claude terminates its own gateway protocol here, so no TLS
// interception or system trust changes are necessary.
type ClaudeDesktop struct {
	listenAddr           string
	ollamaURL            *url.URL
	ollamaProxy          *httputil.ReverseProxy
	logger               *slog.Logger
	model                string
	onCountsChanged      func(ClaudeDesktopCounts)
	cloudModelsAvailable func(context.Context) bool
	routed               atomic.Uint64
	shutdown             chan struct{}
	shutdownOnce         sync.Once

	readyMu    sync.Mutex
	readyUntil time.Time
	readyCheck *upstreamReadinessCheck
	readyWait  time.Duration
	readyPoll  time.Duration
	dial       func(context.Context, string, string) (net.Conn, error)

	mu       sync.Mutex
	listener net.Listener
	server   *http.Server
}

// NewClaudeDesktop creates a Claude Desktop proxy.
func NewClaudeDesktop(config ClaudeDesktopConfig) (*ClaudeDesktop, error) {
	if strings.TrimSpace(config.ListenAddr) == "" {
		return nil, errors.New("Claude gateway listen address is required")
	}
	if strings.TrimSpace(config.Model) == "" {
		return nil, errors.New("Claude gateway model is required")
	}

	ollamaURL, err := url.Parse(config.OllamaURL)
	if err != nil {
		return nil, fmt.Errorf("parse Ollama URL: %w", err)
	}
	if ollamaURL.Scheme == "" || ollamaURL.Host == "" {
		return nil, fmt.Errorf("invalid Ollama URL %q", config.OllamaURL)
	}

	logger := config.Logger
	if logger == nil {
		logger = slog.Default()
	}
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.Proxy = nil
	proxy := httputil.NewSingleHostReverseProxy(ollamaURL)
	proxy.Transport = transport
	proxy.FlushInterval = -1
	p := &ClaudeDesktop{
		listenAddr:           config.ListenAddr,
		ollamaURL:            ollamaURL,
		logger:               logger,
		model:                config.Model,
		onCountsChanged:      config.OnCountsChanged,
		cloudModelsAvailable: config.CloudModelsAvailable,
		shutdown:             make(chan struct{}),
		readyWait:            upstreamReadyTimeout,
		readyPoll:            upstreamReadyPoll,
	}
	dialer := &net.Dialer{Timeout: upstreamReadyPoll}
	p.dial = dialer.DialContext
	proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
		p.markUpstreamNotReady()
		logger.Warn("Claude gateway upstream request failed", "path", r.URL.Path, "error", err)
		http.Error(w, "Ollama gateway unavailable", http.StatusBadGateway)
	}
	proxy.ModifyResponse = func(response *http.Response) error {
		retried, err := retryWithoutUnsupportedImages(response, transport)
		if err != nil {
			logger.Warn("Claude gateway image fallback failed", "path", response.Request.URL.Path, "error", err)
			return nil
		}
		if retried {
			logger.Debug("Claude gateway retried request without unsupported images", "path", response.Request.URL.Path)
		}
		return nil
	}

	p.ollamaProxy = proxy
	return p, nil
}

// ProbeClaudeDesktop verifies that baseURL is served by Ollama's Claude gateway rather than
// another process using the configured port.
func ProbeClaudeDesktop(ctx context.Context, baseURL string) error {
	u, err := url.Parse(strings.TrimRight(baseURL, "/") + healthPath)
	if err != nil || u.Scheme == "" || u.Host == "" {
		return fmt.Errorf("invalid Claude gateway URL %q", baseURL)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	if err != nil {
		return fmt.Errorf("create Claude gateway health request: %w", err)
	}
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.Proxy = nil
	defer transport.CloseIdleConnections()
	resp, err := (&http.Client{Transport: transport}).Do(req)
	if err != nil {
		return fmt.Errorf("connect to Claude gateway: %w", err)
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 4<<10))
	if resp.StatusCode != http.StatusNoContent || resp.Header.Get(healthHeader) != "1" {
		return fmt.Errorf("another service is using %s", u.Host)
	}
	return nil
}

func (p *ClaudeDesktop) Start() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.listener != nil {
		return errors.New("Claude gateway already started")
	}

	listener, err := net.Listen("tcp", p.listenAddr)
	if err != nil {
		return fmt.Errorf("listen for Claude gateway: %w", err)
	}
	p.listener = listener
	p.server = &http.Server{
		Handler:           p,
		ReadHeaderTimeout: 10 * time.Second,
	}
	go func() {
		if err := p.server.Serve(listener); err != nil && !errors.Is(err, http.ErrServerClosed) {
			p.logger.Warn("Claude gateway stopped", "error", err)
		}
	}()
	p.logger.Info("Claude gateway started", "address", listener.Addr().String())
	return nil
}

func (p *ClaudeDesktop) Close(ctx context.Context) error {
	p.shutdownOnce.Do(func() { close(p.shutdown) })
	p.mu.Lock()
	server := p.server
	p.mu.Unlock()
	var err error
	if server != nil {
		err = server.Shutdown(ctx)
	}
	if transport, ok := p.ollamaProxy.Transport.(*http.Transport); ok {
		transport.CloseIdleConnections()
	}
	return err
}

func (p *ClaudeDesktop) Addr() string {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.listener == nil {
		return ""
	}
	return p.listener.Addr().String()
}

func (p *ClaudeDesktop) Counts() ClaudeDesktopCounts {
	return ClaudeDesktopCounts{Routed: p.routed.Load()}
}

func (p *ClaudeDesktop) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if !p.allowsHost(r.Host) {
		http.Error(w, "forbidden", http.StatusForbidden)
		return
	}

	model := p.model
	switch r.URL.Path {
	case healthPath:
		if r.Method != http.MethodGet {
			methodNotAllowed(w, http.MethodGet)
			return
		}
		w.Header().Set(healthHeader, "1")
		w.WriteHeader(http.StatusNoContent)
		return
	case "/v1/models":
		if r.Method != http.MethodGet {
			methodNotAllowed(w, http.MethodGet)
			return
		}
		p.serveModels(w, r.Context())
		return
	case "/v1/messages/count_tokens":
		if r.Method != http.MethodPost {
			methodNotAllowed(w, http.MethodPost)
			return
		}
		p.serveTokenCount(w, r, model)
		return
	case "/v1/messages":
		if r.Method != http.MethodPost {
			methodNotAllowed(w, http.MethodPost)
			return
		}
		if err := p.routeModel(r, model); err != nil {
			writeAnthropicError(w, http.StatusBadRequest, err)
			return
		}
	default:
		http.NotFound(w, r)
		return
	}

	// Claude authenticates to this loopback gateway with a placeholder key.
	// Never forward that credential (or browser cookies) to the Ollama daemon.
	r.Header.Del("Authorization")
	r.Header.Del("Cookie")
	r.Header.Del("Proxy-Authorization")
	r.Header.Set("X-Api-Key", "ollama")
	r.Host = p.ollamaURL.Host

	if err := p.waitForUpstream(r.Context()); err != nil {
		p.logger.Warn("Claude gateway upstream is not ready", "path", r.URL.Path, "error", err)
		http.Error(w, "Ollama gateway unavailable", http.StatusBadGateway)
		return
	}
	counts := ClaudeDesktopCounts{Routed: p.routed.Add(1)}
	if p.onCountsChanged != nil {
		p.onCountsChanged(counts)
	}
	p.ollamaProxy.ServeHTTP(w, r)
}

func (p *ClaudeDesktop) allowsHost(host string) bool {
	_, listenerPort, err := net.SplitHostPort(p.Addr())
	if err != nil {
		return false
	}

	host, port, err := net.SplitHostPort(host)
	if err != nil || port != listenerPort {
		return false
	}
	if strings.EqualFold(host, "localhost") {
		return true
	}

	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

func methodNotAllowed(w http.ResponseWriter, allow string) {
	w.Header().Set("Allow", allow)
	http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
}

func (p *ClaudeDesktop) waitForUpstream(ctx context.Context) error {
	p.readyMu.Lock()
	if time.Now().Before(p.readyUntil) {
		p.readyMu.Unlock()
		return nil
	}
	check := p.readyCheck
	if check == nil {
		check = &upstreamReadinessCheck{done: make(chan struct{})}
		p.readyCheck = check
		go p.runUpstreamReadinessCheck(check)
	}
	p.readyMu.Unlock()

	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-check.done:
		return check.err
	}
}

func (p *ClaudeDesktop) runUpstreamReadinessCheck(check *upstreamReadinessCheck) {
	ctx, cancel := context.WithTimeout(context.Background(), p.readyWait)
	defer cancel()
	ticker := time.NewTicker(p.readyPoll)
	defer ticker.Stop()

	var err error
checkLoop:
	for {
		select {
		case <-p.shutdown:
			err = errors.New("Claude gateway stopped")
			break checkLoop
		default:
		}

		conn, dialErr := p.dial(ctx, "tcp", p.ollamaURL.Host)
		if dialErr == nil {
			_ = conn.Close()
			select {
			case <-p.shutdown:
				err = errors.New("Claude gateway stopped")
			default:
			}
			break
		}

		select {
		case <-p.shutdown:
			err = errors.New("Claude gateway stopped")
			break checkLoop
		case <-ctx.Done():
			err = fmt.Errorf("timed out waiting for %s", p.ollamaURL.Host)
			break checkLoop
		case <-ticker.C:
		}
	}
	p.readyMu.Lock()
	if err == nil {
		p.readyUntil = time.Now().Add(upstreamReadyTTL)
	}
	check.err = err
	if p.readyCheck == check {
		p.readyCheck = nil
	}
	close(check.done)
	p.readyMu.Unlock()
}

func (p *ClaudeDesktop) markUpstreamNotReady() {
	p.readyMu.Lock()
	p.readyUntil = time.Time{}
	p.readyMu.Unlock()
}

func (p *ClaudeDesktop) serveModels(w http.ResponseWriter, ctx context.Context) {
	models := claudeModels
	if p.cloudModelsAvailable != nil && !p.cloudModelsAvailable(ctx) {
		models = make([]gatewayModel, 0, len(claudeModels))
		for _, model := range claudeModels {
			if !modelref.HasExplicitCloudSource(model.OllamaModel) {
				models = append(models, model)
			}
		}
	}

	var firstID, lastID string
	if len(models) > 0 {
		firstID = models[0].ID
		lastID = models[len(models)-1].ID
	}
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(struct {
		Data    []gatewayModel `json:"data"`
		FirstID string         `json:"first_id"`
		LastID  string         `json:"last_id"`
		HasMore bool           `json:"has_more"`
	}{
		Data:    models,
		FirstID: firstID,
		LastID:  lastID,
	}); err != nil {
		p.logger.Debug("write Claude model catalog", "error", err)
	}
}

func (p *ClaudeDesktop) serveTokenCount(w http.ResponseWriter, r *http.Request, model string) {
	body, err := readRequestBody(r)
	if err != nil {
		writeAnthropicError(w, http.StatusBadRequest, err)
		return
	}
	var request anthropic.CountTokensRequest
	if err := json.Unmarshal(body, &request); err != nil {
		writeAnthropicError(w, http.StatusBadRequest, fmt.Errorf("decode Claude token-count request: %w", err))
		return
	}
	selected := p.modelForClaudeID(request.Model, model)
	request.Model = selected.OllamaModel
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(anthropic.CountTokensResponse{
		InputTokens: anthropic.EstimateCountTokens(request),
	}); err != nil {
		p.logger.Debug("write Claude token-count response", "error", err)
	}
}

func (p *ClaudeDesktop) routeModel(r *http.Request, model string) error {
	body, err := readRequestBody(r)
	if err != nil {
		return err
	}
	var payload map[string]json.RawMessage
	if err := json.Unmarshal(body, &payload); err != nil {
		return fmt.Errorf("decode Claude gateway request: %w", err)
	}
	var requestedModel string
	if err := json.Unmarshal(payload["model"], &requestedModel); err != nil || strings.TrimSpace(requestedModel) == "" {
		return errors.New("decode Claude gateway request: model is required")
	}
	selected := p.modelForClaudeID(requestedModel, model)
	targetModel := selected.OllamaModel
	if targetModel == requestedModel {
		setRequestBody(r, body)
		return nil
	}
	modelJSON, _ := json.Marshal(targetModel)
	payload["model"] = modelJSON
	rewritten, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("encode Claude gateway request: %w", err)
	}
	setRequestBody(r, rewritten)
	return nil
}

func retryWithoutUnsupportedImages(response *http.Response, transport http.RoundTripper) (bool, error) {
	if response.StatusCode != http.StatusBadRequest || response.Request.Method != http.MethodPost || response.Request.URL.Path != "/v1/messages" || response.Request.GetBody == nil {
		return false, nil
	}
	if !responseRejectsImages(response) {
		return false, nil
	}

	body, err := response.Request.GetBody()
	if err != nil {
		return false, fmt.Errorf("replay Claude gateway request: %w", err)
	}
	defer body.Close()
	rewrittenBody, err := io.ReadAll(io.LimitReader(body, maxRequestBodyBytes+1))
	if err != nil {
		return false, fmt.Errorf("read replayable Claude gateway request: %w", err)
	}
	if len(rewrittenBody) > maxRequestBodyBytes {
		return false, nil
	}

	var payload map[string]json.RawMessage
	if err := json.Unmarshal(rewrittenBody, &payload); err != nil {
		return false, nil
	}
	var targetModel string
	if err := json.Unmarshal(payload["model"], &targetModel); err != nil {
		return false, nil
	}
	selected, _ := gatewayModelForClaudeID(targetModel, targetModel)
	if selected.SupportsVision {
		return false, nil
	}

	changed, err := replaceUnsupportedImages(payload)
	if err != nil || !changed {
		return false, err
	}
	rewrittenBody, err = json.Marshal(payload)
	if err != nil {
		return false, fmt.Errorf("encode Claude image fallback request: %w", err)
	}

	retry := response.Request.Clone(response.Request.Context())
	setRequestBody(retry, rewrittenBody)
	retryResponse, err := transport.RoundTrip(retry)
	if err != nil {
		if retryResponse != nil && retryResponse.Body != nil {
			_ = retryResponse.Body.Close()
		}
		return false, fmt.Errorf("retry Claude gateway request: %w", err)
	}
	defer func() {
		if retryResponse.Body != nil {
			_ = retryResponse.Body.Close()
		}
	}()
	_ = response.Body.Close()
	*response = *retryResponse
	retryResponse.Body = nil
	return true, nil
}

func responseRejectsImages(response *http.Response) bool {
	if response.Body == nil {
		return false
	}
	body := response.Body
	prefix, err := io.ReadAll(io.LimitReader(body, maxErrorBodyBytes+1))
	response.Body = struct {
		io.Reader
		io.Closer
	}{Reader: io.MultiReader(bytes.NewReader(prefix), body), Closer: body}
	if err != nil || len(prefix) > maxErrorBodyBytes {
		return false
	}
	var responseError anthropic.ErrorResponse
	if err := json.Unmarshal(prefix, &responseError); err != nil ||
		responseError.Type != "error" || responseError.Error.Type != "invalid_request_error" {
		return false
	}
	lower := strings.ToLower(responseError.Error.Message)
	mentionsImages := strings.Contains(lower, "image") || strings.Contains(lower, "vision")
	unsupported := strings.Contains(lower, "does not support") ||
		strings.Contains(lower, "not support") ||
		strings.Contains(lower, "unsupported")
	return mentionsImages && unsupported
}

func setRequestBody(r *http.Request, body []byte) {
	_ = r.Body.Close()
	r.Body = io.NopCloser(bytes.NewReader(body))
	r.GetBody = func() (io.ReadCloser, error) {
		return io.NopCloser(bytes.NewReader(body)), nil
	}
	r.ContentLength = int64(len(body))
	r.Header.Set("Content-Length", fmt.Sprintf("%d", len(body)))
}

func (p *ClaudeDesktop) modelForClaudeID(id, fallback string) gatewayModel {
	model, known := gatewayModelForClaudeID(id, fallback)
	if !known {
		p.logger.Debug("Claude gateway using default for unknown model", "requested_model", id, "default_model", model.OllamaModel)
	}
	return model
}

func gatewayModelForClaudeID(id, fallback string) (gatewayModel, bool) {
	for _, model := range claudeModels {
		if model.ID == id {
			return model, true
		}
	}
	for _, model := range claudeModels {
		if model.OllamaModel == fallback {
			return model, false
		}
	}
	return gatewayModel{DisplayName: fallback, OllamaModel: fallback}, false
}

func replaceUnsupportedImages(payload map[string]json.RawMessage) (bool, error) {
	rawMessages, ok := payload["messages"]
	if !ok {
		return false, nil
	}

	var messages []json.RawMessage
	if err := json.Unmarshal(rawMessages, &messages); err != nil {
		return false, fmt.Errorf("decode Claude messages for image fallback: %w", err)
	}

	changed := false
	for i, rawMessage := range messages {
		var message map[string]json.RawMessage
		if err := json.Unmarshal(rawMessage, &message); err != nil {
			continue
		}
		content, ok := message["content"]
		if !ok {
			continue
		}
		rewritten, contentChanged, err := replaceImagesInContent(content)
		if err != nil {
			return false, err
		}
		if !contentChanged {
			continue
		}
		message["content"] = rewritten
		messages[i], err = json.Marshal(message)
		if err != nil {
			return false, fmt.Errorf("encode sanitized Claude message: %w", err)
		}
		changed = true
	}
	if !changed {
		return false, nil
	}

	rewritten, err := json.Marshal(messages)
	if err != nil {
		return false, fmt.Errorf("encode sanitized Claude messages: %w", err)
	}
	payload["messages"] = rewritten
	return true, nil
}

func replaceImagesInContent(content json.RawMessage) (json.RawMessage, bool, error) {
	var blocks []json.RawMessage
	if err := json.Unmarshal(content, &blocks); err != nil {
		return content, false, fmt.Errorf("decode Claude content for image fallback: %w", err)
	}

	changed := false
	for i, raw := range blocks {
		var block struct {
			Type    string          `json:"type"`
			Content json.RawMessage `json:"content"`
		}
		if err := json.Unmarshal(raw, &block); err != nil {
			continue
		}
		switch block.Type {
		case "image":
			replacement, err := json.Marshal(struct {
				Type string `json:"type"`
				Text string `json:"text"`
			}{Type: "text", Text: unsupportedImageNotice})
			if err != nil {
				return nil, false, fmt.Errorf("encode unsupported-image notice: %w", err)
			}
			blocks[i] = replacement
			changed = true
		case "tool_result":
			rewritten, contentChanged, err := replaceImagesInContent(block.Content)
			if err != nil {
				return nil, false, err
			}
			if !contentChanged {
				continue
			}
			var fields map[string]json.RawMessage
			if err := json.Unmarshal(raw, &fields); err != nil {
				continue
			}
			fields["content"] = rewritten
			blocks[i], err = json.Marshal(fields)
			if err != nil {
				return nil, false, fmt.Errorf("encode sanitized Claude tool result: %w", err)
			}
			changed = true
		}
	}
	if !changed {
		return content, false, nil
	}

	rewritten, err := json.Marshal(blocks)
	if err != nil {
		return nil, false, fmt.Errorf("encode sanitized Claude content: %w", err)
	}
	return rewritten, true, nil
}

func writeAnthropicError(w http.ResponseWriter, status int, err error) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if encodeErr := json.NewEncoder(w).Encode(anthropic.NewError(status, err.Error())); encodeErr != nil {
		slog.Debug("write Claude gateway error", "error", encodeErr)
	}
}

func readRequestBody(r *http.Request) ([]byte, error) {
	if r.Body == nil {
		return nil, errors.New("Claude gateway request body is required")
	}
	body, err := io.ReadAll(io.LimitReader(r.Body, maxRequestBodyBytes+1))
	if err != nil {
		return nil, fmt.Errorf("read Claude gateway request: %w", err)
	}
	if len(body) > maxRequestBodyBytes {
		return nil, fmt.Errorf("Claude gateway request exceeds %d bytes", maxRequestBodyBytes)
	}
	return body, nil
}
