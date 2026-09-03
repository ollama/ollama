package codexproxy

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/klauspost/compress/zstd"
)

const (
	// PathPrefix is the loopback-only Ollama server route used as Codex's
	// openai_base_url. Codex appends /v1/responses and related paths to it.
	PathPrefix = "/api/codex"

	// ModelCatalogFilename is the combined native and Ollama catalog shown by
	// the Codex model picker.
	ModelCatalogFilename = "ollama-launch-models.json"

	// RoutingCatalogFilename contains only Ollama model slugs. Keeping the
	// router allow-list separate from the picker catalog ensures native Codex
	// models continue to pass through to their normal OpenAI upstream.
	RoutingCatalogFilename = "ollama-launch-codex-routing.json"

	// ManagedAPIKey is a local sentinel used to let signed-out ChatGPT users
	// open Codex directly with Ollama models. It is never valid for an OpenAI
	// upstream and the router must reject it before any native request leaves
	// the machine.
	ManagedAPIKey = "ollama-local-codex"

	// autoReviewModel is the native Codex reviewer alias. The routing catalog
	// may map requests for this alias to a selected Ollama model.
	autoReviewModel          = "codex-auto-review"
	guardianDecisionToolName = "submit_guardian_decision"

	defaultMaxBodyBytes         = int64(64 << 20)
	defaultOpenAIURL            = "https://api.openai.com/v1"
	guardianDecisionInstruction = `

When you have finished reviewing the action, call submit_guardian_decision exactly once with your final decision. Do not return the final decision as assistant text.`
)

var guardianDecisionTool = map[string]any{
	"type":        "function",
	"name":        guardianDecisionToolName,
	"description": "Submit the final Codex Guardian approval decision after completing any necessary investigation.",
	"strict":      true,
	"parameters": map[string]any{
		"type": "object",
		"properties": map[string]any{
			"risk_level": map[string]any{
				"type": "string",
				"enum": []string{"low", "medium", "high", "critical"},
			},
			"user_authorization": map[string]any{
				"type": "string",
				"enum": []string{"unknown", "low", "medium", "high"},
			},
			"outcome": map[string]any{
				"type": "string",
				"enum": []string{"allow", "deny"},
			},
			"rationale": map[string]any{
				"type":      "string",
				"minLength": 1,
			},
		},
		"required":             []string{"risk_level", "user_authorization", "outcome", "rationale"},
		"additionalProperties": false,
	},
}

var hopByHopHeaders = map[string]struct{}{
	"connection":          {},
	"content-length":      {},
	"keep-alive":          {},
	"proxy-authenticate":  {},
	"proxy-authorization": {},
	"proxy-connection":    {},
	"te":                  {},
	"trailer":             {},
	"transfer-encoding":   {},
	"upgrade":             {},
}

// Config describes the upstreams and Ollama-only routing catalog.
type Config struct {
	PathPrefix         string
	OllamaURL          string
	ChatGPTURL         string
	OpenAIURL          string
	RoutingCatalogPath string
	ActivityLogPath    string
	MaxBodyBytes       int64
	Logger             *slog.Logger
	Transport          http.RoundTripper
}

// Handler routes catalog-listed model slugs to Ollama and passes every other
// request to the native upstream selected by Codex's authentication mode.
type Handler struct {
	pathPrefix         string
	ollamaURL          *url.URL
	chatGPTURL         *url.URL
	openAIURL          *url.URL
	routingCatalogPath string
	activityLogPath    string
	maxBodyBytes       int64
	logger             *slog.Logger
	client             *http.Client
	ollamaRequests     atomic.Uint64
	chatGPTRequests    atomic.Uint64
	upstreamErrors     atomic.Uint64
	lastRoute          atomic.Value
	activityLogMu      sync.Mutex
}

type routeSnapshot struct {
	Model          string
	Route          string
	UpstreamStatus int
}

type statusResponse struct {
	OK                 bool   `json:"ok"`
	OllamaRequests     uint64 `json:"ollama_requests"`
	ChatGPTRequests    uint64 `json:"chatgpt_requests"`
	UpstreamErrors     uint64 `json:"upstream_errors"`
	LastModel          string `json:"last_model,omitempty"`
	LastRoute          string `json:"last_route,omitempty"`
	LastUpstreamStatus int    `json:"last_upstream_status,omitempty"`
}

func New(config Config) (*Handler, error) {
	pathPrefix := strings.TrimRight(strings.TrimSpace(config.PathPrefix), "/")
	if pathPrefix == "" {
		pathPrefix = PathPrefix
	}
	if !strings.HasPrefix(pathPrefix, "/") {
		return nil, fmt.Errorf("Codex proxy path prefix must start with '/': %q", pathPrefix)
	}

	ollamaURL, err := parseBaseURL("Ollama", config.OllamaURL)
	if err != nil {
		return nil, err
	}
	chatGPTURL, err := parseBaseURL("ChatGPT", config.ChatGPTURL)
	if err != nil {
		return nil, err
	}
	openAIRawURL := strings.TrimSpace(config.OpenAIURL)
	if openAIRawURL == "" {
		openAIRawURL = defaultOpenAIURL
	}
	openAIURL, err := parseBaseURL("OpenAI API", openAIRawURL)
	if err != nil {
		return nil, err
	}

	maxBodyBytes := config.MaxBodyBytes
	if maxBodyBytes <= 0 {
		maxBodyBytes = defaultMaxBodyBytes
	}
	logger := config.Logger
	if logger == nil {
		logger = slog.Default()
	}
	transport := config.Transport
	if transport == nil {
		transport = http.DefaultTransport.(*http.Transport).Clone()
	}

	handler := &Handler{
		pathPrefix:         pathPrefix,
		ollamaURL:          ollamaURL,
		chatGPTURL:         chatGPTURL,
		openAIURL:          openAIURL,
		routingCatalogPath: config.RoutingCatalogPath,
		activityLogPath:    strings.TrimSpace(config.ActivityLogPath),
		maxBodyBytes:       maxBodyBytes,
		logger:             logger,
		client: &http.Client{
			Transport: transport,
			CheckRedirect: func(*http.Request, []*http.Request) error {
				return http.ErrUseLastResponse
			},
		},
	}
	return handler, nil
}

func parseBaseURL(name, raw string) (*url.URL, error) {
	u, err := url.Parse(strings.TrimSpace(raw))
	if err != nil {
		return nil, fmt.Errorf("parse %s URL: %w", name, err)
	}
	if u.Scheme == "" || u.Host == "" {
		return nil, fmt.Errorf("invalid %s URL %q", name, raw)
	}
	return u, nil
}

func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if !isLoopbackRequest(r) {
		writeJSONError(w, http.StatusForbidden, "Codex proxy only accepts loopback requests")
		return
	}

	suffix, ok := strings.CutPrefix(r.URL.Path, h.pathPrefix)
	if !ok || suffix == "" {
		http.NotFound(w, r)
		return
	}
	if suffix == "/_health" {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, `{"ok":true}`)
		return
	}
	if suffix == "/_status" {
		h.writeStatus(w)
		return
	}
	started := time.Now()
	if isWebSocketUpgrade(r) {
		// Codex's built-in OpenAI provider prefers the Responses WebSocket
		// transport. This proxy deliberately routes each complete HTTP request by
		// model, so tell Codex to use its supported HTTP fallback instead of
		// retrying a failed WebSocket connection. Codex treats 426 as a permanent
		// transport fallback for the current session.
		h.logActivity(started, r.Method, suffix, "", "none", http.StatusUpgradeRequired, "http_fallback")
		writeJSONError(w, http.StatusUpgradeRequired, "Codex proxy uses the HTTP Responses transport")
		return
	}

	rawBody, decodedBody, err := h.readBodies(r)
	if err != nil {
		h.logActivity(started, r.Method, suffix, "", "none", http.StatusBadRequest, "request_error")
		writeJSONError(w, http.StatusBadRequest, err.Error())
		return
	}

	model, hasModel := extractModel(decodedBody)
	isAutoReview := hasModel && modelKey(model) == modelKey(autoReviewModel)
	routed := false
	var routedModel routingModel
	if hasModel {
		catalog, err := loadRoutingCatalog(h.routingCatalogPath)
		if err != nil {
			h.logActivity(started, r.Method, suffix, model, "none", http.StatusServiceUnavailable, "catalog_error")
			writeJSONError(w, http.StatusServiceUnavailable, "read Codex Ollama model catalog: "+err.Error())
			return
		}
		if modelKey(model) == modelKey(autoReviewModel) && catalog.autoReviewModel != "" {
			model = catalog.autoReviewModel
			decodedBody, err = replaceRequestModel(decodedBody, model)
			if err != nil {
				h.logActivity(started, r.Method, suffix, model, "ollama", http.StatusBadRequest, "request_error")
				writeJSONError(w, http.StatusBadRequest, "prepare Codex Auto-review request for Ollama: "+err.Error())
				return
			}
		}
		routedModel, routed = catalog.models[modelKey(model)]
	}
	if isAutoReview && routed && suffix == "/v1/responses" {
		decodedBody, err = prepareAutoReviewRequest(decodedBody)
		if err != nil {
			h.logActivity(started, r.Method, suffix, model, "ollama", http.StatusBadRequest, "request_error")
			writeJSONError(w, http.StatusBadRequest, "prepare Codex Auto-review request for Ollama: "+err.Error())
			return
		}
	}
	if !routed && usesManagedAPIKey(r.Header) {
		h.lastRoute.Store(routeSnapshot{Model: model, Route: "none", UpstreamStatus: http.StatusUnauthorized})
		h.logActivity(started, r.Method, suffix, model, "none", http.StatusUnauthorized, "auth_required")
		writeJSONError(w, http.StatusUnauthorized, "OpenAI models require signing in to ChatGPT or adding an OpenAI API key")
		return
	}

	// The built-in OpenAI provider sends both authentication modes through
	// openai_base_url. ChatGPT-account requests carry ChatGPT-Account-ID;
	// API-key requests do not. Preserve Codex's normal auth-aware upstream.
	targetBase := h.openAIURL
	targetSuffix := strings.TrimPrefix(suffix, "/v1")
	route := "openai"
	if usesChatGPTBackend(r.Header) {
		targetBase = h.chatGPTURL
		route = "chatgpt"
	}
	var (
		requestBody           []byte
		requestBodyNormalized bool
	)
	if routed {
		targetBase = h.ollamaURL
		targetSuffix = suffix
		route = "ollama"
		requestBody, err = normalizeOllamaRequestBody(decodedBody, routedModel)
		if err != nil {
			h.logActivity(started, r.Method, suffix, model, "ollama", http.StatusBadRequest, "request_error")
			writeJSONError(w, http.StatusBadRequest, "prepare Codex request for Ollama: "+err.Error())
			return
		}
	} else {
		requestBody, requestBodyNormalized, err = normalizeNativeRequestBody(decodedBody)
		if err != nil {
			h.logActivity(started, r.Method, suffix, model, route, http.StatusBadRequest, "request_error")
			writeJSONError(w, http.StatusBadRequest, "prepare Codex request for OpenAI: "+err.Error())
			return
		}
		if !requestBodyNormalized {
			requestBody = rawBody
		}
	}
	h.lastRoute.Store(routeSnapshot{Model: model, Route: route})
	targetURL := resolveTarget(targetBase, targetSuffix, r.URL.RawQuery)

	outReq, err := http.NewRequestWithContext(r.Context(), r.Method, targetURL.String(), bytes.NewReader(requestBody))
	if err != nil {
		h.logActivity(started, r.Method, suffix, model, route, http.StatusInternalServerError, "request_error")
		writeJSONError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if routed {
		copyOllamaRequestHeaders(outReq.Header, r.Header)
	} else {
		copyHeaders(outReq.Header, r.Header)
		if requestBodyNormalized {
			outReq.Header.Del("Content-Encoding")
		}
	}

	h.logger.Debug("routing Codex request", "path", suffix, "model", model, "ollama", routed)
	resp, err := h.client.Do(outReq)
	if err != nil {
		h.upstreamErrors.Add(1)
		h.logActivity(started, r.Method, suffix, model, route, http.StatusBadGateway, "upstream_error")
		writeJSONError(w, http.StatusBadGateway, err.Error())
		return
	}
	defer resp.Body.Close()
	h.lastRoute.Store(routeSnapshot{Model: model, Route: route, UpstreamStatus: resp.StatusCode})
	if resp.StatusCode >= http.StatusInternalServerError {
		h.upstreamErrors.Add(1)
	}
	if isAcceptedModelRequest(r.Method, suffix, hasModel, resp.StatusCode) {
		if routed {
			h.ollamaRequests.Add(1)
		} else {
			h.chatGPTRequests.Add(1)
		}
	}

	result := "ok"
	if resp.StatusCode >= http.StatusInternalServerError {
		result = "upstream_error"
	}
	if isAutoReview && routed && suffix == "/v1/responses" && resp.StatusCode >= http.StatusOK && resp.StatusCode < http.StatusMultipleChoices {
		responseBody, readErr := io.ReadAll(io.LimitReader(resp.Body, h.maxBodyBytes+1))
		if readErr != nil {
			h.upstreamErrors.Add(1)
			h.logActivity(started, r.Method, suffix, model, route, http.StatusBadGateway, "stream_error")
			writeJSONError(w, http.StatusBadGateway, "read Codex Auto-review response from Ollama: "+readErr.Error())
			return
		}
		if int64(len(responseBody)) > h.maxBodyBytes {
			h.upstreamErrors.Add(1)
			h.logActivity(started, r.Method, suffix, model, route, http.StatusBadGateway, "response_error")
			writeJSONError(w, http.StatusBadGateway, fmt.Sprintf("Codex Auto-review response exceeds %d bytes", h.maxBodyBytes))
			return
		}
		responseBody, _, err = transformAutoReviewResponse(responseBody, resp.Header.Get("Content-Type"))
		if err != nil {
			h.upstreamErrors.Add(1)
			h.logActivity(started, r.Method, suffix, model, route, http.StatusBadGateway, "response_error")
			writeJSONError(w, http.StatusBadGateway, "invalid Codex Auto-review response from Ollama: "+err.Error())
			return
		}
		copyHeaders(w.Header(), resp.Header)
		w.WriteHeader(resp.StatusCode)
		if _, err := w.Write(responseBody); err != nil {
			result = "stream_error"
			h.upstreamErrors.Add(1)
			h.logger.Warn("Codex proxy response write failed", "path", suffix, "model", model, "ollama", routed, "error", err)
		}
		h.logActivity(started, r.Method, suffix, model, route, resp.StatusCode, result)
		return
	}
	copyHeaders(w.Header(), resp.Header)
	w.WriteHeader(resp.StatusCode)
	if err := copyStreaming(w, resp.Body); err != nil {
		result = "canceled"
		if !errors.Is(err, r.Context().Err()) {
			result = "stream_error"
			h.upstreamErrors.Add(1)
			h.logger.Warn("Codex proxy response stream failed", "path", suffix, "model", model, "ollama", routed, "error", err)
		}
	}
	h.logActivity(started, r.Method, suffix, model, route, resp.StatusCode, result)
}

func (h *Handler) logActivity(started time.Time, method, path, model, route string, status int, result string) {
	h.writeActivity(
		"route=%s model=%q method=%s path=%s status=%d duration=%s result=%s",
		route,
		model,
		method,
		path,
		status,
		time.Since(started).Round(time.Millisecond),
		result,
	)
}

func (h *Handler) writeActivity(format string, args ...any) {
	if h.activityLogPath == "" {
		return
	}
	h.activityLogMu.Lock()
	defer h.activityLogMu.Unlock()

	if err := os.MkdirAll(filepath.Dir(h.activityLogPath), 0o700); err != nil {
		h.logger.Warn("failed to create Codex proxy log directory", "error", err)
		return
	}
	file, err := os.OpenFile(h.activityLogPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600)
	if err != nil {
		h.logger.Warn("failed to open Codex proxy activity log", "error", err)
		return
	}
	defer file.Close()

	_, err = fmt.Fprintf(file, "%s %s\n", time.Now().Format(time.RFC3339), fmt.Sprintf(format, args...))
	if err != nil {
		h.logger.Warn("failed to write Codex proxy activity log", "error", err)
	}
}

func (h *Handler) writeStatus(w http.ResponseWriter) {
	status := statusResponse{
		OK:              true,
		OllamaRequests:  h.ollamaRequests.Load(),
		ChatGPTRequests: h.chatGPTRequests.Load(),
		UpstreamErrors:  h.upstreamErrors.Load(),
	}
	if value := h.lastRoute.Load(); value != nil {
		last := value.(routeSnapshot)
		status.LastModel = last.Model
		status.LastRoute = last.Route
		status.LastUpstreamStatus = last.UpstreamStatus
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(status)
}

func isAcceptedModelRequest(method, path string, hasModel bool, status int) bool {
	return method == http.MethodPost &&
		path == "/v1/responses" &&
		hasModel &&
		status >= http.StatusOK &&
		status < http.StatusMultipleChoices
}

func (h *Handler) readBodies(r *http.Request) ([]byte, []byte, error) {
	if r.Body == nil {
		return nil, nil, nil
	}
	raw, err := io.ReadAll(io.LimitReader(r.Body, h.maxBodyBytes+1))
	if err != nil {
		return nil, nil, fmt.Errorf("read request body: %w", err)
	}
	if int64(len(raw)) > h.maxBodyBytes {
		return nil, nil, fmt.Errorf("request body exceeds %d bytes", h.maxBodyBytes)
	}
	if !strings.EqualFold(strings.TrimSpace(r.Header.Get("Content-Encoding")), "zstd") {
		return raw, raw, nil
	}

	decoder, err := zstd.NewReader(bytes.NewReader(raw), zstd.WithDecoderMaxMemory(uint64(h.maxBodyBytes)))
	if err != nil {
		return nil, nil, fmt.Errorf("decompress zstd request body: %w", err)
	}
	defer decoder.Close()
	decoded, err := io.ReadAll(io.LimitReader(decoder, h.maxBodyBytes+1))
	if err != nil {
		return nil, nil, fmt.Errorf("decompress zstd request body: %w", err)
	}
	if int64(len(decoded)) > h.maxBodyBytes {
		return nil, nil, fmt.Errorf("decompressed request body exceeds %d bytes", h.maxBodyBytes)
	}
	return raw, decoded, nil
}

type routingThinkingMetadata struct {
	Supported bool                       `json:"supported"`
	Levels    []string                   `json:"levels,omitempty"`
	Values    map[string]json.RawMessage `json:"values,omitempty"`
}

type routingModel struct {
	Slug     string                   `json:"slug"`
	Thinking *routingThinkingMetadata `json:"thinking,omitempty"`
}

type routingCatalog struct {
	models          map[string]routingModel
	autoReviewModel string
}

func loadRoutingCatalog(path string) (routingCatalog, error) {
	if strings.TrimSpace(path) == "" {
		return routingCatalog{}, fmt.Errorf("model catalog path is empty")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return routingCatalog{}, err
	}
	var catalog struct {
		Models          []routingModel `json:"models"`
		AutoReviewModel string         `json:"auto_review_model"`
	}
	if err := json.Unmarshal(data, &catalog); err != nil {
		return routingCatalog{}, err
	}
	models := make(map[string]routingModel, len(catalog.Models))
	for _, model := range catalog.Models {
		if key := modelKey(model.Slug); key != "" {
			models[key] = model
		}
	}
	autoReviewModel := strings.TrimSpace(catalog.AutoReviewModel)
	if autoReviewModel != "" {
		if _, ok := models[modelKey(autoReviewModel)]; !ok {
			return routingCatalog{}, fmt.Errorf("Auto-review model %q is not in the Ollama routing catalog", autoReviewModel)
		}
	}
	return routingCatalog{models: models, autoReviewModel: autoReviewModel}, nil
}

func extractModel(body []byte) (string, bool) {
	if len(body) == 0 {
		return "", false
	}
	var payload struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		return "", false
	}
	payload.Model = strings.TrimSpace(payload.Model)
	return payload.Model, payload.Model != ""
}

func replaceRequestModel(body []byte, model string) ([]byte, error) {
	var payload map[string]json.RawMessage
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, err
	}
	encodedModel, err := json.Marshal(model)
	if err != nil {
		return nil, err
	}
	payload["model"] = encodedModel
	return json.Marshal(payload)
}

// prepareAutoReviewRequest gives Ollama a typed terminal action for the
// Guardian decision. Codex does not know about this proxy-owned tool, so the
// response path consumes it and returns the JSON text Codex already expects.
func prepareAutoReviewRequest(body []byte) ([]byte, error) {
	var payload map[string]json.RawMessage
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, err
	}

	var tools []json.RawMessage
	if raw := bytes.TrimSpace(payload["tools"]); len(raw) > 0 && !bytes.Equal(raw, []byte("null")) {
		if err := json.Unmarshal(raw, &tools); err != nil {
			return nil, fmt.Errorf("decode tools: %w", err)
		}
	}
	for _, raw := range tools {
		var tool struct {
			Name string `json:"name"`
		}
		if err := json.Unmarshal(raw, &tool); err != nil {
			return nil, fmt.Errorf("decode tool: %w", err)
		}
		if tool.Name == guardianDecisionToolName {
			return nil, fmt.Errorf("tool name %q is reserved by the Codex proxy", guardianDecisionToolName)
		}
	}
	decisionTool, err := json.Marshal(guardianDecisionTool)
	if err != nil {
		return nil, fmt.Errorf("encode Guardian decision tool: %w", err)
	}
	tools = append(tools, decisionTool)
	encodedTools, err := json.Marshal(tools)
	if err != nil {
		return nil, fmt.Errorf("encode tools: %w", err)
	}
	payload["tools"] = encodedTools

	var input []json.RawMessage
	if err := json.Unmarshal(payload["input"], &input); err != nil {
		return nil, fmt.Errorf("decode input: %w", err)
	}
	foundUserMessage := false
	for i := len(input) - 1; i >= 0; i-- {
		var message struct {
			Type    string          `json:"type"`
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		}
		if err := json.Unmarshal(input[i], &message); err != nil {
			return nil, fmt.Errorf("decode input item: %w", err)
		}
		if message.Role != "user" || (message.Type != "" && message.Type != "message") {
			continue
		}

		content, changed, err := appendAutoReviewInstructionToContent(message.Content)
		if err != nil {
			return nil, err
		}
		if !changed {
			continue
		}
		foundUserMessage = true

		var item map[string]json.RawMessage
		if err := json.Unmarshal(input[i], &item); err != nil {
			return nil, fmt.Errorf("decode user message: %w", err)
		}
		item["content"] = content
		input[i], err = json.Marshal(item)
		if err != nil {
			return nil, fmt.Errorf("encode user message: %w", err)
		}
		break
	}
	if !foundUserMessage {
		return nil, errors.New("Guardian request has no user message")
	}
	encodedInput, err := json.Marshal(input)
	if err != nil {
		return nil, fmt.Errorf("encode input: %w", err)
	}
	payload["input"] = encodedInput
	encoded, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("encode request: %w", err)
	}
	return encoded, nil
}

func appendAutoReviewInstructionToContent(content json.RawMessage) (json.RawMessage, bool, error) {
	trimmed := bytes.TrimSpace(content)
	if len(trimmed) == 0 {
		return content, false, nil
	}
	if trimmed[0] == '"' {
		var text string
		if err := json.Unmarshal(trimmed, &text); err != nil {
			return nil, false, fmt.Errorf("decode user message content: %w", err)
		}
		encoded, err := json.Marshal(text + guardianDecisionInstruction)
		if err != nil {
			return nil, false, fmt.Errorf("encode user message content: %w", err)
		}
		return encoded, true, nil
	}
	if trimmed[0] != '[' {
		return content, false, nil
	}

	var parts []json.RawMessage
	if err := json.Unmarshal(trimmed, &parts); err != nil {
		return nil, false, fmt.Errorf("decode user message content: %w", err)
	}
	instruction, err := json.Marshal(map[string]string{
		"type": "input_text",
		"text": guardianDecisionInstruction,
	})
	if err != nil {
		return nil, false, err
	}
	parts = append(parts, instruction)
	encoded, err := json.Marshal(parts)
	if err != nil {
		return nil, false, fmt.Errorf("encode user message content: %w", err)
	}
	return encoded, true, nil
}

type guardianDecision struct {
	RiskLevel         string `json:"risk_level"`
	UserAuthorization string `json:"user_authorization"`
	Outcome           string `json:"outcome"`
	Rationale         string `json:"rationale"`
}

type autoReviewOutputItem struct {
	ID        string `json:"id"`
	Type      string `json:"type"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

func transformAutoReviewResponse(body []byte, contentType string) ([]byte, bool, error) {
	if strings.HasPrefix(strings.ToLower(strings.TrimSpace(contentType)), "text/event-stream") {
		return transformAutoReviewEventStream(body)
	}
	return transformAutoReviewJSON(body)
}

func transformAutoReviewJSON(body []byte) ([]byte, bool, error) {
	var response map[string]json.RawMessage
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, false, fmt.Errorf("decode response: %w", err)
	}
	var status string
	if err := json.Unmarshal(response["status"], &status); err != nil {
		return nil, false, fmt.Errorf("decode response status: %w", err)
	}
	if status != "completed" {
		return body, false, nil
	}
	var output []json.RawMessage
	if err := json.Unmarshal(response["output"], &output); err != nil {
		return nil, false, fmt.Errorf("decode response output: %w", err)
	}
	inspection, err := inspectAutoReviewOutput(output)
	if err != nil || inspection.passthrough {
		return body, false, err
	}
	transformedOutput := make([]json.RawMessage, 0, len(output)-len(inspection.discardedMessageIDs))
	for i, raw := range output {
		var item autoReviewOutputItem
		if err := json.Unmarshal(raw, &item); err != nil {
			return nil, false, fmt.Errorf("decode response output item: %w", err)
		}
		if _, discard := inspection.discardedMessageIDs[item.ID]; discard && item.Type == "message" {
			continue
		}
		if item.Type == "function_call" && item.Name == guardianDecisionToolName {
			output[i], err = json.Marshal(autoReviewDecisionMessage(inspection.itemID, inspection.decisionJSON))
			if err != nil {
				return nil, false, fmt.Errorf("encode Guardian decision message: %w", err)
			}
		}
		transformedOutput = append(transformedOutput, output[i])
	}
	response["output"], err = json.Marshal(transformedOutput)
	if err != nil {
		return nil, false, fmt.Errorf("encode response output: %w", err)
	}
	transformed, err := json.Marshal(response)
	if err != nil {
		return nil, false, fmt.Errorf("encode response: %w", err)
	}
	return transformed, true, nil
}

type autoReviewInspection struct {
	decisionJSON        string
	itemID              string
	discardedMessageIDs map[string]struct{}
	passthrough         bool
}

func inspectAutoReviewOutput(output []json.RawMessage) (autoReviewInspection, error) {
	result := autoReviewInspection{discardedMessageIDs: make(map[string]struct{})}
	decisionCalls := 0
	otherTerminalOutput := false
	for _, raw := range output {
		var item autoReviewOutputItem
		if err := json.Unmarshal(raw, &item); err != nil {
			return result, fmt.Errorf("decode response output item: %w", err)
		}
		switch item.Type {
		case "reasoning":
			continue
		case "message":
			// Some otherwise reliable tool-capable models emit explanatory prose
			// alongside the decision call. The validated call is authoritative;
			// record the prose item so both JSON and SSE paths can discard it.
			if item.ID == "" {
				return result, errors.New("Guardian assistant message has no item ID")
			}
			result.discardedMessageIDs[item.ID] = struct{}{}
		case "function_call":
			if item.Name != guardianDecisionToolName {
				if decisionCalls > 0 {
					otherTerminalOutput = true
				} else {
					result.passthrough = true
				}
				continue
			}
			decisionCalls++
			if decisionCalls > 1 {
				return result, errors.New("Guardian called the decision tool more than once")
			}
			decisionJSON, err := validateGuardianDecision(item.Arguments)
			if err != nil {
				return result, err
			}
			result.decisionJSON = decisionJSON
			result.itemID = item.ID
		default:
			otherTerminalOutput = true
		}
	}
	if result.passthrough && decisionCalls == 0 {
		return result, nil
	}
	if decisionCalls == 0 {
		return result, errors.New("Guardian did not call submit_guardian_decision")
	}
	if result.passthrough || otherTerminalOutput {
		return result, errors.New("Guardian mixed its decision with other terminal output")
	}
	return result, nil
}

func validateGuardianDecision(arguments string) (string, error) {
	decoder := json.NewDecoder(strings.NewReader(arguments))
	decoder.DisallowUnknownFields()
	var decision guardianDecision
	if err := decoder.Decode(&decision); err != nil {
		return "", fmt.Errorf("decode Guardian decision arguments: %w", err)
	}
	if decoder.Decode(&struct{}{}) != io.EOF {
		return "", errors.New("Guardian decision arguments contain trailing data")
	}
	if !oneOf(decision.RiskLevel, "low", "medium", "high", "critical") {
		return "", fmt.Errorf("invalid Guardian risk_level %q", decision.RiskLevel)
	}
	if !oneOf(decision.UserAuthorization, "unknown", "low", "medium", "high") {
		return "", fmt.Errorf("invalid Guardian user_authorization %q", decision.UserAuthorization)
	}
	if !oneOf(decision.Outcome, "allow", "deny") {
		return "", fmt.Errorf("invalid Guardian outcome %q", decision.Outcome)
	}
	if strings.TrimSpace(decision.Rationale) == "" {
		return "", errors.New("Guardian rationale is empty")
	}
	encoded, err := json.Marshal(decision)
	if err != nil {
		return "", fmt.Errorf("encode Guardian decision: %w", err)
	}
	return string(encoded), nil
}

func oneOf(value string, allowed ...string) bool {
	for _, candidate := range allowed {
		if value == candidate {
			return true
		}
	}
	return false
}

func autoReviewDecisionMessage(itemID, decision string) map[string]any {
	return map[string]any{
		"id":     itemID,
		"type":   "message",
		"status": "completed",
		"role":   "assistant",
		"content": []any{map[string]any{
			"type":        "output_text",
			"text":        decision,
			"annotations": []any{},
			"logprobs":    []any{},
		}},
	}
}

type serverSentEvent struct {
	event string
	data  []byte
}

func transformAutoReviewEventStream(body []byte) ([]byte, bool, error) {
	events, err := parseServerSentEvents(body)
	if err != nil {
		return nil, false, err
	}
	var completedOutput []json.RawMessage
	terminalFailure := false
	for _, event := range events {
		if event.event == "response.failed" || event.event == "response.incomplete" {
			terminalFailure = true
		}
		if event.event != "response.completed" {
			continue
		}
		var payload struct {
			Response struct {
				Output []json.RawMessage `json:"output"`
			} `json:"response"`
		}
		if err := json.Unmarshal(event.data, &payload); err != nil {
			return nil, false, fmt.Errorf("decode response.completed event: %w", err)
		}
		completedOutput = payload.Response.Output
	}
	if completedOutput == nil {
		if terminalFailure {
			return body, false, nil
		}
		return nil, false, errors.New("Guardian stream has no response.completed event")
	}
	inspection, err := inspectAutoReviewOutput(completedOutput)
	if err != nil || inspection.passthrough {
		return body, false, err
	}
	outputIndexes := make(map[int]int, len(completedOutput)-len(inspection.discardedMessageIDs))
	nextOutputIndex := 0
	for oldOutputIndex, raw := range completedOutput {
		var item autoReviewOutputItem
		if err := json.Unmarshal(raw, &item); err != nil {
			return nil, false, fmt.Errorf("decode completed output item: %w", err)
		}
		if _, discard := inspection.discardedMessageIDs[item.ID]; discard && item.Type == "message" {
			continue
		}
		outputIndexes[oldOutputIndex] = nextOutputIndex
		nextOutputIndex++
	}

	var transformed []serverSentEvent
	addedDecisionMessage := false
	finishedDecisionMessage := false
	for _, event := range events {
		if bytes.Equal(bytes.TrimSpace(event.data), []byte("[DONE]")) {
			transformed = append(transformed, event)
			continue
		}
		var payload map[string]any
		if err := json.Unmarshal(event.data, &payload); err != nil {
			return nil, false, fmt.Errorf("decode %s event: %w", event.event, err)
		}
		if _, discard := inspection.discardedMessageIDs[autoReviewEventItemID(payload)]; discard {
			continue
		}
		if outputIndex, ok := payload["output_index"].(float64); ok {
			if mapped, keep := outputIndexes[int(outputIndex)]; keep {
				payload["output_index"] = mapped
			}
		}
		switch event.event {
		case "response.output_item.added":
			item, _ := payload["item"].(map[string]any)
			if itemString(item, "id") == inspection.itemID && itemString(item, "name") == guardianDecisionToolName {
				outputIndex := payload["output_index"]
				transformed = append(transformed,
					newServerSentEvent("response.output_item.added", map[string]any{
						"output_index": outputIndex,
						"item": map[string]any{
							"id": inspection.itemID, "type": "message", "status": "in_progress", "role": "assistant", "content": []any{},
						},
					}),
					newServerSentEvent("response.content_part.added", map[string]any{
						"item_id": inspection.itemID, "output_index": outputIndex, "content_index": 0,
						"part": map[string]any{"type": "output_text", "text": "", "annotations": []any{}, "logprobs": []any{}},
					}),
				)
				addedDecisionMessage = true
				continue
			}
		case "response.function_call_arguments.delta", "response.function_call_arguments.done":
			if itemString(payload, "item_id") == inspection.itemID {
				continue
			}
		case "response.output_item.done":
			item, _ := payload["item"].(map[string]any)
			if itemString(item, "id") == inspection.itemID && itemString(item, "name") == guardianDecisionToolName {
				if !addedDecisionMessage {
					return nil, false, errors.New("Guardian decision stream has no output_item.added event")
				}
				outputIndex := payload["output_index"]
				message := autoReviewDecisionMessage(inspection.itemID, inspection.decisionJSON)
				content := message["content"].([]any)[0]
				transformed = append(transformed,
					newServerSentEvent("response.output_text.delta", map[string]any{
						"item_id": inspection.itemID, "output_index": outputIndex, "content_index": 0, "delta": inspection.decisionJSON, "logprobs": []any{},
					}),
					newServerSentEvent("response.output_text.done", map[string]any{
						"item_id": inspection.itemID, "output_index": outputIndex, "content_index": 0, "text": inspection.decisionJSON, "logprobs": []any{},
					}),
					newServerSentEvent("response.content_part.done", map[string]any{
						"item_id": inspection.itemID, "output_index": outputIndex, "content_index": 0, "part": content,
					}),
					newServerSentEvent("response.output_item.done", map[string]any{
						"output_index": outputIndex, "item": message,
					}),
				)
				finishedDecisionMessage = true
				continue
			}
		case "response.completed":
			if !finishedDecisionMessage {
				return nil, false, errors.New("Guardian decision stream has no output_item.done event")
			}
			response, ok := payload["response"].(map[string]any)
			if !ok {
				return nil, false, errors.New("response.completed event has no response object")
			}
			output, ok := response["output"].([]any)
			if !ok {
				return nil, false, errors.New("response.completed event has no output array")
			}
			transformedOutput := make([]any, 0, len(output)-len(inspection.discardedMessageIDs))
			for _, value := range output {
				item, _ := value.(map[string]any)
				if _, discard := inspection.discardedMessageIDs[itemString(item, "id")]; discard && itemString(item, "type") == "message" {
					continue
				}
				if itemString(item, "id") == inspection.itemID && itemString(item, "name") == guardianDecisionToolName {
					value = autoReviewDecisionMessage(inspection.itemID, inspection.decisionJSON)
				}
				transformedOutput = append(transformedOutput, value)
			}
			response["output"] = transformedOutput
		}
		encoded, err := json.Marshal(payload)
		if err != nil {
			return nil, false, fmt.Errorf("encode %s event: %w", event.event, err)
		}
		transformed = append(transformed, serverSentEvent{event: event.event, data: encoded})
	}
	encoded, err := encodeServerSentEvents(transformed)
	if err != nil {
		return nil, false, err
	}
	return encoded, true, nil
}

func autoReviewEventItemID(payload map[string]any) string {
	if itemID := itemString(payload, "item_id"); itemID != "" {
		return itemID
	}
	item, _ := payload["item"].(map[string]any)
	return itemString(item, "id")
}

func itemString(item map[string]any, key string) string {
	value, _ := item[key].(string)
	return value
}

func newServerSentEvent(event string, payload map[string]any) serverSentEvent {
	payload["type"] = event
	encoded, _ := json.Marshal(payload)
	return serverSentEvent{event: event, data: encoded}
}

func parseServerSentEvents(body []byte) ([]serverSentEvent, error) {
	normalized := bytes.ReplaceAll(body, []byte("\r\n"), []byte("\n"))
	var events []serverSentEvent
	for _, frame := range bytes.Split(normalized, []byte("\n\n")) {
		if len(bytes.TrimSpace(frame)) == 0 {
			continue
		}
		var event serverSentEvent
		var dataLines [][]byte
		for _, line := range bytes.Split(frame, []byte("\n")) {
			switch {
			case bytes.HasPrefix(line, []byte("event:")):
				event.event = strings.TrimSpace(string(bytes.TrimPrefix(line, []byte("event:"))))
			case bytes.HasPrefix(line, []byte("data:")):
				dataLines = append(dataLines, bytes.TrimSpace(bytes.TrimPrefix(line, []byte("data:"))))
			}
		}
		if event.event == "" || len(dataLines) == 0 {
			return nil, errors.New("malformed Guardian event stream")
		}
		event.data = bytes.Join(dataLines, []byte("\n"))
		events = append(events, event)
	}
	return events, nil
}

func encodeServerSentEvents(events []serverSentEvent) ([]byte, error) {
	var result bytes.Buffer
	sequenceNumber := 0
	for _, event := range events {
		data := event.data
		if !bytes.Equal(bytes.TrimSpace(data), []byte("[DONE]")) {
			var payload map[string]any
			if err := json.Unmarshal(data, &payload); err != nil {
				return nil, fmt.Errorf("decode transformed %s event: %w", event.event, err)
			}
			payload["type"] = event.event
			payload["sequence_number"] = sequenceNumber
			sequenceNumber++
			var err error
			data, err = json.Marshal(payload)
			if err != nil {
				return nil, fmt.Errorf("encode transformed %s event: %w", event.event, err)
			}
		}
		fmt.Fprintf(&result, "event: %s\ndata: %s\n\n", event.event, data)
	}
	return result.Bytes(), nil
}

// normalizeOllamaRequestBody translates Codex-specific Responses history into
// the subset accepted by Ollama's OpenAI-compatible Responses endpoint.
// Provider-specific native reasoning is omitted while Ollama reasoning stays
// available to its own multi-step tool loop.
func normalizeOllamaRequestBody(body []byte, model routingModel) ([]byte, error) {
	normalized, _, err := normalizeRequestInput(body, normalizeOllamaInputItem)
	if err != nil || model.Thinking == nil {
		return normalized, err
	}
	return normalizeOllamaThinking(normalized, *model.Thinking)
}

func normalizeOllamaThinking(body []byte, metadata routingThinkingMetadata) ([]byte, error) {
	var payload map[string]json.RawMessage
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, err
	}
	_, hadThinkOverride := payload["think"]
	delete(payload, "think")
	if !metadata.Supported {
		if _, ok := payload["reasoning"]; !ok && !hadThinkOverride {
			return body, nil
		}
		delete(payload, "reasoning")
		return json.Marshal(payload)
	}

	reasoningData, ok := payload["reasoning"]
	if !ok || len(metadata.Levels) == 0 {
		if hadThinkOverride {
			return json.Marshal(payload)
		}
		return body, nil
	}
	var reasoning map[string]json.RawMessage
	if err := json.Unmarshal(reasoningData, &reasoning); err != nil {
		return nil, fmt.Errorf("decode reasoning: %w", err)
	}
	var effort string
	if rawEffort, ok := reasoning["effort"]; ok {
		if err := json.Unmarshal(rawEffort, &effort); err != nil {
			return nil, fmt.Errorf("decode reasoning effort: %w", err)
		}
	}
	if effort == "" {
		if hadThinkOverride {
			return json.Marshal(payload)
		}
		return body, nil
	}

	normalizedEffort := normalizeThinkingEffort(effort, metadata.Levels)
	if normalizedEffort == "" {
		// A stale selection from another model should not make this request fail.
		// Omitting effort lets Ollama apply the selected model's own behavior.
		delete(reasoning, "effort")
	} else {
		encodedEffort, err := json.Marshal(normalizedEffort)
		if err != nil {
			return nil, fmt.Errorf("encode reasoning effort: %w", err)
		}
		reasoning["effort"] = encodedEffort
		if rawValue, ok := metadata.Values[normalizedEffort]; ok {
			// Codex exposes named reasoning efforts, while Ollama models may use
			// boolean thinking controls. Preserve the server-advertised raw value
			// in an Ollama-only extension consumed by the Responses adapter.
			payload["think"] = rawValue
		}
	}
	encodedReasoning, err := json.Marshal(reasoning)
	if err != nil {
		return nil, fmt.Errorf("encode reasoning: %w", err)
	}
	payload["reasoning"] = encodedReasoning
	return json.Marshal(payload)
}

func normalizeThinkingEffort(effort string, levels []string) string {
	var normalized string
	switch effort {
	case "minimal":
		normalized = "low"
	case "xhigh", "ultra":
		normalized = "max"
	case "none", "low", "medium", "high", "max":
		normalized = effort
	default:
		return ""
	}

	if slices.Equal(levels, []string{"none", "medium"}) && normalized != "none" {
		// Ollama represents the enabled side of a binary thinking control as
		// medium even when the underlying model has no adjustable effort ladder.
		return "medium"
	}
	if slices.Contains(levels, normalized) {
		return normalized
	}
	if normalized == "max" && slices.Contains(levels, "high") {
		// A stale xhigh or ultra choice should use the strongest supported level.
		return "high"
	}
	return ""
}

// normalizeNativeRequestBody removes Ollama reasoning items before a native
// request reaches OpenAI. Ollama's Responses adapter currently serializes
// plaintext thinking as encrypted_content, which OpenAI correctly rejects as
// invalid ciphertext. Visible messages and tool history remain in the input.
func normalizeNativeRequestBody(body []byte) ([]byte, bool, error) {
	return normalizeRequestInput(body, normalizeChatGPTInputItem)
}

func normalizeRequestInput(
	body []byte,
	normalizeItem func(json.RawMessage) (json.RawMessage, bool, error),
) ([]byte, bool, error) {
	if len(body) == 0 {
		return body, false, nil
	}

	var payload map[string]json.RawMessage
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, false, err
	}
	input, ok := payload["input"]
	if !ok || len(input) == 0 || input[0] != '[' {
		return body, false, nil
	}

	var items []json.RawMessage
	if err := json.Unmarshal(input, &items); err != nil {
		return nil, false, fmt.Errorf("decode input: %w", err)
	}

	normalized := make([]json.RawMessage, 0, len(items))
	changed := false
	for _, item := range items {
		converted, keep, err := normalizeItem(item)
		if err != nil {
			return nil, false, err
		}
		if keep {
			normalized = append(normalized, converted)
		}
		if !keep || !bytes.Equal(item, converted) {
			changed = true
		}
	}
	if !changed {
		return body, false, nil
	}

	encodedInput, err := json.Marshal(normalized)
	if err != nil {
		return nil, false, fmt.Errorf("encode input: %w", err)
	}
	payload["input"] = encodedInput
	encoded, err := json.Marshal(payload)
	if err != nil {
		return nil, false, fmt.Errorf("encode request: %w", err)
	}
	return encoded, true, nil
}

func normalizeOllamaInputItem(item json.RawMessage) (json.RawMessage, bool, error) {
	var header struct {
		Type string `json:"type"`
		Role string `json:"role"`
	}
	if err := json.Unmarshal(item, &header); err != nil {
		return nil, false, fmt.Errorf("decode input item: %w", err)
	}

	// Ollama accepts message shorthand without an explicit type as well as
	// the supported Responses item types below.
	itemType := header.Type
	if itemType == "" && header.Role != "" {
		itemType = "message"
	}
	switch itemType {
	case "message":
		if header.Role != "developer" {
			return item, true, nil
		}
		// Codex carries collaboration-mode and safety instructions in developer
		// messages. Ollama models and provider adapters do not consistently give
		// that role instruction priority, while system is universally supported.
		// Translate the role only on the Ollama route so Plan mode and the rest of
		// Codex's instruction contract receive the highest broadly supported
		// priority.
		var message map[string]json.RawMessage
		if err := json.Unmarshal(item, &message); err != nil {
			return nil, false, fmt.Errorf("decode developer message: %w", err)
		}
		message["role"] = json.RawMessage(`"system"`)
		converted, err := json.Marshal(message)
		if err != nil {
			return nil, false, fmt.Errorf("encode system message: %w", err)
		}
		return converted, true, nil
	case "function_call", "function_call_output":
		return item, true, nil
	case "tool_search_call", "tool_search_output", "compaction_trigger":
		// These client-executed control items are handled by Ollama's Responses
		// adapter. Preserve them so tool discovery and compaction work through the
		// ChatGPT loopback router, not only when calling /v1/responses directly.
		return item, true, nil
	case "compaction":
		// Native OpenAI compaction state is opaque and cannot be consumed by
		// Ollama. Ollama compaction state is a versioned JSON payload and must
		// reach the server middleware so it can be expanded before inference.
		return item, isOllamaCompactionItem(item), nil
	case "reasoning":
		var reasoning struct {
			ID string `json:"id"`
		}
		if err := json.Unmarshal(item, &reasoning); err != nil {
			return nil, false, fmt.Errorf("decode reasoning item: %w", err)
		}
		// Native encrypted reasoning is provider-specific opaque state. Do not
		// send it to Ollama, while retaining Ollama's own reasoning during its
		// multi-step tool loop.
		return item, isOllamaReasoningItemID(reasoning.ID), nil
	case "custom_tool_call":
		var call struct {
			ID     string `json:"id,omitempty"`
			CallID string `json:"call_id"`
			Name   string `json:"name"`
			Input  string `json:"input"`
		}
		if err := json.Unmarshal(item, &call); err != nil {
			return nil, false, fmt.Errorf("decode custom tool call: %w", err)
		}
		arguments, err := json.Marshal(map[string]string{"input": call.Input})
		if err != nil {
			return nil, false, fmt.Errorf("encode custom tool input: %w", err)
		}
		converted, err := json.Marshal(map[string]any{
			"id":        call.ID,
			"type":      "function_call",
			"call_id":   call.CallID,
			"name":      call.Name,
			"arguments": string(arguments),
		})
		return converted, true, err
	case "custom_tool_call_output":
		var output struct {
			CallID string          `json:"call_id"`
			Output json.RawMessage `json:"output"`
		}
		if err := json.Unmarshal(item, &output); err != nil {
			return nil, false, fmt.Errorf("decode custom tool output: %w", err)
		}
		converted, err := json.Marshal(map[string]any{
			"type":    "function_call_output",
			"call_id": output.CallID,
			"output":  output.Output,
		})
		return converted, true, err
	default:
		// Compaction data is encrypted for the native OpenAI backend, and other
		// Codex-only item types have no Ollama Responses equivalent. Omitting
		// them is preferable to rejecting the entire otherwise usable history.
		return nil, false, nil
	}
}

func isOllamaCompactionItem(item json.RawMessage) bool {
	var wire struct {
		EncryptedContent string `json:"encrypted_content"`
	}
	if json.Unmarshal(item, &wire) != nil || wire.EncryptedContent == "" {
		return false
	}
	var payload struct {
		Type string `json:"type"`
	}
	return json.Unmarshal([]byte(wire.EncryptedContent), &payload) == nil &&
		payload.Type == "ollama_compaction"
}

func normalizeChatGPTInputItem(item json.RawMessage) (json.RawMessage, bool, error) {
	var header struct {
		ID   string `json:"id"`
		Type string `json:"type"`
	}
	if err := json.Unmarshal(item, &header); err != nil {
		return nil, false, fmt.Errorf("decode input item: %w", err)
	}
	if header.Type == "reasoning" && isOllamaReasoningItemID(header.ID) {
		return nil, false, nil
	}
	if header.Type == "compaction" && isOllamaCompactionItem(item) {
		// Ollama's compaction payload is local plaintext state stored in the
		// Responses encrypted_content field. OpenAI cannot decrypt it, so omit it
		// on a provider switch just as native compaction state is omitted on the
		// Ollama route.
		return nil, false, nil
	}
	return item, true, nil
}

func isOllamaReasoningItemID(id string) bool {
	suffix, ok := strings.CutPrefix(strings.TrimSpace(id), "rs_")
	if !ok {
		return false
	}
	if responseSuffix, ok := strings.CutPrefix(suffix, "resp_"); ok {
		suffix = responseSuffix
	}
	if suffix == "" || len(suffix) > 6 {
		return false
	}
	for _, char := range suffix {
		if char < '0' || char > '9' {
			return false
		}
	}
	return true
}

func modelKey(model string) string {
	return strings.TrimSuffix(strings.TrimSpace(model), ":latest")
}

func resolveTarget(base *url.URL, suffix, rawQuery string) *url.URL {
	target := *base
	target.Path = strings.TrimRight(base.Path, "/") + "/" + strings.TrimLeft(suffix, "/")
	target.RawPath = ""
	target.RawQuery = rawQuery
	target.Fragment = ""
	return &target
}

func isLoopbackRequest(r *http.Request) bool {
	host, _, err := net.SplitHostPort(strings.TrimSpace(r.RemoteAddr))
	if err != nil {
		host = strings.TrimSpace(r.RemoteAddr)
	}
	if strings.EqualFold(host, "localhost") {
		return true
	}
	ip := net.ParseIP(strings.Trim(host, "[]"))
	return ip != nil && ip.IsLoopback()
}

func isWebSocketUpgrade(r *http.Request) bool {
	if !strings.EqualFold(strings.TrimSpace(r.Header.Get("Upgrade")), "websocket") {
		return false
	}
	_, ok := connectionHeaderTokens(r.Header)["upgrade"]
	return ok
}

func usesChatGPTBackend(header http.Header) bool {
	return strings.TrimSpace(header.Get("ChatGPT-Account-ID")) != ""
}

func usesManagedAPIKey(header http.Header) bool {
	parts := strings.Fields(header.Get("Authorization"))
	return len(parts) == 2 && strings.EqualFold(parts[0], "Bearer") && parts[1] == ManagedAPIKey
}

func copyOllamaRequestHeaders(dst, src http.Header) {
	for _, key := range []string{"Accept", "Content-Type", "OpenAI-Beta", "User-Agent"} {
		for _, value := range src.Values(key) {
			dst.Add(key, value)
		}
	}
}

func copyHeaders(dst, src http.Header) {
	connectionTokens := connectionHeaderTokens(src)
	for key, values := range src {
		if isHopByHopHeader(key) || isConnectionTokenHeader(key, connectionTokens) {
			continue
		}
		dst.Del(key)
		for _, value := range values {
			dst.Add(key, value)
		}
	}
}

func connectionHeaderTokens(header http.Header) map[string]struct{} {
	tokens := map[string]struct{}{}
	for _, raw := range header.Values("Connection") {
		for _, token := range strings.Split(raw, ",") {
			if token = strings.TrimSpace(strings.ToLower(token)); token != "" {
				tokens[token] = struct{}{}
			}
		}
	}
	return tokens
}

func isHopByHopHeader(name string) bool {
	_, ok := hopByHopHeaders[strings.ToLower(name)]
	return ok
}

func isConnectionTokenHeader(name string, tokens map[string]struct{}) bool {
	_, ok := tokens[strings.ToLower(name)]
	return ok
}

func copyStreaming(dst http.ResponseWriter, src io.Reader) error {
	flusher, canFlush := dst.(http.Flusher)
	buffer := make([]byte, 32*1024)
	for {
		n, err := src.Read(buffer)
		if n > 0 {
			if _, writeErr := dst.Write(buffer[:n]); writeErr != nil {
				return writeErr
			}
			if canFlush {
				flusher.Flush()
			}
		}
		if err != nil {
			if errors.Is(err, io.EOF) {
				return nil
			}
			return err
		}
	}
}

func writeJSONError(w http.ResponseWriter, status int, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]string{"error": message})
}
