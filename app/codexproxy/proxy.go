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
	// models continue to pass through to ChatGPT.
	RoutingCatalogFilename = "ollama-launch-codex-routing.json"

	defaultMaxBodyBytes = int64(64 << 20)
)

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

// Config describes the two upstreams and Ollama-only routing catalog.
type Config struct {
	PathPrefix         string
	OllamaURL          string
	ChatGPTURL         string
	RoutingCatalogPath string
	ActivityLogPath    string
	MaxBodyBytes       int64
	Logger             *slog.Logger
	Transport          http.RoundTripper
}

// Handler routes catalog-listed model slugs to Ollama and passes every other
// request through to the authenticated ChatGPT Codex backend.
type Handler struct {
	pathPrefix         string
	ollamaURL          *url.URL
	chatGPTURL         *url.URL
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
	routed := false
	if hasModel {
		models, err := loadCatalogModels(h.routingCatalogPath)
		if err != nil {
			h.logActivity(started, r.Method, suffix, model, "none", http.StatusServiceUnavailable, "catalog_error")
			writeJSONError(w, http.StatusServiceUnavailable, "read Codex Ollama model catalog: "+err.Error())
			return
		}
		_, routed = models[modelKey(model)]
	}

	targetBase := h.chatGPTURL
	targetSuffix := strings.TrimPrefix(suffix, "/v1")
	requestBody := rawBody
	requestBodyNormalized := false
	if routed {
		targetBase = h.ollamaURL
		targetSuffix = suffix
		requestBody, err = normalizeOllamaRequestBody(decodedBody)
		if err != nil {
			h.logActivity(started, r.Method, suffix, model, "ollama", http.StatusBadRequest, "request_error")
			writeJSONError(w, http.StatusBadRequest, "prepare Codex request for Ollama: "+err.Error())
			return
		}
	} else {
		requestBody, requestBodyNormalized, err = normalizeChatGPTRequestBody(decodedBody)
		if err != nil {
			h.logActivity(started, r.Method, suffix, model, "chatgpt", http.StatusBadRequest, "request_error")
			writeJSONError(w, http.StatusBadRequest, "prepare Codex request for ChatGPT: "+err.Error())
			return
		}
		if !requestBodyNormalized {
			requestBody = rawBody
		}
	}
	route := "chatgpt"
	if routed {
		route = "ollama"
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

func loadCatalogModels(path string) (map[string]struct{}, error) {
	if strings.TrimSpace(path) == "" {
		return nil, fmt.Errorf("model catalog path is empty")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var catalog struct {
		Models []struct {
			Slug string `json:"slug"`
		} `json:"models"`
	}
	if err := json.Unmarshal(data, &catalog); err != nil {
		return nil, err
	}
	models := make(map[string]struct{}, len(catalog.Models))
	for _, model := range catalog.Models {
		if key := modelKey(model.Slug); key != "" {
			models[key] = struct{}{}
		}
	}
	return models, nil
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

// normalizeOllamaRequestBody translates Codex-specific Responses history into
// the subset accepted by Ollama's OpenAI-compatible Responses endpoint.
// Provider-specific native reasoning is omitted while Ollama reasoning stays
// available to its own multi-step tool loop.
func normalizeOllamaRequestBody(body []byte) ([]byte, error) {
	normalized, _, err := normalizeRequestInput(body, normalizeOllamaInputItem)
	return normalized, err
}

// normalizeChatGPTRequestBody removes Ollama reasoning items before a native
// request reaches ChatGPT. Ollama's Responses adapter currently serializes
// plaintext thinking as encrypted_content, which ChatGPT correctly rejects as
// invalid ciphertext. Visible messages and tool history remain in the input.
func normalizeChatGPTRequestBody(body []byte) ([]byte, bool, error) {
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
	if header.Type == "" && header.Role != "" {
		return item, true, nil
	}
	switch header.Type {
	case "message", "function_call", "function_call_output":
		return item, true, nil
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
