package proxy

import (
	"bufio"
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
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/klauspost/compress/zstd"
)

const (
	// CodexDesktopPathPrefix is the loopback-only Ollama server route used as
	// Codex's openai_base_url. Codex appends /v1/responses and related paths.
	CodexDesktopPathPrefix = "/api/codex"

	// CodexDesktopModelCatalogFilename is the combined native and Ollama catalog
	// shown by the Codex model picker.
	CodexDesktopModelCatalogFilename = "ollama-launch-models.json"

	// CodexDesktopRoutingCatalogFilename contains only Ollama model slugs.
	// Keeping the router allow-list separate from the picker catalog ensures native Codex
	// models continue to pass through to their normal OpenAI upstream.
	CodexDesktopRoutingCatalogFilename = "ollama-launch-codex-routing.json"

	// CodexDesktopManagedAPIKey is a local sentinel used to let signed-out
	// ChatGPT users open Codex directly with Ollama models. It is never valid for an OpenAI
	// upstream and the router must reject it before any native request leaves
	// the machine.
	CodexDesktopManagedAPIKey = "ollama-local-codex"

	defaultMaxBodyBytes = int64(64 << 20)
	defaultOpenAIURL    = "https://api.openai.com/v1"
)

// CodexDesktopConfig describes the upstreams and Ollama-only routing catalog.
type CodexDesktopConfig struct {
	OllamaURL          string
	ChatGPTURL         string
	OpenAIURL          string
	RoutingCatalogPath string
	ActivityLogPath    string
	MaxBodyBytes       int64
	Logger             *slog.Logger
	Transport          http.RoundTripper
}

// CodexDesktop routes catalog-listed model slugs to Ollama and passes every
// other request to the native upstream selected by Codex's authentication mode.
// The Ollama server mounts this handler on its existing listener, so one
// loopback port serves both routes.
type CodexDesktop struct {
	ollamaURL          *url.URL
	chatGPTURL         *url.URL
	openAIURL          *url.URL
	routingCatalogPath string
	activityLogPath    string
	maxBodyBytes       int64
	logger             *slog.Logger
	proxy              *httputil.ReverseProxy
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

func NewCodexDesktop(config CodexDesktopConfig) (*CodexDesktop, error) {
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

	handler := &CodexDesktop{
		ollamaURL:          ollamaURL,
		chatGPTURL:         chatGPTURL,
		openAIURL:          openAIURL,
		routingCatalogPath: config.RoutingCatalogPath,
		activityLogPath:    strings.TrimSpace(config.ActivityLogPath),
		maxBodyBytes:       maxBodyBytes,
		logger:             logger,
	}
	handler.proxy = &httputil.ReverseProxy{
		Rewrite:        handler.rewrite,
		ModifyResponse: handler.modifyResponse,
		ErrorHandler:   handler.upstreamError,
		Transport:      transport,
		// Flush immediately so upstream SSE streams reach Codex as they arrive.
		FlushInterval: -1,
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

func (h *CodexDesktop) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if !isLoopbackRequest(r) {
		writeJSONError(w, http.StatusForbidden, "Codex proxy only accepts loopback requests")
		return
	}

	suffix, ok := strings.CutPrefix(r.URL.Path, CodexDesktopPathPrefix)
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
	var autoReview autoReviewState
	var routedModel routingModel
	if hasModel {
		catalog, err := loadRoutingCatalog(h.routingCatalogPath)
		if err != nil {
			h.logActivity(started, r.Method, suffix, model, "none", http.StatusServiceUnavailable, "catalog_error")
			writeJSONError(w, http.StatusServiceUnavailable, "read Codex Ollama model catalog: "+err.Error())
			return
		}
		model, decodedBody, err = autoReview.resolveModel(model, catalog, decodedBody)
		if err != nil {
			h.logActivity(started, r.Method, suffix, model, "ollama", http.StatusBadRequest, "request_error")
			writeJSONError(w, http.StatusBadRequest, "prepare Codex Auto-review request for Ollama: "+err.Error())
			return
		}
		routedModel, routed = catalog.models[modelKey(model)]
	}
	decodedBody, err = autoReview.prepareRequest(routed, suffix, decodedBody)
	if err != nil {
		h.logActivity(started, r.Method, suffix, model, "ollama", http.StatusBadRequest, "request_error")
		writeJSONError(w, http.StatusBadRequest, "prepare Codex Auto-review request for Ollama: "+err.Error())
		return
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

	state := &proxyRequest{
		started:      started,
		method:       r.Method,
		suffix:       suffix,
		model:        model,
		hasModel:     hasModel,
		routed:       routed,
		route:        route,
		autoReview:   autoReview,
		target:       targetBase,
		targetSuffix: targetSuffix,
		body:         requestBody,
		normalized:   requestBodyNormalized,
	}
	h.logger.Debug("routing Codex request", "path", suffix, "model", model, "ollama", routed)
	recorder := &responseRecorder{ResponseWriter: w}
	h.proxy.ServeHTTP(recorder, r.WithContext(context.WithValue(r.Context(), proxyRequestKey{}, state)))

	// ReverseProxy owns the response write; recover the terminal result for the
	// activity log from the state its hooks recorded and any client write error.
	result := state.result
	if result == "" {
		result = "ok"
		if recorder.writeErr != nil {
			if r.Context().Err() != nil && errors.Is(recorder.writeErr, r.Context().Err()) {
				result = "canceled"
			} else {
				result = "stream_error"
				h.upstreamErrors.Add(1)
				h.logger.Warn("Codex proxy response write failed", "path", suffix, "model", model, "ollama", routed, "error", recorder.writeErr)
			}
		}
	}
	status := state.status
	if status == 0 {
		status = http.StatusBadGateway
	}
	h.logActivity(started, r.Method, suffix, model, route, status, result)
}

// proxyRequestKey carries the per-request routing state through the reverse
// proxy hooks.
type proxyRequestKey struct{}

// proxyRequest holds the routing decisions ServeHTTP made for one request;
// the reverse proxy hooks apply them and record how the upstream responded.
type proxyRequest struct {
	started      time.Time
	method       string
	suffix       string
	model        string
	hasModel     bool
	routed       bool
	route        string
	autoReview   autoReviewState
	target       *url.URL
	targetSuffix string
	body         []byte
	normalized   bool
	status       int    // upstream response status once known
	result       string // terminal activity-log result set by a hook
}

// proxyError routes a hook failure to the exact response and activity-log
// result the request path requires.
type proxyError struct {
	status  int
	message string
	result  string
}

func (e *proxyError) Error() string { return e.message }

func (h *CodexDesktop) rewrite(pr *httputil.ProxyRequest) {
	state := pr.In.Context().Value(proxyRequestKey{}).(*proxyRequest)
	pr.Out.URL = resolveTarget(state.target, state.targetSuffix, pr.In.URL.RawQuery)
	// An empty Host makes the outbound Host header follow the upstream URL.
	pr.Out.Host = ""
	pr.Out.Body = io.NopCloser(bytes.NewReader(state.body))
	pr.Out.ContentLength = int64(len(state.body))
	if state.routed {
		// Only forward headers Ollama accepts; this also keeps Codex
		// credentials from ever reaching the local server.
		pr.Out.Header = make(http.Header)
		copyOllamaRequestHeaders(pr.Out.Header, pr.In.Header)
		return
	}
	if state.normalized {
		pr.Out.Header.Del("Content-Encoding")
	}
	// ReverseProxy already stripped hop-by-hop and connection-token headers.
	// X-Forwarded-* is opt-in via SetXForwarded and stays unset, matching the
	// wire behavior of the previous hand-rolled forwarder.
}

func (h *CodexDesktop) modifyResponse(resp *http.Response) error {
	state := resp.Request.Context().Value(proxyRequestKey{}).(*proxyRequest)
	state.status = resp.StatusCode
	h.lastRoute.Store(routeSnapshot{Model: state.model, Route: state.route, UpstreamStatus: resp.StatusCode})
	if resp.StatusCode >= http.StatusInternalServerError {
		h.upstreamErrors.Add(1)
		state.result = "upstream_error"
	}
	if isAcceptedModelRequest(state.method, state.suffix, state.hasModel, resp.StatusCode) {
		if state.routed {
			h.ollamaRequests.Add(1)
		} else {
			h.chatGPTRequests.Add(1)
		}
	}
	if !state.autoReview.buffersResponse(resp.StatusCode) {
		return nil
	}
	responseBody, err := io.ReadAll(io.LimitReader(resp.Body, h.maxBodyBytes+1))
	if err != nil {
		h.upstreamErrors.Add(1)
		return &proxyError{
			status:  http.StatusBadGateway,
			message: "read Codex Auto-review response from Ollama: " + err.Error(),
			result:  "stream_error",
		}
	}
	if int64(len(responseBody)) > h.maxBodyBytes {
		h.upstreamErrors.Add(1)
		return &proxyError{
			status:  http.StatusBadGateway,
			message: fmt.Sprintf("Codex Auto-review response exceeds %d bytes", h.maxBodyBytes),
			result:  "response_error",
		}
	}
	responseBody, _, err = transformAutoReviewResponse(responseBody, resp.Header.Get("Content-Type"))
	if err != nil {
		h.upstreamErrors.Add(1)
		return &proxyError{
			status:  http.StatusBadGateway,
			message: "invalid Codex Auto-review response from Ollama: " + err.Error(),
			result:  "response_error",
		}
	}
	// Let the server compute the framing, as the previous write path did.
	resp.Body.Close()
	resp.Body = io.NopCloser(bytes.NewReader(responseBody))
	resp.ContentLength = -1
	resp.Header.Del("Content-Length")
	return nil
}

func (h *CodexDesktop) upstreamError(w http.ResponseWriter, r *http.Request, err error) {
	state, _ := r.Context().Value(proxyRequestKey{}).(*proxyRequest)
	if state != nil {
		state.status = http.StatusBadGateway
	}
	var handled *proxyError
	if errors.As(err, &handled) {
		if state != nil {
			state.result = handled.result
		}
		writeJSONError(w, handled.status, handled.message)
		return
	}
	h.upstreamErrors.Add(1)
	if state != nil {
		state.result = "upstream_error"
	}
	writeJSONError(w, http.StatusBadGateway, err.Error())
}

// responseRecorder records how the response write to Codex went so
// ServeHTTP can log the terminal result of the request.
type responseRecorder struct {
	http.ResponseWriter
	writeErr error
}

func (r *responseRecorder) Write(body []byte) (int, error) {
	n, err := r.ResponseWriter.Write(body)
	if err != nil {
		r.writeErr = err
	}
	return n, err
}

func (r *responseRecorder) Flush() {
	if flusher, ok := r.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
	}
}

func (r *responseRecorder) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	if hijacker, ok := r.ResponseWriter.(http.Hijacker); ok {
		return hijacker.Hijack()
	}
	return nil, nil, errors.New("response writer does not support hijacking")
}

func (h *CodexDesktop) logActivity(started time.Time, method, path, model, route string, status int, result string) {
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

func (h *CodexDesktop) writeActivity(format string, args ...any) {
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

func (h *CodexDesktop) writeStatus(w http.ResponseWriter) {
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

func (h *CodexDesktop) readBodies(r *http.Request) ([]byte, []byte, error) {
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
	for _, raw := range r.Header.Values("Connection") {
		for _, token := range strings.Split(raw, ",") {
			if strings.EqualFold(strings.TrimSpace(token), "upgrade") {
				return true
			}
		}
	}
	return false
}

func usesChatGPTBackend(header http.Header) bool {
	return strings.TrimSpace(header.Get("ChatGPT-Account-ID")) != ""
}

func usesManagedAPIKey(header http.Header) bool {
	parts := strings.Fields(header.Get("Authorization"))
	return len(parts) == 2 && strings.EqualFold(parts[0], "Bearer") && parts[1] == CodexDesktopManagedAPIKey
}

func copyOllamaRequestHeaders(dst, src http.Header) {
	for _, key := range []string{"Accept", "Content-Type", "OpenAI-Beta", "User-Agent"} {
		for _, value := range src.Values(key) {
			dst.Add(key, value)
		}
	}
}

func writeJSONError(w http.ResponseWriter, status int, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]string{"error": message})
}
