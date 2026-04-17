package server

// Optional HTTP webhooks that inspect, modify, block, or request
// approval for each inference request/response. See
// docs/inference-webhooks.mdx for the wire protocol.

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
)

type inferenceHook struct {
	preURL  string
	postURL string
	timeout time.Duration
	onError string // "deny" (fail-closed) or "allow" (fail-open)
	headers http.Header
	client  *http.Client
}

// newInferenceHook returns a configured hook if any OLLAMA_HOOK_*_URL is
// set, nil otherwise. Invalid config returns an error so startup fails
// fast.
func newInferenceHook() (*inferenceHook, error) {
	pre := envconfig.HookPreInferenceURL()
	post := envconfig.HookPostInferenceURL()
	if pre == "" && post == "" {
		return nil, nil
	}

	if err := validateHookURL("OLLAMA_HOOK_PRE_INFERENCE_URL", pre); err != nil {
		return nil, err
	}
	if err := validateHookURL("OLLAMA_HOOK_POST_INFERENCE_URL", post); err != nil {
		return nil, err
	}

	timeout := envconfig.HookTimeout()
	if timeout == 0 {
		timeout = 5 * time.Second
	}
	onErr := strings.ToLower(envconfig.HookOnError())
	if onErr != "allow" {
		onErr = "deny"
	}

	headers := http.Header{}
	for _, hv := range envconfig.HookHeaders() {
		if k, v, ok := strings.Cut(hv, ":"); ok {
			headers.Set(strings.TrimSpace(k), strings.TrimSpace(v))
		}
	}

	return &inferenceHook{
		preURL:  pre,
		postURL: post,
		timeout: timeout,
		onError: onErr,
		headers: headers,
		client:  &http.Client{Timeout: timeout},
	}, nil
}

// validateHookURL rejects malformed or non-http(s) hook URLs at startup.
func validateHookURL(env, raw string) error {
	if raw == "" {
		return nil
	}
	u, err := url.Parse(raw)
	if err != nil {
		return fmt.Errorf("%s: invalid URL %q: %w", env, raw, err)
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return fmt.Errorf("%s: URL scheme must be http or https, got %q", env, u.Scheme)
	}
	if u.Host == "" {
		return fmt.Errorf("%s: URL missing host: %q", env, raw)
	}
	return nil
}

func (s *Server) initInferenceHook() error {
	h, err := newInferenceHook()
	if err != nil {
		return err
	}
	s.inferenceHook = h
	if h != nil {
		slog.Info("inference webhooks enabled",
			"pre_url", redactURL(h.preURL),
			"post_url", redactURL(h.postURL),
			"on_error", h.onError,
			"timeout", h.timeout)
	}
	return nil
}

// withInferenceHook prepends the pre-inference middleware when a pre URL
// is configured, otherwise returns handlers unchanged. The post hook is
// invoked from within the handlers via applyPostInference.
func (s *Server) withInferenceHook(route string, handlers ...gin.HandlerFunc) []gin.HandlerFunc {
	if s.inferenceHook == nil || s.inferenceHook.preURL == "" {
		return handlers
	}
	return append([]gin.HandlerFunc{s.inferenceHook.preMiddleware(route)}, handlers...)
}

// hookedChain composes requestLogging + protocol-conversion + preHook +
// handler for r.POST. Order: [requestLogging, convert..., preHook?, handler].
func (s *Server) hookedChain(route string, convert []gin.HandlerFunc, handler gin.HandlerFunc) []gin.HandlerFunc {
	chain := append(convert, s.withInferenceHook(route, handler)...)
	return s.withInferenceRequestLogging(route, chain...)
}

// HookSchemaVersion is stamped on every HookRequest. Hooks should
// branch on it. Bump on any backwards-incompatible change.
const HookSchemaVersion = 1

// HookRequest is the JSON payload POSTed to a hook URL.
type HookRequest struct {
	SchemaVersion  int            `json:"schema_version"`
	Event          string         `json:"event"`
	RequestID      string         `json:"request_id"`
	Route          string         `json:"route"`
	Model          string         `json:"model,omitempty"`
	Messages       []HookMessage  `json:"messages,omitempty"`
	Tools          []HookTool     `json:"tools,omitempty"`
	Options        map[string]any `json:"options,omitempty"`
	OutputText     string         `json:"output_text,omitempty"`
	OutputThinking string         `json:"output_thinking,omitempty"`
	ToolCalls      []HookToolCall `json:"tool_calls,omitempty"`
}

// HookMessage is an OpenAI-format chat message. Ollama's native shape is
// normalized to this so the wire contract is stable across /api/chat,
// /v1/chat/completions, and /v1/messages. api.Message.Images is not
// propagated in v1.
type HookMessage struct {
	Role       string         `json:"role"`
	Content    string         `json:"content"`
	Thinking   string         `json:"thinking,omitempty"`
	ToolCalls  []HookToolCall `json:"tool_calls,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
	Name       string         `json:"name,omitempty"`
}

// HookTool is the OpenAI-format tool (function) definition.
type HookTool struct {
	Type     string      `json:"type"`
	Function HookToolDef `json:"function"`
}

type HookToolDef struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

type HookToolCall struct {
	ID       string         `json:"id,omitempty"`
	Type     string         `json:"type"`
	Function HookToolCallFn `json:"function"`
}

type HookToolCallFn struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// HookResponse is the JSON body expected from a hook URL. The
// permission verbs align with Cursor's hooks contract; "modify" is an
// Ollama-specific extension. See docs/inference-webhooks.mdx.
type HookResponse struct {
	Permission     string         `json:"permission"` // "allow" | "deny" | "ask" | "modify"
	UserMessage    string         `json:"user_message,omitempty"`
	AgentMessage   string         `json:"agent_message,omitempty"`
	Messages       []HookMessage  `json:"messages,omitempty"`        // pre: modify
	OutputText     string         `json:"output_text,omitempty"`     // post: modify
	OutputThinking string         `json:"output_thinking,omitempty"` // post: modify
	ToolCalls      []HookToolCall `json:"tool_calls,omitempty"`      // post: modify
}

// preMiddleware reads the inbound body, calls the hook, and applies
// the returned permission. Must run AFTER format-conversion middleware
// on /v1/* routes so it sees the normalized api.ChatRequest /
// api.GenerateRequest body.
func (h *inferenceHook) preMiddleware(route string) gin.HandlerFunc {
	return func(c *gin.Context) {
		if c.Request == nil || c.Request.Body == nil {
			c.Next()
			return
		}
		// +1 byte so we can tell "at limit" from "oversized".
		body, err := io.ReadAll(io.LimitReader(c.Request.Body, maxInboundBody+1))
		if err != nil {
			slog.Warn("inference hook: read body", "route", route, "err", err)
			c.Request.Body = io.NopCloser(bytes.NewReader(body))
			c.Next()
			return
		}
		if int64(len(body)) > maxInboundBody {
			c.AbortWithStatusJSON(http.StatusRequestEntityTooLarge, gin.H{
				"error": fmt.Sprintf("request body exceeds %d bytes", maxInboundBody),
			})
			return
		}
		c.Request.Body = io.NopCloser(bytes.NewReader(body))

		reqID := uuid.NewString()
		c.Set("inference_hook_request_id", reqID)
		c.Header(headerRequestID, reqID)

		payload, buildErr := buildPrePayload(route, reqID, body)
		if buildErr != nil {
			// Non-inference body; pass through.
			slog.Debug("inference hook: non-inference body", "route", route, "err", buildErr)
			c.Next()
			return
		}

		res, err := h.call(c.Request.Context(), h.preURL, "pre_inference", payload)
		if err != nil {
			if h.onError == "allow" {
				logFailOpen("pre", route, err)
				c.Next()
				return
			}
			slog.Warn("inference hook: pre call failed, fail-closed", "route", route, "err", err)
			c.AbortWithStatusJSON(http.StatusServiceUnavailable, unavailableBody())
			return
		}

		switch res.Permission {
		case "deny":
			user, agent := sanitizeReason(res.UserMessage), sanitizeReason(res.AgentMessage)
			c.AbortWithStatusJSON(http.StatusBadRequest, denyBody("denied", user, agent))
			return
		case "ask":
			user, agent := sanitizeReason(res.UserMessage), sanitizeReason(res.AgentMessage)
			c.AbortWithStatusJSON(http.StatusForbidden, askBody(user, agent))
			return
		case "modify":
			newBody, err := applyPreModify(body, res.Messages)
			if err != nil {
				// Shape-unsupported is hook misconfiguration, not
				// transient failure; bypass onError=allow so every
				// modify doesn't silently drop.
				if errors.Is(err, errModifyShapeUnsupported) {
					slog.Warn("inference hook: modify shape unsupported", "route", route, "err", err)
					c.AbortWithStatusJSON(http.StatusBadGateway, modifyFailedBody(err.Error()))
					return
				}
				slog.Warn("inference hook: apply modify", "route", route, "err", err)
				if h.onError == "allow" {
					c.Next()
					return
				}
				c.AbortWithStatusJSON(http.StatusInternalServerError, modifyFailedBody(""))
				return
			}
			c.Request.Body = io.NopCloser(bytes.NewReader(newBody))
			c.Request.ContentLength = int64(len(newBody))
			c.Next()
			return
		case "", "allow":
			c.Next()
			return
		default:
			// Unknown permission = misconfig; refuse rather than silently allow.
			perm := sanitizeReason(res.Permission)
			slog.Warn("inference hook: unknown permission, treating as deny",
				"route", route, "permission", perm)
			c.AbortWithStatusJSON(http.StatusBadRequest, denyBody(
				"denied",
				"hook returned unknown permission",
				"hook returned unknown permission "+perm,
			))
			return
		}
	}
}

// buildPrePayload parses the request body as a chat or generate request and
// builds a normalized HookRequest. Returns an error if the body doesn't
// parse as either.
func buildPrePayload(route, requestID string, body []byte) (HookRequest, error) {
	hp := HookRequest{
		SchemaVersion: HookSchemaVersion,
		Event:         "pre_inference",
		RequestID:     requestID,
		Route:         route,
	}

	// Try chat first.
	var chat api.ChatRequest
	if err := json.Unmarshal(body, &chat); err == nil && len(chat.Messages) > 0 {
		hp.Model = chat.Model
		hp.Messages = messagesToHook(chat.Messages)
		hp.Tools = toolsToHook(chat.Tools)
		hp.Options = chat.Options
		return hp, nil
	}

	// Then generate.
	var gen api.GenerateRequest
	if err := json.Unmarshal(body, &gen); err == nil && (gen.Prompt != "" || gen.System != "") {
		hp.Model = gen.Model
		if gen.System != "" {
			hp.Messages = append(hp.Messages, HookMessage{Role: "system", Content: gen.System})
		}
		if gen.Prompt != "" {
			hp.Messages = append(hp.Messages, HookMessage{Role: "user", Content: gen.Prompt})
		}
		hp.Options = gen.Options
		return hp, nil
	}

	return hp, errors.New("not a chat or generate request")
}

// errModifyShapeUnsupported is returned when a modify response cannot
// be represented by the inbound request shape (today: /api/generate,
// which accepts only one system + one user prompt).
var errModifyShapeUnsupported = errors.New("modify result unsupported on this route")

// applyPreModify replaces the messages in the body with those from the
// hook response, preserving other request fields.
//
// Invariant: we decode to map[string]json.RawMessage so any top-level
// field absent from HookMessage (including future additions to
// api.ChatRequest) round-trips byte-for-byte. Modify only touches
// "messages" (chat) or "prompt"/"system" (generate).
func applyPreModify(body []byte, modified []HookMessage) ([]byte, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, err
	}
	if _, hasMessages := raw["messages"]; hasMessages {
		modRaw, err := json.Marshal(messagesFromHook(modified))
		if err != nil {
			return nil, err
		}
		raw["messages"] = modRaw
		return json.Marshal(raw)
	}
	// Generate path: refuse anything we cannot represent as system + user
	// rather than silently dropping the rest.
	if err := validateGenerateModifyShape(modified); err != nil {
		return nil, err
	}
	var newSystem, newPrompt string
	for _, m := range modified {
		if strings.EqualFold(m.Role, "system") {
			newSystem = m.Content
		} else {
			newPrompt = m.Content
		}
	}
	if b, err := json.Marshal(newPrompt); err == nil {
		raw["prompt"] = b
	}
	if newSystem != "" {
		if b, err := json.Marshal(newSystem); err == nil {
			raw["system"] = b
		}
	}
	return json.Marshal(raw)
}

// validateGenerateModifyShape enforces the /api/generate constraint:
// at most one system + exactly one user (empty role counts as user).
func validateGenerateModifyShape(messages []HookMessage) error {
	var systemCount, userCount int
	for _, m := range messages {
		switch strings.ToLower(m.Role) {
		case "system":
			systemCount++
		case "user", "":
			userCount++
		default:
			return fmt.Errorf("%w: cannot carry role %q on /api/generate; use a chat endpoint for multi-role rewrites",
				errModifyShapeUnsupported, m.Role)
		}
	}
	if systemCount > 1 {
		return fmt.Errorf("%w: produced %d system messages; /api/generate accepts at most one",
			errModifyShapeUnsupported, systemCount)
	}
	if userCount != 1 {
		return fmt.Errorf("%w: produced %d user messages; /api/generate requires exactly one",
			errModifyShapeUnsupported, userCount)
	}
	return nil
}

// postInferenceUnavailable is an internal permission returned when the
// post-hook is unreachable under fail-closed. Kept distinct from "deny"
// so 503 (infrastructure) doesn't collide with 400 (policy).
const postInferenceUnavailable = "unavailable"

// PostInferenceResult is the outcome of a post-inference hook call.
type PostInferenceResult struct {
	// Permission: "allow" | "deny" | "ask" | "modify" | "unavailable".
	// "deny", "ask", and "unavailable" are terminal (400 / 403 / 503).
	Permission     string
	UserMessage    string
	AgentMessage   string
	OutputText     string
	OutputThinking string
	ToolCalls      []HookToolCall
}

// HTTPStatus returns the status code for a terminal verdict, or 0 for
// non-terminal ones (allow/modify).
func (r PostInferenceResult) HTTPStatus() int {
	switch r.Permission {
	case "deny":
		return http.StatusBadRequest
	case "ask":
		return http.StatusForbidden
	case postInferenceUnavailable:
		return http.StatusServiceUnavailable
	default:
		return 0
	}
}

// Terminated reports whether the caller must abort the response.
func (r PostInferenceResult) Terminated() bool {
	return r.Permission == "deny" || r.Permission == "ask" || r.Permission == postInferenceUnavailable
}

func (s *Server) callPostInference(c *gin.Context, route, model, outputText, outputThinking string, toolCalls []HookToolCall) PostInferenceResult {
	if s.inferenceHook == nil || s.inferenceHook.postURL == "" {
		return PostInferenceResult{
			Permission:     "allow",
			OutputText:     outputText,
			OutputThinking: outputThinking,
			ToolCalls:      toolCalls,
		}
	}

	// Reuse the pre-middleware's request id when present; otherwise
	// generate one for post-only deployments so the hook call and the
	// client response correlate.
	reqIDStr, _ := c.Get("inference_hook_request_id")
	reqID, _ := reqIDStr.(string)
	if reqID == "" {
		reqID = uuid.NewString()
		c.Set("inference_hook_request_id", reqID)
	}
	c.Header(headerRequestID, reqID)

	payload := HookRequest{
		SchemaVersion:  HookSchemaVersion,
		Event:          "post_inference",
		RequestID:      reqID,
		Route:          route,
		Model:          model,
		OutputText:     outputText,
		OutputThinking: outputThinking,
		ToolCalls:      toolCalls,
	}

	res, err := s.inferenceHook.call(c.Request.Context(), s.inferenceHook.postURL, "post_inference", payload)
	if err != nil {
		if s.inferenceHook.onError == "allow" {
			logFailOpen("post", route, err)
			return PostInferenceResult{
				Permission:     "allow",
				OutputText:     outputText,
				OutputThinking: outputThinking,
				ToolCalls:      toolCalls,
			}
		}
		slog.Warn("inference hook: post call failed, fail-closed", "route", route, "err", err)
		return PostInferenceResult{
			Permission:  postInferenceUnavailable,
			UserMessage: "post hook unavailable",
		}
	}

	switch res.Permission {
	case "deny":
		return PostInferenceResult{
			Permission:   "deny",
			UserMessage:  res.UserMessage,
			AgentMessage: res.AgentMessage,
		}
	case "ask":
		return PostInferenceResult{
			Permission:   "ask",
			UserMessage:  res.UserMessage,
			AgentMessage: res.AgentMessage,
		}
	case "modify":
		if res.OutputText != "" {
			outputText = res.OutputText
		}
		if res.OutputThinking != "" {
			outputThinking = res.OutputThinking
		}
		if res.ToolCalls != nil {
			toolCalls = res.ToolCalls
		}
		return PostInferenceResult{
			Permission:     "modify",
			UserMessage:    res.UserMessage,
			AgentMessage:   res.AgentMessage,
			OutputText:     outputText,
			OutputThinking: outputThinking,
			ToolCalls:      toolCalls,
		}
	default:
		return PostInferenceResult{
			Permission:     "allow",
			OutputText:     outputText,
			OutputThinking: outputThinking,
			ToolCalls:      toolCalls,
		}
	}
}

// PostInference invokes the post-inference hook. Returns an allow
// result when the hook is unconfigured so handlers can call it
// unconditionally.
func (s *Server) PostInference(c *gin.Context, route, model, outputText, outputThinking string, toolCalls []HookToolCall) PostInferenceResult {
	return s.callPostInference(c, route, model, outputText, outputThinking, toolCalls)
}

// PostInferenceConfigured reports whether a post-inference URL is set.
func (s *Server) PostInferenceConfigured() bool {
	return s.inferenceHook != nil && s.inferenceHook.postURL != ""
}

// applyPostInference invokes the post-hook and applies its verdict. On
// a terminal verdict it writes the HTTP body and returns done=true so
// the caller can return.
func (s *Server) applyPostInference(c *gin.Context, route, model, outputText, outputThinking string, toolCalls []api.ToolCall) (string, string, []api.ToolCall, bool) {
	if !s.PostInferenceConfigured() {
		return outputText, outputThinking, toolCalls, false
	}
	verdict := s.PostInference(c, route, model, outputText, outputThinking, toolCallsToHook(toolCalls))
	if !verdict.Terminated() {
		return verdict.OutputText, verdict.OutputThinking, toolCallsFromHook(verdict.ToolCalls), false
	}
	switch verdict.Permission {
	case postInferenceUnavailable:
		c.AbortWithStatusJSON(http.StatusServiceUnavailable, unavailableBody())
	case "ask":
		c.AbortWithStatusJSON(verdict.HTTPStatus(), askBody(
			sanitizeReason(verdict.UserMessage),
			sanitizeReason(verdict.AgentMessage),
		))
	default:
		c.AbortWithStatusJSON(verdict.HTTPStatus(), denyBody(
			"denied",
			sanitizeReason(verdict.UserMessage),
			sanitizeReason(verdict.AgentMessage),
		))
	}
	return outputText, outputThinking, toolCalls, true
}

// WarnStreamingPostHookSkipped emits a one-shot warning when a streamed
// request arrives with a post-hook configured; streaming bypasses the
// post hook today (see docs/inference-webhooks.mdx).
func (s *Server) WarnStreamingPostHookSkipped(route string) {
	if !s.PostInferenceConfigured() {
		return
	}
	streamingPostHookWarnOnce.Do(func() {
		slog.Warn("inference hook: post-inference webhook is skipped for streaming responses; "+
			"set stream=false if post-inference enforcement is required",
			"route", route)
	})
}

var streamingPostHookWarnOnce sync.Once

// call is the shared HTTP round-trip helper for pre and post.
func (h *inferenceHook) call(ctx context.Context, url, event string, payload HookRequest) (HookResponse, error) {
	var out HookResponse
	buf, err := json.Marshal(payload)
	if err != nil {
		return out, err
	}

	reqCtx, cancel := context.WithTimeout(ctx, h.timeout)
	defer cancel()

	req, err := http.NewRequestWithContext(reqCtx, http.MethodPost, url, bytes.NewReader(buf))
	if err != nil {
		return out, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", hookUserAgent)
	req.Header.Set("X-Ollama-Hook-Event", event)
	req.Header.Set(headerRequestID, payload.RequestID)
	for k, vals := range h.headers {
		for _, v := range vals {
			req.Header.Add(k, v)
		}
	}

	resp, err := h.client.Do(req)
	if err != nil {
		return out, err
	}
	defer resp.Body.Close()

	// +1 byte so we can tell "at limit" from "oversized" below.
	respBody, err := io.ReadAll(io.LimitReader(resp.Body, maxHookBody+1))
	if err != nil {
		return out, fmt.Errorf("read hook body: %w", err)
	}
	if int64(len(respBody)) > maxHookBody {
		return out, fmt.Errorf("hook response body exceeds %d bytes", maxHookBody)
	}
	if resp.StatusCode >= 300 {
		return out, fmt.Errorf("hook http %d: %s", resp.StatusCode, truncateForError(string(respBody), 256))
	}
	if err := json.Unmarshal(respBody, &out); err != nil {
		return out, fmt.Errorf("hook body decode: %w", err)
	}
	return out, nil
}

// maxHookBody caps the size of a hook response body we will buffer.
const maxHookBody int64 = 4 << 20

func messagesToHook(in []api.Message) []HookMessage {
	out := make([]HookMessage, 0, len(in))
	for _, m := range in {
		hm := HookMessage{
			Role:       strings.ToLower(m.Role),
			Content:    m.Content,
			Thinking:   m.Thinking,
			ToolCallID: m.ToolCallID,
			Name:       m.ToolName,
		}
		for _, tc := range m.ToolCalls {
			hm.ToolCalls = append(hm.ToolCalls, HookToolCall{
				ID:   tc.ID,
				Type: "function",
				Function: HookToolCallFn{
					Name:      tc.Function.Name,
					Arguments: tc.Function.Arguments.String(),
				},
			})
		}
		out = append(out, hm)
	}
	return out
}

// toolCallsToHook converts api.ToolCall to the wire-format HookToolCall.
// Arguments are serialized as a JSON string (OpenAI convention).
func toolCallsToHook(in []api.ToolCall) []HookToolCall {
	if len(in) == 0 {
		return nil
	}
	out := make([]HookToolCall, 0, len(in))
	for _, tc := range in {
		out = append(out, HookToolCall{
			ID:   tc.ID,
			Type: "function",
			Function: HookToolCallFn{
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments.String(),
			},
		})
	}
	return out
}

// toolCallsFromHook reverses toolCallsToHook. Parse failures leave
// Arguments empty rather than reflecting the attacker-controlled string.
func toolCallsFromHook(in []HookToolCall) []api.ToolCall {
	if len(in) == 0 {
		return nil
	}
	out := make([]api.ToolCall, 0, len(in))
	for _, tc := range in {
		args := api.NewToolCallFunctionArguments()
		if tc.Function.Arguments != "" {
			_ = json.Unmarshal([]byte(tc.Function.Arguments), &args)
		}
		out = append(out, api.ToolCall{
			ID: tc.ID,
			Function: api.ToolCallFunction{
				Name:      tc.Function.Name,
				Arguments: args,
			},
		})
	}
	return out
}

// messagesFromHook reverses messagesToHook. api.Message.Images is not
// carried in the v1 wire format, so a multimodal modify drops images.
func messagesFromHook(in []HookMessage) []api.Message {
	out := make([]api.Message, 0, len(in))
	for _, m := range in {
		out = append(out, api.Message{
			Role:       m.Role,
			Content:    m.Content,
			Thinking:   m.Thinking,
			ToolName:   m.Name,
			ToolCallID: m.ToolCallID,
			ToolCalls:  toolCallsFromHook(m.ToolCalls),
		})
	}
	return out
}

func toolsToHook(in api.Tools) []HookTool {
	if len(in) == 0 {
		return nil
	}
	out := make([]HookTool, 0, len(in))
	for _, t := range in {
		ht := HookTool{Type: t.Type}
		ht.Function.Name = t.Function.Name
		ht.Function.Description = t.Function.Description
		if t.Function.Parameters.Properties != nil {
			params := map[string]any{
				"type": t.Function.Parameters.Type,
			}
			// Best-effort; not round-tripped.
			if raw, err := json.Marshal(t.Function.Parameters); err == nil {
				var decoded map[string]any
				if json.Unmarshal(raw, &decoded) == nil {
					params = decoded
				}
			}
			ht.Function.Parameters = params
		}
		out = append(out, ht)
	}
	return out
}

// hookUserAgent identifies Ollama in outbound hook calls. Bumped
// alongside the wire-format major.
const hookUserAgent = "ollama-hooks/1"

// headerRequestID is stamped on outbound hook calls and reflected on
// responses so audit logs can correlate the two ends.
const headerRequestID = "X-Ollama-Request-Id"

// hookErrorBody is the standard envelope for hook-decision responses:
// top-level "error" for clients that read only that field, plus a
// nested "hook" object with the structured fields. Empty user_message
// and agent_message are omitted from the nested object.
func hookErrorBody(message, permission, userMessage, agentMessage string) gin.H {
	hook := gin.H{"permission": permission}
	if userMessage != "" {
		hook["user_message"] = userMessage
	}
	if agentMessage != "" {
		hook["agent_message"] = agentMessage
	}
	return gin.H{
		"error": message,
		"hook":  hook,
	}
}

func denyBody(label, userMessage, agentMessage string) gin.H {
	msg := label + " by inference hook"
	if userMessage != "" {
		msg = msg + ": " + userMessage
	}
	return hookErrorBody(msg, "deny", userMessage, agentMessage)
}

func askBody(userMessage, agentMessage string) gin.H {
	msg := "approval required by inference hook"
	if userMessage != "" {
		msg = msg + ": " + userMessage
	}
	return hookErrorBody(msg, "ask", userMessage, agentMessage)
}

// unavailableBody is returned on fail-closed when the hook is
// unreachable. The "unavailable" permission distinguishes it from a
// hook-returned deny.
func unavailableBody() gin.H {
	return hookErrorBody("inference hook unavailable", postInferenceUnavailable, "", "")
}

// modifyFailedBody is returned when Ollama cannot apply a hook's
// modify response. detail is appended to the error message.
func modifyFailedBody(detail string) gin.H {
	msg := "inference hook modify failed"
	if detail != "" {
		msg = msg + ": " + detail
	}
	return hookErrorBody(msg, "modify", "", "")
}

// truncateForError keeps error messages a bounded length.
func truncateForError(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

// maxInboundBody caps the request body the hook middleware will
// buffer; exceeding returns 413.
const maxInboundBody int64 = 32 << 20 // 32 MiB

// sanitizeReason bounds a hook-provided reason string and strips
// control characters to prevent log corruption and HTTP response
// splitting when reflected into a response body.
func sanitizeReason(s string) string {
	if s == "" {
		return ""
	}
	const max = 256
	if len(s) > max {
		s = s[:max]
	}
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range s {
		switch {
		case r == '\n', r == '\r', r == '\t':
			b.WriteByte(' ')
		case r < 0x20, r == 0x7f:
			// drop other control chars
		default:
			b.WriteRune(r)
		}
	}
	return b.String()
}

// redactURL returns raw with userinfo replaced by "REDACTED" so
// embedded credentials don't spill into logs. Parse failures return an
// opaque marker.
func redactURL(raw string) string {
	if raw == "" {
		return ""
	}
	u, err := url.Parse(raw)
	if err != nil {
		return "<unparseable url>"
	}
	if u.User != nil {
		u.User = url.User("REDACTED")
	}
	return u.String()
}

// logFailOpen emits a Warn the first time a channel ("pre" or "post")
// fails open in this process, then demotes subsequent events to Debug.
func logFailOpen(channel, route string, err error) {
	var first bool
	switch channel {
	case "pre":
		preFailOpenOnce.Do(func() { first = true })
	case "post":
		postFailOpenOnce.Do(func() { first = true })
	}
	if first {
		slog.Warn("inference hook: fail-open engaged; subsequent fail-opens on this channel demoted to debug",
			"channel", channel, "route", route, "err", err)
		return
	}
	slog.Debug("inference hook: fail-open", "channel", channel, "route", route, "err", err)
}

var (
	preFailOpenOnce  sync.Once
	postFailOpenOnce sync.Once
)
