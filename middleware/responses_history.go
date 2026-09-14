package middleware

import (
	"bytes"
	"container/list"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/klauspost/compress/zstd"

	"github.com/ollama/ollama/openai"
)

const (
	responsesHistoryBytes   = 64 << 20
	responsesHistoryEntries = 1024
	responsesHistoryTTL     = 30 * time.Minute
)

type responsesHistoryKey struct {
	id         string
	credential [32]byte
}

type responsesHistoryEntry struct {
	key     responsesHistoryKey
	input   json.RawMessage
	expires time.Time
}

// A cache belongs to one router, not the process or an upstream provider. Entries
// own immutable JSON, so parallel continuations can safely branch from any turn.
type responsesHistory struct {
	mu                   sync.Mutex
	entries              map[responsesHistoryKey]*list.Element
	order                list.List
	bytes                int
	maxBytes, maxEntries int
	ttl                  time.Duration
	now                  func() time.Time
}

func newResponsesHistory() *responsesHistory {
	return &responsesHistory{
		entries:  make(map[responsesHistoryKey]*list.Element),
		maxBytes: responsesHistoryBytes, maxEntries: responsesHistoryEntries,
		ttl: responsesHistoryTTL, now: time.Now,
	}
}

func (h *responsesHistory) remove(e *list.Element) {
	entry := e.Value.(responsesHistoryEntry)
	delete(h.entries, entry.key)
	h.bytes -= len(entry.input)
	h.order.Remove(e)
}

func (h *responsesHistory) get(key responsesHistoryKey) (json.RawMessage, bool) {
	h.mu.Lock()
	defer h.mu.Unlock()
	e, ok := h.entries[key]
	if !ok {
		return nil, false
	}
	entry := e.Value.(responsesHistoryEntry)
	if !h.now().Before(entry.expires) {
		h.remove(e)
		return nil, false
	}
	return entry.input, true
}

func (h *responsesHistory) put(key responsesHistoryKey, input json.RawMessage) bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	if len(input) > h.maxBytes || len(input) > maxDecompressedBodySize || h.maxEntries < 1 {
		return false
	}
	if old := h.entries[key]; old != nil {
		h.remove(old)
	}
	now := h.now()
	for e := h.order.Front(); e != nil; e = h.order.Front() {
		entry := e.Value.(responsesHistoryEntry)
		if now.Before(entry.expires) && h.bytes+len(input) <= h.maxBytes && h.order.Len() < h.maxEntries {
			break
		}
		h.remove(e)
	}
	entry := responsesHistoryEntry{key: key, input: bytes.Clone(input), expires: now.Add(h.ttl)}
	h.entries[key] = h.order.PushBack(entry)
	h.bytes += len(input)
	return true
}

// ResponsesHistoryMiddleware must run before both cloud passthrough and local
// Responses conversion. Keeping the original wire items preserves reasoning,
// function call IDs, multimodal tool results and provider-specific fields.
func ResponsesHistoryMiddleware() gin.HandlerFunc {
	return newResponsesHistory().middleware()
}

func (h *responsesHistory) middleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		if c.GetHeader("Content-Encoding") == "zstd" {
			reader, err := zstd.NewReader(c.Request.Body, zstd.WithDecoderMaxMemory(8<<20))
			if err != nil {
				c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "failed to decompress zstd body"))
				return
			}
			defer reader.Close()
			c.Request.Body = io.NopCloser(reader)
			c.Request.Header.Del("Content-Encoding")
		}
		body, err := io.ReadAll(http.MaxBytesReader(c.Writer, c.Request.Body, maxDecompressedBodySize))
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}
		var fields map[string]json.RawMessage
		var req struct {
			PreviousResponseID *string         `json:"previous_response_id"`
			Store              *bool           `json:"store"`
			Stream             bool            `json:"stream"`
			Background         bool            `json:"background"`
			Conversation       json.RawMessage `json:"conversation"`
			Input              json.RawMessage `json:"input"`
		}
		if err := json.Unmarshal(body, &req); err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}
		if err := json.Unmarshal(body, &fields); err != nil || fields == nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "request must be an object"))
			return
		}
		if req.Background || (len(req.Conversation) > 0 && !bytes.Equal(bytes.TrimSpace(req.Conversation), []byte("null"))) {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "background and conversation are not supported; use previous_response_id for continuation"))
			return
		}
		input, err := responsesInputItems(req.Input)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}
		key := responsesHistoryKey{credential: sha256.Sum256([]byte(c.GetHeader("Authorization")))}
		if req.PreviousResponseID != nil {
			key.id = *req.PreviousResponseID
			previous, found := h.get(key)
			if !found {
				c.AbortWithStatusJSON(http.StatusNotFound, gin.H{"error": gin.H{
					"type": "invalid_request_error", "code": "previous_response_not_found", "param": "previous_response_id",
					"message": "previous_response_id was not found or has expired; resend the full conversation",
				}})
				return
			}
			var history []json.RawMessage
			if err := json.Unmarshal(previous, &history); err != nil {
				c.AbortWithStatusJSON(http.StatusInternalServerError, openai.NewError(http.StatusInternalServerError, "invalid stored response"))
				return
			}
			input = append(history, input...)
		}
		// Only input/output items carry forward. In particular, instructions and
		// generation options belong to the current request, as in the Responses API.
		fields["input"], _ = json.Marshal(input)
		delete(fields, "previous_response_id")
		// Ollama owns this history; do not additionally persist it upstream.
		fields["store"] = json.RawMessage("false")
		body, err = json.Marshal(fields)
		if err != nil || len(body) > maxDecompressedBodySize {
			c.AbortWithStatusJSON(http.StatusRequestEntityTooLarge, openai.NewError(http.StatusRequestEntityTooLarge, "continued conversation is too large; compact it or resend a shorter history"))
			return
		}
		c.Request.Body = io.NopCloser(bytes.NewReader(body))
		c.Request.ContentLength = int64(len(body))
		c.Request.Header.Del("Content-Length")
		// Inspectable JSON/SSE is required to assign IDs and record completion.
		c.Request.Header.Set("Accept-Encoding", "identity")
		key.id = "resp_" + uuid.NewString()
		writer := &responsesHistoryWriter{
			ResponseWriter: c.Writer, history: h, key: key, input: input,
			previous: req.PreviousResponseID, store: req.Store == nil || *req.Store, stream: req.Stream,
		}
		c.Writer = writer
		c.Next()
		if err := writer.finish(); err != nil {
			// Do not advertise a successful stored response if the upstream ends
			// early or its response is not a valid Responses API envelope.
			_ = c.Error(err)
			writer.fail(err)
		}
	}
}

func responsesInputItems(raw json.RawMessage) ([]json.RawMessage, error) {
	if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return []json.RawMessage{}, nil
	}
	var text string
	if json.Unmarshal(raw, &text) == nil {
		item, _ := json.Marshal(map[string]any{"type": "message", "role": "user", "content": text})
		return []json.RawMessage{item}, nil
	}
	var items []json.RawMessage
	if err := json.Unmarshal(raw, &items); err != nil {
		return nil, fmt.Errorf("input must be a string or array")
	}
	return items, nil
}

type responsesHistoryWriter struct {
	gin.ResponseWriter
	history       *responsesHistory
	key           responsesHistoryKey
	input         []json.RawMessage
	previous      *string
	store, stream bool
	terminal      bool
	pending       bytes.Buffer
	err           error
}

func (w *responsesHistoryWriter) WriteHeader(code int) {
	w.Header().Del("Content-Length")
	w.ResponseWriter.WriteHeader(code)
}

func (w *responsesHistoryWriter) WriteString(s string) (int, error) { return w.Write([]byte(s)) }

func (w *responsesHistoryWriter) Write(data []byte) (int, error) {
	if w.Status() != http.StatusOK {
		return w.ResponseWriter.Write(data)
	}
	if w.err != nil {
		return 0, w.err
	}
	if !w.stream && w.pending.Len()+len(data) > maxDecompressedBodySize {
		w.err = fmt.Errorf("Responses output exceeds %d bytes", maxDecompressedBodySize)
		return 0, w.err
	}
	w.Header().Del("Content-Length")
	w.pending.Write(data)
	if w.stream {
		for {
			b := w.pending.Bytes()
			n, delimiter := bytes.Index(b, []byte("\n\n")), 2
			if crlf := bytes.Index(b, []byte("\r\n\r\n")); crlf >= 0 && (n < 0 || crlf < n) {
				n, delimiter = crlf, 4
			}
			if n < 0 {
				break
			}
			if n > maxDecompressedBodySize {
				w.err = fmt.Errorf("Responses event exceeds %d bytes", maxDecompressedBodySize)
				return 0, w.err
			}
			frame := bytes.Clone(w.pending.Next(n + delimiter))
			if err := w.writeEvent(frame); err != nil {
				w.err = err
				return 0, err
			}
		}
	}
	if w.pending.Len() > maxDecompressedBodySize {
		w.err = fmt.Errorf("Responses output exceeds %d bytes", maxDecompressedBodySize)
		return 0, w.err
	}
	return len(data), nil
}

func (w *responsesHistoryWriter) Flush() {
	if w.stream || w.Status() != http.StatusOK {
		w.ResponseWriter.Flush()
	}
}

func (w *responsesHistoryWriter) WriteHeaderNow() {
	if w.stream || w.Status() != http.StatusOK {
		w.ResponseWriter.WriteHeaderNow()
	}
}

func (w *responsesHistoryWriter) finish() error {
	if w.err != nil {
		return w.err
	}
	if w.Status() != http.StatusOK {
		return nil
	}
	if w.stream {
		// A partial SSE event is not a completed response and must never be cached.
		if len(bytes.TrimSpace(w.pending.Bytes())) != 0 || !w.terminal {
			return fmt.Errorf("Responses stream ended before a complete terminal event")
		}
		return nil
	}
	data, err := w.rewriteResponse(w.pending.Bytes(), true)
	if err != nil {
		return err
	}
	_, err = w.ResponseWriter.Write(data)
	return err
}

func (w *responsesHistoryWriter) fail(err error) {
	w.Header().Del("Content-Length")
	if !w.Written() {
		w.Header().Set("Content-Type", "application/json")
		w.ResponseWriter.WriteHeader(http.StatusBadGateway)
		_ = json.NewEncoder(w.ResponseWriter).Encode(openai.NewError(http.StatusBadGateway, err.Error()))
		return
	}
	if w.stream {
		data, _ := json.Marshal(map[string]any{"type": "error", "code": "invalid_response", "message": err.Error()})
		_, _ = fmt.Fprintf(w.ResponseWriter, "event: error\ndata: %s\n\n", data)
		w.ResponseWriter.Flush()
	}
}

func (w *responsesHistoryWriter) rewriteResponse(data []byte, terminal bool) ([]byte, error) {
	var response map[string]json.RawMessage
	if err := json.Unmarshal(data, &response); err != nil {
		return nil, err
	}
	if response == nil {
		return nil, fmt.Errorf("missing Responses response")
	}
	response["id"], _ = json.Marshal(w.key.id)
	response["previous_response_id"], _ = json.Marshal(w.previous)
	stored := w.store
	if terminal {
		var status string
		var output []json.RawMessage
		_ = json.Unmarshal(response["status"], &status)
		if status != "completed" && status != "incomplete" {
			stored = false
		}
		if err := json.Unmarshal(response["output"], &output); err != nil {
			stored = false
		}
		if stored {
			items := make([]json.RawMessage, 0, len(w.input)+len(output))
			// A compaction result replaces its transcript. Replaying the old
			// compaction_trigger would compact again on every continuation.
			var compacted bool
			if len(output) == 1 {
				var item struct {
					Type string `json:"type"`
				}
				_ = json.Unmarshal(output[0], &item)
				compacted = item.Type == "compaction"
			}
			if !compacted {
				items = append(items, w.input...)
			}
			items = append(items, output...)
			history, err := json.Marshal(items)
			stored = err == nil && w.history.put(w.key, history)
		}
	}
	response["store"], _ = json.Marshal(stored)
	return json.Marshal(response)
}

func (w *responsesHistoryWriter) writeEvent(frame []byte) error {
	lines := strings.Split(strings.ReplaceAll(string(frame), "\r\n", "\n"), "\n")
	var data []string
	var other []string
	for _, line := range lines {
		if strings.HasPrefix(line, "data:") {
			data = append(data, strings.TrimPrefix(strings.TrimPrefix(line, "data:"), " "))
		} else if line != "" {
			other = append(other, line)
		}
	}
	payload := strings.Join(data, "\n")
	if len(data) == 0 || payload == "[DONE]" {
		_, err := w.ResponseWriter.Write(frame)
		return err
	}
	var event map[string]json.RawMessage
	if err := json.Unmarshal([]byte(payload), &event); err != nil {
		return err
	}
	var eventType string
	_ = json.Unmarshal(event["type"], &eventType)
	if response, ok := event["response"]; ok {
		terminal := eventType == "response.completed" || eventType == "response.incomplete" || eventType == "response.failed"
		rewritten, err := w.rewriteResponse(response, terminal)
		if err != nil {
			return err
		}
		event["response"] = rewritten
		w.terminal = w.terminal || terminal
	}
	if eventType == "error" {
		w.terminal = true
	}
	if _, ok := event["response_id"]; ok {
		event["response_id"], _ = json.Marshal(w.key.id)
	}
	encoded, err := json.Marshal(event)
	if err != nil {
		return err
	}
	other = append(other, "data: "+string(encoded), "", "")
	_, err = w.ResponseWriter.Write([]byte(strings.Join(other, "\n")))
	return err
}
