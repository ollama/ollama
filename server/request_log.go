package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
)

// requestLog holds the shared per-slot log file.
// Each concurrent model runner slot gets one file that accumulates all
// requests handled by that slot in sequence. With a single parallel slot
// (the common case) there is exactly one file: requests_0.log.
type requestLog struct {
	mu sync.Mutex
	f  *os.File
}

var (
	requestLogSlots []*requestLog
	requestLogsMu   sync.Mutex
)

func requestLogDir() (string, error) {
	dir := os.Getenv("OLLAMA_REQUEST_LOG_DIR")
	if dir == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", err
		}
		dir = filepath.Join(home, ".ollama", "logs")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	return dir, nil
}

func acquireRequestLogSlot() (*requestLog, error) {
	requestLogsMu.Lock()
	defer requestLogsMu.Unlock()

	// find a slot that is not currently locked (i.e. free)
	for _, slot := range requestLogSlots {
		if slot.mu.TryLock() {
			return slot, nil
		}
	}

	// all existing slots are busy — open a new file
	dir, err := requestLogDir()
	if err != nil {
		return nil, err
	}
	idx := len(requestLogSlots)
	name := filepath.Join(dir, fmt.Sprintf("ollama_requests-%d.log", idx))
	f, err := os.OpenFile(name, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, err
	}
	slot := &requestLog{f: f}
	slot.mu.Lock()
	requestLogSlots = append(requestLogSlots, slot)
	slog.Info("log_request: opened slot log", "path", name)
	return slot, nil
}

// requestLogger is the per-request handle; it holds the slot lock until close.
type requestLogger struct {
	slot *requestLog
}

// resolveLogRequest returns true when request logging should be active.
// Explicit per-request value wins; nil falls back to OLLAMA_REQUEST_LOG_ENABLED.
// Note: upstream uses OLLAMA_DEBUG_LOG_REQUESTS (server/inference_request_log.go) for
// request-body-only logging with curl replays. This feature adds response streaming,
// thinking, done-metrics, and per-request control — they coexist without conflict.
func resolveLogRequest(v *bool) bool {
	if v != nil {
		return *v
	}
	return os.Getenv("OLLAMA_REQUEST_LOG_ENABLED") != ""
}

func openRequestLog() (*requestLogger, error) {
	slot, err := acquireRequestLogSlot()
	if err != nil {
		return nil, err
	}
	return &requestLogger{slot: slot}, nil
}

// requestMeta carries the parts of a request that are not in the prompt text but
// decide how the model answers: the tools it was offered, a schema that constrains
// its output, the thinking level, and how many images came along. Without them a
// tool-call bug is not diagnosable from the log — a model that ignored a tool and a
// model that was never offered one look exactly alike.
type requestMeta struct {
	Tools  api.Tools
	Format json.RawMessage
	Think  *api.ThinkValue
	Images int
}

func (l *requestLogger) writeRequestHeader(model string, options map[string]any, meta requestMeta, body string) {
	sep := strings.Repeat("=", 80)
	ts := time.Now().Format(time.RFC3339)
	fmt.Fprintf(l.slot.f, "\n%s\nREQUEST %s  model=%s\n%s\n", sep, ts, model, sep)
	if len(options) > 0 {
		fmt.Fprintf(l.slot.f, "OPTIONS:\n")
		for k, v := range options {
			enc, _ := json.Marshal(v)
			fmt.Fprintf(l.slot.f, "  %s: %s\n", k, enc)
		}
		fmt.Fprintln(l.slot.f)
	}
	// The offered tools in short form, not as full JSON schema: they are identical in
	// every request of a session, and name plus parameters answer the question one
	// actually has here. Required parameters carry a trailing *.
	if len(meta.Tools) > 0 {
		fmt.Fprintf(l.slot.f, "TOOLS (%d):\n", len(meta.Tools))
		for _, t := range meta.Tools {
			fmt.Fprintf(l.slot.f, "  %s\n", formatToolSignature(t))
		}
		fmt.Fprintln(l.slot.f)
	}
	if len(meta.Format) > 0 {
		var compact bytes.Buffer
		if err := json.Compact(&compact, meta.Format); err != nil {
			compact.Write(meta.Format)
		}
		fmt.Fprintf(l.slot.f, "FORMAT: %s\n\n", compact.String())
	}
	if meta.Think != nil && meta.Think.Value != nil {
		enc, _ := json.Marshal(meta.Think.Value)
		fmt.Fprintf(l.slot.f, "THINK: %s\n\n", enc)
	}
	if meta.Images > 0 {
		fmt.Fprintf(l.slot.f, "IMAGES: %d\n\n", meta.Images)
	}
	fmt.Fprintf(l.slot.f, "PROMPT/MESSAGES:\n%s\n\n", body)
	fmt.Fprintf(l.slot.f, "%s\nRESPONSE %s  model=%s\n%s\n", sep, ts, model, sep)
}

// formatToolSignature renders one offered tool as name(param, param*) — one line,
// required parameters marked with *.
func formatToolSignature(t api.Tool) string {
	var sb strings.Builder
	sb.WriteString(t.Function.Name)
	sb.WriteByte('(')
	required := make(map[string]bool, len(t.Function.Parameters.Required))
	for _, r := range t.Function.Parameters.Required {
		required[r] = true
	}
	first := true
	for name, prop := range t.Function.Parameters.Properties.All() {
		if !first {
			sb.WriteString(", ")
		}
		first = false
		sb.WriteString(name)
		if len(prop.Type) > 0 {
			fmt.Fprintf(&sb, " %s", strings.Join(prop.Type, "|"))
		}
		if required[name] {
			sb.WriteByte('*')
		}
	}
	sb.WriteByte(')')
	return sb.String()
}

func (l *requestLogger) writeChunk(content string) {
	fmt.Fprint(l.slot.f, content)
}

// writeToolCalls records the tool calls of a response. They do not come through
// writeChunk, because they are not text: the parser (or llama-server itself, in the
// new engine) turns them into a structure, and that structure never passed this log.
// A response that consisted only of tool calls therefore used to be logged empty.
func (l *requestLogger) writeToolCalls(toolCalls []api.ToolCall) {
	if len(toolCalls) == 0 {
		return
	}
	fmt.Fprintf(l.slot.f, "\n\nTOOL CALLS (%d):\n", len(toolCalls))
	for i, tc := range toolCalls {
		args := tc.Function.Arguments.String()
		fmt.Fprintf(l.slot.f, "  [%d] %s(%s)", i, tc.Function.Name, args)
		if tc.ID != "" {
			fmt.Fprintf(l.slot.f, "  id=%s", tc.ID)
		}
		fmt.Fprintln(l.slot.f)
	}
}

// writeError closes a request that did not finish. Without it the log just stops
// mid-sentence, and a broken-off request looks exactly like one still running.
func (l *requestLogger) writeError(err error) {
	if err == nil {
		return
	}
	sep := strings.Repeat("=", 80)
	fmt.Fprintf(l.slot.f, "\n%s\nERROR %s\n%s\n  %s\n", sep, time.Now().Format(time.RFC3339), sep, err.Error())
}

// formatRequestMessages renders the incoming message list for the new engine, which
// has no rendered template to log. The first line of each message keeps the
// "[role]: text" shape that ollama_server.sh's trace --short filters on; everything
// that used to fall off the table is indented underneath it.
func formatRequestMessages(msgs []api.Message) string {
	var sb strings.Builder
	for _, msg := range msgs {
		fmt.Fprintf(&sb, "[%s]: %s\n", msg.Role, msg.Content)
		if msg.ToolName != "" {
			fmt.Fprintf(&sb, "   tool_name: %s\n", msg.ToolName)
		}
		if msg.ToolCallID != "" {
			fmt.Fprintf(&sb, "   tool_call_id: %s\n", msg.ToolCallID)
		}
		if msg.Thinking != "" {
			fmt.Fprintf(&sb, "   thinking: %s\n", msg.Thinking)
		}
		// The tool calls of an earlier turn, as they are fed back in. Only with these
		// does one see what the model had already produced before the tool answered.
		for i, tc := range msg.ToolCalls {
			fmt.Fprintf(&sb, "   tool_call[%d]: %s(%s)", i, tc.Function.Name, tc.Function.Arguments.String())
			if tc.ID != "" {
				fmt.Fprintf(&sb, "  id=%s", tc.ID)
			}
			sb.WriteByte('\n')
		}
		if len(msg.Images) > 0 {
			fmt.Fprintf(&sb, "   images: %d\n", len(msg.Images))
		}
	}
	return sb.String()
}

func (l *requestLogger) writeDone(ts time.Time, doneReason string, metrics api.Metrics) {
	sep := strings.Repeat("=", 80)
	fmt.Fprintf(l.slot.f, "\n%s\nDONE %s  reason=%s\n%s\n", sep, ts.Format(time.RFC3339), doneReason, sep)
	fmt.Fprintf(l.slot.f, "  prompt_eval_count:    %d tokens\n", metrics.PromptEvalCount)
	fmt.Fprintf(l.slot.f, "  prompt_eval_duration: %s\n", metrics.PromptEvalDuration.Round(time.Millisecond))
	fmt.Fprintf(l.slot.f, "  eval_count:           %d tokens\n", metrics.EvalCount)
	fmt.Fprintf(l.slot.f, "  eval_duration:        %s\n", metrics.EvalDuration.Round(time.Millisecond))
	if metrics.EvalDuration > 0 {
		tps := float64(metrics.EvalCount) / metrics.EvalDuration.Seconds()
		fmt.Fprintf(l.slot.f, "  tokens/s:             %.1f\n", tps)
	}
	fmt.Fprintf(l.slot.f, "  load_duration:        %s\n", metrics.LoadDuration.Round(time.Millisecond))
	fmt.Fprintf(l.slot.f, "  total_duration:       %s\n", metrics.TotalDuration.Round(time.Millisecond))
}

func (l *requestLogger) close() {
	if l == nil || l.slot == nil {
		return
	}
	l.slot.mu.Unlock()
}
