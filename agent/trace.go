package agent

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode/utf8"
)

const agentTraceEnv = "OLLAMA_AGENT_TRACE"

type JSONLTraceSink struct {
	mu   sync.Mutex
	file *os.File
}

func NewJSONLTraceSinkFromEnv() (*JSONLTraceSink, error) {
	path := strings.TrimSpace(os.Getenv(agentTraceEnv))
	if path == "" {
		return nil, nil
	}
	return NewJSONLTraceSink(path)
}

func NewJSONLTraceSink(path string) (*JSONLTraceSink, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return nil, nil
	}
	if dir := filepath.Dir(path); dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return nil, err
		}
	}
	file, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, err
	}
	return &JSONLTraceSink{file: file}, nil
}

func (s *JSONLTraceSink) Close() error {
	if s == nil || s.file == nil {
		return nil
	}
	return s.file.Close()
}

func (s *JSONLTraceSink) Emit(event Event) error {
	if s == nil || s.file == nil {
		return nil
	}
	record := traceRecordFromEvent(event)
	data, err := json.Marshal(record)
	if err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	_, err = s.file.Write(append(data, '\n'))
	return err
}

type traceRecord struct {
	Time                      time.Time         `json:"time"`
	Type                      EventType         `json:"type"`
	RunID                     string            `json:"run_id,omitempty"`
	ChatID                    string            `json:"chat_id,omitempty"`
	Model                     string            `json:"model,omitempty"`
	Status                    string            `json:"status,omitempty"`
	ToolName                  string            `json:"tool_name,omitempty"`
	ToolCallID                string            `json:"tool_call_id,omitempty"`
	WorkingDir                string            `json:"working_dir,omitempty"`
	Args                      map[string]string `json:"args,omitempty"`
	PromptTokens              int               `json:"prompt_tokens,omitempty"`
	ContextWindowTokens       int               `json:"context_window_tokens,omitempty"`
	CompactionThresholdTokens int               `json:"compaction_threshold_tokens,omitempty"`
	MessageCount              int               `json:"message_count,omitempty"`
	UserMessageCount          int               `json:"user_message_count,omitempty"`
	AssistantMessageCount     int               `json:"assistant_message_count,omitempty"`
	ToolMessageCount          int               `json:"tool_message_count,omitempty"`
	SystemMessageCount        int               `json:"system_message_count,omitempty"`
	ToolCallCount             int               `json:"tool_call_count,omitempty"`
	ToolCount                 int               `json:"tool_count,omitempty"`
	ToolNames                 []string          `json:"tool_names,omitempty"`
	ToolRound                 int               `json:"tool_round,omitempty"`
	ToolRoundLimit            int               `json:"tool_round_limit,omitempty"`
	ContentApproxTokens       int               `json:"content_approx_tokens,omitempty"`
	ContentRunes              int               `json:"content_runes,omitempty"`
	ThinkingApproxTokens      int               `json:"thinking_approx_tokens,omitempty"`
	ThinkingRunes             int               `json:"thinking_runes,omitempty"`
	Response                  *traceMetrics     `json:"response,omitempty"`
	Error                     string            `json:"error,omitempty"`
}

type traceMetrics struct {
	TotalDurationMS      float64 `json:"total_duration_ms,omitempty"`
	LoadDurationMS       float64 `json:"load_duration_ms,omitempty"`
	PromptEvalCount      int     `json:"prompt_eval_count,omitempty"`
	PromptEvalDurationMS float64 `json:"prompt_eval_duration_ms,omitempty"`
	EvalCount            int     `json:"eval_count,omitempty"`
	EvalDurationMS       float64 `json:"eval_duration_ms,omitempty"`
	PromptEvalRate       float64 `json:"prompt_eval_rate,omitempty"`
	EvalRate             float64 `json:"eval_rate,omitempty"`
}

func traceRecordFromEvent(event Event) traceRecord {
	record := traceRecord{
		Time:                      time.Now(),
		Type:                      event.Type,
		RunID:                     event.RunID,
		ChatID:                    event.ChatID,
		Model:                     event.Model,
		Status:                    event.Status,
		ToolName:                  event.ToolName,
		ToolCallID:                event.ToolCallID,
		WorkingDir:                event.WorkingDir,
		Args:                      traceArgs(event.Args),
		PromptTokens:              event.PromptTokens,
		ContextWindowTokens:       event.ContextWindowTokens,
		CompactionThresholdTokens: event.CompactionThresholdTokens,
		MessageCount:              event.MessageCount,
		UserMessageCount:          event.UserMessageCount,
		AssistantMessageCount:     event.AssistantMessageCount,
		ToolMessageCount:          event.ToolMessageCount,
		SystemMessageCount:        event.SystemMessageCount,
		ToolCallCount:             event.ToolCallCount,
		ToolCount:                 event.ToolCount,
		ToolNames:                 event.ToolNames,
		ToolRound:                 event.ToolRound,
		ToolRoundLimit:            event.ToolRoundLimit,
		ContentRunes:              utf8.RuneCountInString(event.Content),
		ThinkingRunes:             utf8.RuneCountInString(event.Thinking),
		Error:                     event.Error,
	}
	if event.Content != "" {
		record.ContentApproxTokens = estimateCompactionTokens(event.Content)
	}
	if event.Thinking != "" {
		record.ThinkingApproxTokens = estimateCompactionTokens(event.Thinking)
	}
	if event.Response != nil {
		metrics := event.Response.Metrics
		record.Response = &traceMetrics{
			TotalDurationMS:      durationMS(metrics.TotalDuration),
			LoadDurationMS:       durationMS(metrics.LoadDuration),
			PromptEvalCount:      metrics.PromptEvalCount,
			PromptEvalDurationMS: durationMS(metrics.PromptEvalDuration),
			EvalCount:            metrics.EvalCount,
			EvalDurationMS:       durationMS(metrics.EvalDuration),
		}
		if metrics.PromptEvalCount > 0 && metrics.PromptEvalDuration > 0 {
			record.Response.PromptEvalRate = float64(metrics.PromptEvalCount) / metrics.PromptEvalDuration.Seconds()
		}
		if metrics.EvalCount > 0 && metrics.EvalDuration > 0 {
			record.Response.EvalRate = float64(metrics.EvalCount) / metrics.EvalDuration.Seconds()
		}
	}
	return record
}

func durationMS(d time.Duration) float64 {
	if d <= 0 {
		return 0
	}
	return float64(d) / float64(time.Millisecond)
}

func traceArgs(args map[string]any) map[string]string {
	if len(args) == 0 {
		return nil
	}
	keys := make([]string, 0, len(args))
	for key := range args {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	out := make(map[string]string, len(keys))
	for _, key := range keys {
		out[key] = traceString(args[key])
	}
	return out
}

func traceString(value any) string {
	data, err := json.Marshal(value)
	if err != nil {
		data = []byte(strings.TrimSpace(strings.ReplaceAll(strings.ReplaceAll(strings.TrimSpace(toString(value)), "\n", " "), "\t", " ")))
	}
	text := strings.TrimSpace(string(data))
	runes := []rune(text)
	if len(runes) > 240 {
		return string(runes[:240]) + "..."
	}
	return text
}

func toString(value any) string {
	if value == nil {
		return ""
	}
	if text, ok := value.(string); ok {
		return text
	}
	data, _ := json.Marshal(value)
	return string(data)
}
