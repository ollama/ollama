package server

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
)

type metricsLogEntry struct {
	Timestamp         time.Time `json:"timestamp"`
	Model             string    `json:"model"`
	RequestType       string    `json:"request_type"`
	PromptEvalCount   int       `json:"prompt_eval_count"`
	PromptEvalCached  int       `json:"prompt_eval_cached"`
	EvalCount         int       `json:"eval_count"`
	TotalDurationMs   float64   `json:"total_duration_ms"`
	LoadDurationMs    float64   `json:"load_duration_ms"`
	PromptEvalMs      float64   `json:"prompt_eval_duration_ms"`
	EvalDurationMs    float64   `json:"eval_duration_ms"`
	PromptEvalRate    float64   `json:"prompt_eval_tokens_per_sec"`
	EvalRate          float64   `json:"eval_tokens_per_sec"`
}

type metricsLogger struct {
	file *os.File
	mu   sync.Mutex
}

func newMetricsLogger() (*metricsLogger, error) {
	logPath := envconfig.MetricsLogFile()
	if logPath == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return nil, fmt.Errorf("metrics logger: %w", err)
		}
		logPath = filepath.Join(home, ".ollama", "logs", "metrics.jsonl")
	}

	if err := os.MkdirAll(filepath.Dir(logPath), 0o755); err != nil {
		return nil, fmt.Errorf("metrics logger: create directory: %w", err)
	}

	f, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return nil, fmt.Errorf("metrics logger: open file: %w", err)
	}

	slog.Info(fmt.Sprintf("metrics logging enabled; writing to %s", logPath))
	return &metricsLogger{file: f}, nil
}

func (ml *metricsLogger) Log(model string, requestType string, m api.Metrics) {
	if ml == nil {
		return
	}

	cached := 0
	if m.PromptEvalCachedCount != nil {
		cached = *m.PromptEvalCachedCount
	}

	var promptEvalRate, evalRate float64
	if m.PromptEvalDuration > 0 {
		uncached := max(0, m.PromptEvalCount-cached)
		promptEvalRate = float64(uncached) / m.PromptEvalDuration.Seconds()
	}
	if m.EvalDuration > 0 {
		evalRate = float64(m.EvalCount) / m.EvalDuration.Seconds()
	}

	entry := metricsLogEntry{
		Timestamp:        time.Now().UTC(),
		Model:            model,
		RequestType:      requestType,
		PromptEvalCount:  m.PromptEvalCount,
		PromptEvalCached: cached,
		EvalCount:        m.EvalCount,
		TotalDurationMs:  float64(m.TotalDuration) / float64(time.Millisecond),
		LoadDurationMs:   float64(m.LoadDuration) / float64(time.Millisecond),
		PromptEvalMs:     float64(m.PromptEvalDuration) / float64(time.Millisecond),
		EvalDurationMs:   float64(m.EvalDuration) / float64(time.Millisecond),
		PromptEvalRate:   promptEvalRate,
		EvalRate:         evalRate,
	}

	data, err := json.Marshal(entry)
	if err != nil {
		slog.Warn("metrics logger: marshal error", "error", err)
		return
	}

	ml.mu.Lock()
	defer ml.mu.Unlock()

	data = append(data, '\n')
	if _, err := ml.file.Write(data); err != nil {
		slog.Warn("metrics logger: write error", "error", err)
	}
}

func (ml *metricsLogger) Close() error {
	if ml == nil || ml.file == nil {
		return nil
	}
	return ml.file.Close()
}

func (s *Server) initMetricsLogging() error {
	if !envconfig.LogMetrics() {
		return nil
	}

	ml, err := newMetricsLogger()
	if err != nil {
		return fmt.Errorf("enable OLLAMA_LOG_METRICS: %w", err)
	}

	s.metricsLogger = ml
	return nil
}
