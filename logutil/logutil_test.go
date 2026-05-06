package logutil

import (
	"bytes"
	"log/slog"
	"strings"
	"testing"
)

func TestNewLoggerDefaultIsText(t *testing.T) {
	t.Setenv("OLLAMA_LOG_FORMAT", "")
	var buf bytes.Buffer
	logger := NewLogger(&buf, slog.LevelInfo)
	logger.Info("hello")
	out := buf.String()
	if strings.Contains(out, `"msg"`) {
		t.Errorf("expected text format but got JSON-like output: %s", out)
	}
	if !strings.Contains(out, "hello") {
		t.Errorf("expected message in output: %s", out)
	}
}

func TestNewLoggerJSONFormat(t *testing.T) {
	t.Setenv("OLLAMA_LOG_FORMAT", "json")
	var buf bytes.Buffer
	logger := NewLogger(&buf, slog.LevelInfo)
	logger.Info("hello")
	out := buf.String()
	if !strings.Contains(out, `"msg"`) {
		t.Errorf("expected JSON format but got: %s", out)
	}
	if !strings.Contains(out, "hello") {
		t.Errorf("expected message in output: %s", out)
	}
}

func TestNewLoggerJSONFormatCaseInsensitive(t *testing.T) {
	for _, val := range []string{"JSON", "Json", "jSoN"} {
		t.Run(val, func(t *testing.T) {
			t.Setenv("OLLAMA_LOG_FORMAT", val)
			var buf bytes.Buffer
			logger := NewLogger(&buf, slog.LevelInfo)
			logger.Info("test")
			if !strings.Contains(buf.String(), `"msg"`) {
				t.Errorf("OLLAMA_LOG_FORMAT=%q: expected JSON output, got: %s", val, buf.String())
			}
		})
	}
}

func TestNewLoggerUnknownFormatFallsBackToText(t *testing.T) {
	t.Setenv("OLLAMA_LOG_FORMAT", "yaml")
	var buf bytes.Buffer
	logger := NewLogger(&buf, slog.LevelInfo)
	logger.Info("hello")
	out := buf.String()
	if strings.Contains(out, `"msg"`) {
		t.Errorf("unknown format should fall back to text, got: %s", out)
	}
}
