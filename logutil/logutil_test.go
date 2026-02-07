package logutil

import (
	"bytes"
	"encoding/json"
	"log/slog"
	"os"
	"strings"
	"testing"
)

func TestNewLogger_DefaultTextFormat(t *testing.T) {
	// Ensure OLLAMA_LOG_FORMAT is not set
	os.Unsetenv("OLLAMA_LOG_FORMAT")

	var buf bytes.Buffer
	logger := NewLogger(&buf, slog.LevelInfo)

	logger.Info("test message", "key", "value")

	output := buf.String()
	// Text format should contain key=value pairs
	if !strings.Contains(output, "key=value") {
		t.Errorf("Expected text format with key=value, got: %s", output)
	}
	// Text format should not be valid JSON
	var jsonData map[string]interface{}
	if err := json.Unmarshal(buf.Bytes(), &jsonData); err == nil {
		t.Error("Expected text format (not JSON), but output was valid JSON")
	}
}

func TestNewLogger_JSONFormat(t *testing.T) {
	// Set OLLAMA_LOG_FORMAT to json
	os.Setenv("OLLAMA_LOG_FORMAT", "json")
	defer os.Unsetenv("OLLAMA_LOG_FORMAT")

	var buf bytes.Buffer
	logger := NewLogger(&buf, slog.LevelInfo)

	logger.Info("test message", "key", "value")

	output := buf.String()
	// JSON format should be valid JSON
	var jsonData map[string]interface{}
	if err := json.Unmarshal([]byte(strings.TrimSpace(output)), &jsonData); err != nil {
		t.Errorf("Expected valid JSON output, got error: %v, output: %s", err, output)
	}

	// Verify JSON contains expected fields
	if jsonData["msg"] != "test message" {
		t.Errorf("Expected msg='test message', got: %v", jsonData["msg"])
	}
	if jsonData["key"] != "value" {
		t.Errorf("Expected key='value', got: %v", jsonData["key"])
	}
	if jsonData["level"] == nil {
		t.Error("Expected 'level' field in JSON output")
	}
}

func TestNewLogger_JSONFormatCaseInsensitive(t *testing.T) {
	testCases := []string{"JSON", "Json", "json", "jSoN"}

	for _, format := range testCases {
		t.Run(format, func(t *testing.T) {
			os.Setenv("OLLAMA_LOG_FORMAT", format)
			defer os.Unsetenv("OLLAMA_LOG_FORMAT")

			var buf bytes.Buffer
			logger := NewLogger(&buf, slog.LevelInfo)

			logger.Info("test message")

			output := buf.String()
			var jsonData map[string]interface{}
			if err := json.Unmarshal([]byte(strings.TrimSpace(output)), &jsonData); err != nil {
				t.Errorf("Format '%s' should produce valid JSON, got error: %v", format, err)
			}
		})
	}
}

func TestNewLogger_InvalidFormatDefaultsToText(t *testing.T) {
	os.Setenv("OLLAMA_LOG_FORMAT", "invalid")
	defer os.Unsetenv("OLLAMA_LOG_FORMAT")

	var buf bytes.Buffer
	logger := NewLogger(&buf, slog.LevelInfo)

	logger.Info("test message", "key", "value")

	output := buf.String()
	// Should default to text format
	if !strings.Contains(output, "key=value") {
		t.Errorf("Expected text format with key=value, got: %s", output)
	}
}

func TestNewLogger_TraceLevel(t *testing.T) {
	os.Setenv("OLLAMA_LOG_FORMAT", "json")
	defer os.Unsetenv("OLLAMA_LOG_FORMAT")

	var buf bytes.Buffer
	logger := NewLogger(&buf, LevelTrace)
	slog.SetDefault(logger)

	Trace("trace message", "trace_key", "trace_value")

	output := buf.String()
	var jsonData map[string]interface{}
	if err := json.Unmarshal([]byte(strings.TrimSpace(output)), &jsonData); err != nil {
		t.Errorf("Expected valid JSON output for trace, got error: %v", err)
	}

	if jsonData["level"] != "TRACE" {
		t.Errorf("Expected level='TRACE', got: %v", jsonData["level"])
	}
	if jsonData["msg"] != "trace message" {
		t.Errorf("Expected msg='trace message', got: %v", jsonData["msg"])
	}
}

func TestNewLogger_SourceField(t *testing.T) {
	os.Setenv("OLLAMA_LOG_FORMAT", "json")
	defer os.Unsetenv("OLLAMA_LOG_FORMAT")

	var buf bytes.Buffer
	logger := NewLogger(&buf, slog.LevelInfo)

	logger.Info("test message")

	output := buf.String()
	var jsonData map[string]interface{}
	if err := json.Unmarshal([]byte(strings.TrimSpace(output)), &jsonData); err != nil {
		t.Errorf("Expected valid JSON output, got error: %v", err)
	}

	// Verify source field exists (AddSource is true)
	if jsonData["source"] == nil {
		t.Error("Expected 'source' field in JSON output")
	}

	// Verify source file is basename only (not full path)
	source, ok := jsonData["source"].(map[string]interface{})
	if !ok {
		t.Error("Expected 'source' to be an object")
	}
	file, ok := source["file"].(string)
	if !ok {
		t.Error("Expected 'source.file' to be a string")
	}
	if strings.Contains(file, "/") {
		t.Errorf("Expected basename only in source.file, got: %s", file)
	}
}
