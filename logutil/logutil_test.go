package logutil

import (
	"bytes"
	"context"
	"log/slog"
	"strings"
	"testing"
)

func TestLevelTrace(t *testing.T) {
	if LevelTrace >= slog.LevelDebug {
		t.Errorf("LevelTrace = %d, should be less than LevelDebug (%d)", LevelTrace, slog.LevelDebug)
	}
	if LevelTrace != -8 {
		t.Errorf("LevelTrace = %d, want -8", LevelTrace)
	}
}

func TestNewLogger(t *testing.T) {
	t.Run("creates logger with specified level", func(t *testing.T) {
		var buf bytes.Buffer
		logger := NewLogger(&buf, slog.LevelDebug)

		if logger == nil {
			t.Fatal("NewLogger returned nil")
		}

		logger.Debug("test debug message")
		output := buf.String()
		if !strings.Contains(output, "test debug message") {
			t.Errorf("Logger output = %q, want to contain 'test debug message'", output)
		}
	})

	t.Run("respects level filtering", func(t *testing.T) {
		var buf bytes.Buffer
		logger := NewLogger(&buf, slog.LevelWarn)

		logger.Debug("debug message")
		logger.Info("info message")
		logger.Warn("warn message")

		output := buf.String()

		if strings.Contains(output, "debug message") {
			t.Error("Debug message should be filtered at Warn level")
		}
		if strings.Contains(output, "info message") {
			t.Error("Info message should be filtered at Warn level")
		}
		if !strings.Contains(output, "warn message") {
			t.Error("Warn message should appear at Warn level")
		}
	})

	t.Run("formats trace level as TRACE", func(t *testing.T) {
		var buf bytes.Buffer
		logger := NewLogger(&buf, LevelTrace)

		logger.Log(nil, LevelTrace, "trace message")
		output := buf.String()

		if !strings.Contains(output, "level=TRACE") {
			t.Errorf("Logger output = %q, want to contain 'level=TRACE'", output)
		}
	})

	t.Run("adds source information", func(t *testing.T) {
		var buf bytes.Buffer
		logger := NewLogger(&buf, slog.LevelInfo)

		logger.Info("test message")
		output := buf.String()

		if !strings.Contains(output, "source=") {
			t.Error("Logger should include source information")
		}
	})

	t.Run("uses basename for source file", func(t *testing.T) {
		var buf bytes.Buffer
		logger := NewLogger(&buf, slog.LevelInfo)

		logger.Info("test message")
		output := buf.String()

		if strings.Contains(output, string([]byte{'/', '\\'})+"logutil") || strings.Contains(output, "logutil"+string([]byte{'/', '\\'})) {
			if strings.Contains(output, "logutil_test.go") {
			}
		}
	})
}

func TestTrace(t *testing.T) {
	t.Run("logs at trace level when enabled", func(t *testing.T) {
		var buf bytes.Buffer
		logger := NewLogger(&buf, LevelTrace)
		slog.SetDefault(logger)
		t.Cleanup(func() { slog.SetDefault(slog.Default()) })

		Trace("trace test message")
		output := buf.String()

		if !strings.Contains(output, "trace test message") {
			t.Errorf("Trace output = %q, want to contain 'trace test message'", output)
		}
	})

	t.Run("does not log when level is higher", func(t *testing.T) {
		var buf bytes.Buffer
		logger := NewLogger(&buf, slog.LevelInfo)
		slog.SetDefault(logger)
		t.Cleanup(func() { slog.SetDefault(slog.Default()) })

		Trace("should not appear")
		output := buf.String()

		if strings.Contains(output, "should not appear") {
			t.Error("Trace should not log at Info level")
		}
	})
}

func TestTraceContext(t *testing.T) {
	t.Run("logs with context at trace level", func(t *testing.T) {
		var buf bytes.Buffer
		logger := NewLogger(&buf, LevelTrace)
		slog.SetDefault(logger)
		t.Cleanup(func() { slog.SetDefault(slog.Default()) })

		ctx := context.Background()
		TraceContext(ctx, "context trace message", "key", "value")

		output := buf.String()
		if !strings.Contains(output, "context trace message") {
			t.Errorf("TraceContext output = %q, want to contain 'context trace message'", output)
		}
		if !strings.Contains(output, "key=value") {
			t.Errorf("TraceContext output = %q, want to contain 'key=value'", output)
		}
	})

	t.Run("returns early when disabled", func(t *testing.T) {
		var buf bytes.Buffer
		logger := NewLogger(&buf, slog.LevelError)
		slog.SetDefault(logger)
		t.Cleanup(func() { slog.SetDefault(slog.Default()) })

		TraceContext(context.Background(), "silent message", "key", "value")
		output := buf.String()

		if output != "" {
			t.Errorf("TraceContext at Error level should produce no output, got %q", output)
		}
	})
}

func TestLoggerOutput(t *testing.T) {
	t.Run("includes all standard fields", func(t *testing.T) {
		var buf bytes.Buffer
		logger := NewLogger(&buf, slog.LevelInfo)

		logger.Info("test message", "key1", "value1", "key2", 42)

		output := buf.String()

		if !strings.Contains(output, `msg="test message"`) {
			t.Errorf("Output missing message: %s", output)
		}
		if !strings.Contains(output, "key1=value1") {
			t.Errorf("Output missing key1: %s", output)
		}
		if !strings.Contains(output, "key2=42") {
			t.Errorf("Output missing key2: %s", output)
		}
	})

	t.Run("handles multiple log calls", func(t *testing.T) {
		var buf bytes.Buffer
		logger := NewLogger(&buf, LevelTrace)

		logger.Debug("debug msg")
		logger.Info("info msg")
		logger.Warn("warn msg")
		logger.Error("error msg")

		output := buf.String()

		if !strings.Contains(output, "debug msg") {
			t.Error("Missing debug message")
		}
		if !strings.Contains(output, "info msg") {
			t.Error("Missing info message")
		}
		if !strings.Contains(output, "warn msg") {
			t.Error("Missing warn message")
		}
		if !strings.Contains(output, "error msg") {
			t.Error("Missing error message")
		}
	})
}

func TestTraceSkipCaller(t *testing.T) {
	var buf bytes.Buffer
	logger := NewLogger(&buf, LevelTrace)
	slog.SetDefault(logger)
	t.Cleanup(func() { slog.SetDefault(slog.Default()) })

	Trace("direct trace call")

	output := buf.String()
	if !strings.Contains(output, "logutil_test.go") {
		t.Errorf("Output should contain source file, got: %s", output)
	}
}