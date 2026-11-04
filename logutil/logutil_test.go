package logutil

import (
	"bytes"
	"log/slog"
	"strings"
	"testing"
)

func TestNewLoggerAllowsSourceAttr(t *testing.T) {
	var buf bytes.Buffer
	logger := NewLogger(&buf, slog.LevelDebug)

	logger.Debug("message", "source", "runtime")

	if !strings.Contains(buf.String(), `source=runtime`) {
		t.Fatalf("expected user source attr in log, got %q", buf.String())
	}
}
