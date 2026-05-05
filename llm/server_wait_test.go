package llm

import (
	"context"
	"strings"
	"testing"
)

func TestWaitUntilRunningUsesStatusMessageWhenDoneErrIsNil(t *testing.T) {
	done := make(chan struct{})
	close(done)

	status := &StatusWriter{}
	status.SetLastError("llama_init_from_model: failed to initialize the context: failed to initialize Metal backend")

	s := &llmServer{
		done:   done,
		status: status,
	}

	err := s.WaitUntilRunning(context.Background())
	if err == nil {
		t.Fatal("expected error")
	}
	if strings.Contains(err.Error(), "%!w(<nil>)") {
		t.Fatalf("unexpected wrapped nil error: %q", err)
	}
	if !strings.Contains(err.Error(), s.status.LastError()) {
		t.Fatalf("error %q does not include status message %q", err, s.status.LastError())
	}
}
