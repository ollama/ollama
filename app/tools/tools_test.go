//go:build windows || darwin

package tools

import (
	"context"
	"errors"
	"testing"
	"time"
)

var errMissingToolDeadline = errors.New("tool context has no deadline")

type deadlineCapturingTool struct {
	deadline time.Time
}

func (d *deadlineCapturingTool) Name() string {
	return "deadline_capture"
}

func (d *deadlineCapturingTool) Description() string {
	return "captures the execution context deadline"
}

func (d *deadlineCapturingTool) Schema() map[string]any {
	return map[string]any{}
}

func (d *deadlineCapturingTool) Execute(ctx context.Context, args map[string]any) (any, string, error) {
	deadline, ok := ctx.Deadline()
	if !ok {
		return nil, "", errMissingToolDeadline
	}
	d.deadline = deadline
	return nil, "", nil
}

func (d *deadlineCapturingTool) Prompt() string {
	return ""
}

func TestRegistryExecuteAddsDefaultToolDeadline(t *testing.T) {
	registry := NewRegistry()
	tool := &deadlineCapturingTool{}
	registry.Register(tool)

	start := time.Now()
	if _, _, err := registry.Execute(context.Background(), tool.Name(), nil); err != nil {
		t.Fatalf("Execute() error = %v", err)
	}

	if tool.deadline.IsZero() {
		t.Fatal("Execute() did not set a tool deadline")
	}

	if got := tool.deadline.Sub(start); got <= 0 || got > 61*time.Second {
		t.Fatalf("tool deadline = %v after start, want within default timeout", got)
	}
}

type blockingTool struct {
	release chan struct{}
}

func (b *blockingTool) Name() string {
	return "blocking"
}

func (b *blockingTool) Description() string {
	return "blocks until released"
}

func (b *blockingTool) Schema() map[string]any {
	return map[string]any{}
}

func (b *blockingTool) Execute(ctx context.Context, args map[string]any) (any, string, error) {
	<-b.release
	return nil, "", nil
}

func (b *blockingTool) Prompt() string {
	return ""
}

func TestRegistryExecuteReturnsWhenContextDeadlineExpires(t *testing.T) {
	registry := NewRegistry()
	tool := &blockingTool{release: make(chan struct{})}
	registry.Register(tool)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()

	errc := make(chan error, 1)
	go func() {
		_, _, err := registry.Execute(ctx, tool.Name(), nil)
		errc <- err
	}()

	select {
	case err := <-errc:
		close(tool.release)
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("Execute() error = %v, want %v", err, context.DeadlineExceeded)
		}
	case <-time.After(200 * time.Millisecond):
		close(tool.release)
		err := <-errc
		t.Fatalf("Execute() blocked until the tool returned, err = %v", err)
	}
}
