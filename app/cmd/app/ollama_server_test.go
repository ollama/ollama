//go:build windows || darwin

package main

import (
	"context"
	"sync"
	"testing"
	"time"
)

// blockingServer runs until its context is canceled and records each run.
type blockingServer struct {
	mu      sync.Mutex
	started []context.Context
	running int
	stopped chan struct{}
}

func (s *blockingServer) Run(ctx context.Context) error {
	s.mu.Lock()
	s.started = append(s.started, ctx)
	s.running++
	s.mu.Unlock()

	<-ctx.Done()

	s.mu.Lock()
	s.running--
	s.mu.Unlock()
	s.stopped <- struct{}{}
	return ctx.Err()
}

func (s *blockingServer) runs() (started, running int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.started), s.running
}

func waitForRuns(t *testing.T, s *blockingServer, started int) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if got, _ := s.runs(); got >= started {
			return
		}
		time.Sleep(time.Millisecond)
	}
	got, _ := s.runs()
	t.Fatalf("server started %d times, want %d", got, started)
}

func TestOllamaServerRestart(t *testing.T) {
	runner := &blockingServer{stopped: make(chan struct{}, 4)}
	server := &ollamaServer{server: runner}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()

	server.Start(ctx)
	waitForRuns(t, runner, 1)

	// Restart returns only after the old server stops.
	server.Restart()
	select {
	case <-runner.stopped:
	default:
		t.Fatal("restart returned before the old server stopped")
	}
	waitForRuns(t, runner, 2)
	if _, running := runner.runs(); running != 1 {
		t.Fatalf("running servers = %d, want 1", running)
	}

	// Once the app is quitting, a restart must not start another server.
	cancel()
	server.Wait()
	server.Restart()
	if started, running := runner.runs(); started != 2 || running != 0 {
		t.Fatalf("after quitting: started = %d, running = %d", started, running)
	}
}

func TestOllamaServerRestartBeforeStart(t *testing.T) {
	runner := &blockingServer{stopped: make(chan struct{}, 1)}
	server := &ollamaServer{server: runner}
	server.Restart()
	server.Wait()
	if started, _ := runner.runs(); started != 0 {
		t.Fatalf("restart before start ran the server %d times", started)
	}
}
