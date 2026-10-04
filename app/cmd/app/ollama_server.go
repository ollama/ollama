//go:build windows || darwin

package main

import (
	"context"
	"errors"
	"log/slog"
	"sync"
)

// serverRunner runs the ollama server until its context is canceled.
type serverRunner interface {
	Run(ctx context.Context) error
}

// ollamaServer supervises the bundled ollama server so settings that change
// its environment can restart it.
type ollamaServer struct {
	server serverRunner

	mu     sync.Mutex
	ctx    context.Context    //nolint:containedctx // Canceled when the app quits, which ends restarts too.
	cancel context.CancelFunc // stops the current run
	done   chan struct{}      // closed when the current run returns
}

// Start runs the server in the background until ctx is canceled.
func (s *ollamaServer) Start(ctx context.Context) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.ctx = ctx
	s.startLocked()
}

// Restart stops the current server and starts one that reads the latest
// settings. It returns after the old server has exited.
func (s *ollamaServer) Restart() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.cancel == nil || s.ctx.Err() != nil {
		return
	}
	slog.Info("restarting ollama server")
	s.cancel()
	<-s.done
	s.startLocked()
}

// Wait blocks until the current server run returns.
func (s *ollamaServer) Wait() {
	s.mu.Lock()
	done := s.done
	s.mu.Unlock()
	if done != nil {
		<-done
	}
}

func (s *ollamaServer) startLocked() {
	ctx, cancel := context.WithCancel(s.ctx)
	done := make(chan struct{})
	s.cancel, s.done = cancel, done
	go func() {
		defer close(done)
		slog.Info("starting ollama server")
		if err := s.server.Run(ctx); err != nil && !errors.Is(err, context.Canceled) {
			slog.Error("ollama server stopped", "error", err)
		}
	}()
}
