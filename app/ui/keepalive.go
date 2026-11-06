//go:build windows || darwin

package ui

import (
	"context"
	"net/http"
	"sync"
	"time"
)

// keepaliveInterval is how long a chat stream may go without writing before
// a heartbeat is sent. WKWebView (NSURLSession) aborts requests that receive
// no bytes for 60 seconds, which happens during long prompt processing.
var keepaliveInterval = 15 * time.Second

// keepaliveWriter serializes writes to a streaming response and can emit
// heartbeats when the stream has been idle. Heartbeats are bare newlines,
// which JSONL clients skip as blank lines.
type keepaliveWriter struct {
	http.ResponseWriter
	flusher http.Flusher

	mu        sync.Mutex
	lastWrite time.Time
}

func newKeepaliveWriter(w http.ResponseWriter, f http.Flusher) *keepaliveWriter {
	return &keepaliveWriter{ResponseWriter: w, flusher: f, lastWrite: time.Now()}
}

func (k *keepaliveWriter) Write(p []byte) (int, error) {
	k.mu.Lock()
	defer k.mu.Unlock()
	k.lastWrite = time.Now()
	return k.ResponseWriter.Write(p)
}

func (k *keepaliveWriter) Flush() {
	k.mu.Lock()
	defer k.mu.Unlock()
	k.flusher.Flush()
}

// start sends a heartbeat whenever nothing has been written for
// keepaliveInterval. The returned function stops the heartbeat and waits for
// it to exit, so no writes happen after it returns.
func (k *keepaliveWriter) start(ctx context.Context) (stop func()) {
	ctx, cancel := context.WithCancel(ctx)
	done := make(chan struct{})

	go func() {
		defer close(done)
		ticker := time.NewTicker(keepaliveInterval / 2)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				k.mu.Lock()
				if time.Since(k.lastWrite) >= keepaliveInterval {
					k.lastWrite = time.Now()
					_, err := k.ResponseWriter.Write([]byte("\n"))
					if err == nil {
						k.flusher.Flush()
					}
					if err != nil {
						k.mu.Unlock()
						return
					}
				}
				k.mu.Unlock()
			}
		}
	}()

	return func() {
		cancel()
		<-done
	}
}
