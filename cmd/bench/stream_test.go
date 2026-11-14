package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestStreamDrop(t *testing.T) {
	for _, tt := range []struct {
		name string
		rate func(int) int64
		stop bool
	}{
		{"steady", func(int) int64 { return 100 }, false},
		{"sustained drop", func(i int) int64 {
			if i > 10 {
				return 20
			}
			return 100
		}, true},
		{"brief pause recovers", func(i int) int64 {
			if i > 10 && i < 14 {
				return 0
			}
			return 100
		}, false},
		{"speculative bursts", func(i int) int64 {
			if i%2 == 0 {
				return 200
			}
			return 0
		}, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			start := time.Now()
			w := streamWatch{first: start, last: start, history: []streamPoint{{at: start}}}
			var sample streamSample
			for i := 1; i <= 30; i++ {
				now := start.Add(time.Duration(i) * time.Second)
				if chunks := tt.rate(i); chunks > 0 {
					w.chunks += chunks
					w.last = now
				}
				sample = w.sampleLocked(now, streamOptions{drop: 50, hold: 5 * time.Second})
				if sample.StopReason != "" {
					break
				}
			}
			if (sample.StopReason != "") != tt.stop {
				t.Fatalf("sample = %+v, want stop %v", sample, tt.stop)
			}
		})
	}
}

func TestStreamStallStartsWithGeneration(t *testing.T) {
	w := streamWatch{}
	opts := streamOptions{stall: time.Second}
	if got := w.sampleLocked(time.Now().Add(time.Hour), opts); got.StopReason != "" {
		t.Fatalf("prefill treated as generation stall: %+v", got)
	}
	w.add("hello")
	if got := w.sampleLocked(w.last.Add(time.Second), opts); got.StopReason != "no_progress" {
		t.Fatalf("expected generation stall, got %+v", got)
	}
}

func TestStreamCancellationPreservesPartialResult(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, `{"content":"first"}`)
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}))
	t.Cleanup(server.Close)
	backend := &runnerBackend{addr: strings.TrimPrefix(server.URL, "http://"), client: server.Client()}
	path := filepath.Join(t.TempDir(), "stream.jsonl")
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	res, err := completeStream(ctx, backend, completionParams{numPredict: 100}, streamOptions{output: path, stall: time.Millisecond})
	if !errors.Is(err, errStreamSlow) || res.content != "first" {
		t.Fatalf("completion = %+v, %v; want partial output and slowdown error", res, err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var end map[string]any
	s := bufio.NewScanner(f)
	for s.Scan() {
		var record map[string]any
		if err := json.Unmarshal(s.Bytes(), &record); err != nil {
			t.Fatal(err)
		}
		end = record
	}
	if err := s.Err(); err != nil {
		t.Fatal(err)
	}
	if end["status"] != "aborted_slow_stream" || end["chunks"] != float64(1) || end["generated_tokens"] != float64(0) {
		t.Fatalf("incorrect partial measurement: %v", end)
	}
}

func TestStreamOptionsRejectUnsupportedTarget(t *testing.T) {
	f := createTestFlagOptions()
	f.stream = streamOptions{drop: 50, hold: time.Second}
	if err := BenchmarkModel(f); err == nil || !strings.Contains(err.Error(), "require -runner or -spawn") {
		t.Fatalf("unsupported target: %v", err)
	}
}
