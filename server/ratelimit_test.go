package server

import (
	"bytes"
	"context"
	"io"
	"testing"
	"time"

	"golang.org/x/time/rate"
)

func TestParseDownloadSpeed(t *testing.T) {
	tests := []struct {
		input string
		want  int64
	}{
		// empty / zero
		{"", 0},
		{"0", 0},
		{"  ", 0},

		// plain bytes
		{"1048576", 1048576},
		{"500", 500},

		// kilobytes
		{"100k", 100_000},
		{"100K", 100_000},
		{"100kb", 100_000},
		{"100KB", 100_000},
		{"100kb/s", 100_000},

		// megabytes
		{"10m", 10_000_000},
		{"10M", 10_000_000},
		{"10mb", 10_000_000},
		{"10MB", 10_000_000},
		{"10mb/s", 10_000_000},
		{"1.5m", 1_500_000},

		// gigabytes
		{"1g", 1_000_000_000},
		{"1G", 1_000_000_000},
		{"1gb", 1_000_000_000},
		{"1GB", 1_000_000_000},
		{"1gb/s", 1_000_000_000},

		// decimal values
		{"2.5m", 2_500_000},
		{"0.5g", 500_000_000},

		// invalid
		{"abc", 0},
		{"-10m", 0},
		{"10x", 0},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got := parseDownloadSpeed(tt.input)
			if got != tt.want {
				t.Errorf("parseDownloadSpeed(%q) = %d, want %d", tt.input, got, tt.want)
			}
		})
	}
}

func TestNewRateLimitedReaderNilLimiter(t *testing.T) {
	buf := bytes.NewBufferString("hello")
	r := newRateLimitedReader(context.Background(), buf, nil)
	// Should return the original reader when limiter is nil
	if r != buf {
		t.Fatal("expected original reader when limiter is nil")
	}
}

func TestRateLimitedReaderLimitsSpeed(t *testing.T) {
	data := make([]byte, 10000)
	for i := range data {
		data[i] = byte(i % 256)
	}

	// Allow 5000 bytes/sec with burst of 5000
	limiter := rate.NewLimiter(5000, 5000)
	r := newRateLimitedReader(context.Background(), bytes.NewReader(data), limiter)

	start := time.Now()
	out, err := io.ReadAll(r)
	elapsed := time.Since(start)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(out) != len(data) {
		t.Fatalf("got %d bytes, want %d", len(out), len(data))
	}
	if !bytes.Equal(out, data) {
		t.Fatal("data mismatch")
	}

	// With 10000 bytes at 5000 bytes/sec, should take at least ~1 second
	// (first burst of 5000 is instant, second 5000 takes ~1s)
	if elapsed < 800*time.Millisecond {
		t.Errorf("read completed too fast (%v), rate limiting may not be working", elapsed)
	}
}

func TestRateLimitedReaderCancellation(t *testing.T) {
	data := make([]byte, 100000)
	// Very slow limiter: 100 bytes/sec
	limiter := rate.NewLimiter(100, 100)

	ctx, cancel := context.WithCancel(context.Background())

	r := newRateLimitedReader(ctx, bytes.NewReader(data), limiter)

	// Cancel context after a short delay
	go func() {
		time.Sleep(50 * time.Millisecond)
		cancel()
	}()

	_, err := io.ReadAll(r)
	if err == nil {
		t.Fatal("expected error from cancelled context")
	}
}

func TestRateLimitedReaderPreservesEOF(t *testing.T) {
	data := []byte("short")
	// Fast limiter so test doesn't slow down
	limiter := rate.NewLimiter(rate.Inf, 1<<20)
	r := newRateLimitedReader(context.Background(), bytes.NewReader(data), limiter)

	out, err := io.ReadAll(r)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if string(out) != "short" {
		t.Fatalf("got %q, want %q", out, "short")
	}
}
