package server

import (
	"io"
	"net/http"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

// unlimitedReader hands back as much as asked for, instantly, so the only thing
// pacing a read is the limiter under test.
type unlimitedReader struct{}

func (unlimitedReader) Read(p []byte) (int, error) { return len(p), nil }

// measure reports the bytes per second a set of readers sharing one limiter
// achieves over window.
func measure(t *testing.T, l *rateLimiter, streams int, window time.Duration) float64 {
	t.Helper()
	done := make(chan int64, streams)
	start := time.Now()
	for range streams {
		go func() {
			r := newThrottledReader(unlimitedReader{}, l)
			buf := make([]byte, 32*1024)
			var total int64
			for time.Since(start) < window {
				n, err := r.Read(buf)
				total += int64(n)
				if err != nil {
					break
				}
			}
			done <- total
		}()
	}
	var sum int64
	for range streams {
		sum += <-done
	}
	return float64(sum) / time.Since(start).Seconds()
}

// The bucket used to start full, which handed out one free second. The measured
// average was then (t+1)/t times the limit — 2x over one second, and still 9%
// high after ten. Short windows are the sensitive ones, so test those.
func TestRateLimiterNoStartBurst(t *testing.T) {
	const limit = 500 * 1024
	for _, window := range []time.Duration{time.Second, 2 * time.Second} {
		got := measure(t, newRateLimiter(limit), 1, window)
		factor := got / limit
		t.Logf("window %v: %.0f B/s = %.3fx", window, got, factor)
		if factor > 1.15 {
			t.Errorf("window %v: %.3fx of the limit, want <= 1.15 (start burst back?)", window, factor)
		}
	}
}

// One bucket per stream multiplied the limit by the stream count: with
// OLLAMA_MAX_PULL_STREAMS=4 a "500k" limit delivered close to 2 MB/s. The
// limiter is shared, so the total must not grow with the stream count.
func TestRateLimiterTotalNotPerStream(t *testing.T) {
	const limit = 500 * 1024
	for _, streams := range []int{1, 4, 8} {
		got := measure(t, newRateLimiter(limit), streams, 3*time.Second)
		factor := got / limit
		t.Logf("%d streams: %.0f B/s = %.3fx", streams, got, factor)
		if factor > 1.15 {
			t.Errorf("%d streams: %.3fx of the limit, want <= 1.15 (per-stream bucket back?)", streams, factor)
		}
		if factor < 0.80 {
			t.Errorf("%d streams: %.3fx of the limit, want >= 0.80 (limiter too slow)", streams, factor)
		}
	}
}

// refund puts back what a short read did not consume; without it a reader that
// asks for 32 KiB and gets 100 bytes would be charged the full reservation.
func TestRateLimiterRefundsUnusedReservation(t *testing.T) {
	l := newRateLimiter(1024)
	l.budget = 1024
	r := newThrottledReader(io.LimitReader(unlimitedReader{}, 100), l)
	n, err := r.Read(make([]byte, 512))
	if err != nil {
		t.Fatal(err)
	}
	if n != 100 {
		t.Fatalf("read %d bytes, want 100", n)
	}
	if want := int64(1024 - 100); l.budget != want {
		t.Errorf("budget = %d, want %d (unused reservation not refunded)", l.budget, want)
	}
}

// The per-request limits must win over the environment, and a zero field must
// fall back to it. Without this the flags would be silently decorative. This
// calls the production rule rather than restating it, so the test cannot agree
// with a copy of the logic while the real one is wrong — which is how the
// part-size call site came to take max(request, environment) and let a smaller
// request be overridden.
func TestPullLimitsPrecedence(t *testing.T) {
	for _, tt := range []struct {
		name        string
		envRate     string
		envStreams  string
		req         api.PullLimits
		wantRate    int64
		wantStreams int
	}{
		{
			name: "request wins over environment", envRate: "1m", envStreams: "16",
			req: api.PullLimits{Rate: 512000, Streams: 3}, wantRate: 512000, wantStreams: 3,
		},
		{
			name: "a smaller request is not raised to the environment", envRate: "1m", envStreams: "16",
			req: api.PullLimits{Streams: 2}, wantRate: 1024 * 1024, wantStreams: 2,
		},
		{
			name: "zero falls back to the environment", envRate: "1m", envStreams: "7",
			req: api.PullLimits{}, wantRate: 1024 * 1024, wantStreams: 7,
		},
		{
			name: "rate from request, streams from environment", envRate: "0", envStreams: "5",
			req: api.PullLimits{Rate: 999}, wantRate: 999, wantStreams: 5,
		},
		{
			name: "streams never drop below one", envRate: "0", envStreams: "0",
			req: api.PullLimits{}, wantRate: 0, wantStreams: 1,
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("OLLAMA_MAX_PULL_RATE", tt.envRate)
			t.Setenv("OLLAMA_MAX_PULL_STREAMS", tt.envStreams)

			b := &blobDownload{Limits: tt.req}
			rate, streams := b.effectiveLimits()

			if rate != tt.wantRate {
				t.Errorf("rate = %d, want %d", rate, tt.wantRate)
			}
			if streams != tt.wantStreams {
				t.Errorf("streams = %d, want %d", streams, tt.wantStreams)
			}
		})
	}
}

// throttledClient is what gives the tensor-layer path the same limit, since
// transfer.DownloadOptions has no rate field. A nil limiter must leave the
// client alone so the no-limit path is untouched.
func TestThrottledClientWrapsBodies(t *testing.T) {
	if got := throttledClient(nil, nil); got != nil {
		t.Errorf("throttledClient(nil, nil) = %v, want nil", got)
	}

	base := &http.Client{}
	if got := throttledClient(base, nil); got != base {
		t.Error("a nil limiter must return the base client unchanged")
	}

	l := newRateLimiter(64 * 1024)
	c := throttledClient(base, l)
	if c == base {
		t.Fatal("a limiter must produce a new client, not mutate the base")
	}
	tr, ok := c.Transport.(*throttledTransport)
	if !ok {
		t.Fatalf("Transport = %T, want *throttledTransport", c.Transport)
	}
	if tr.l != l {
		t.Error("transport does not carry the limiter it was given")
	}
	if tr.base == nil {
		t.Error("transport has no base RoundTripper")
	}
}
