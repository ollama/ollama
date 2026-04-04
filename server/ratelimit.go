package server

import (
	"context"
	"io"
	"strings"

	"golang.org/x/time/rate"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
)

// downloadLimiter is a shared rate limiter for all concurrent download chunks.
// When nil, downloads are unlimited.
var downloadLimiter *rate.Limiter

func init() {
	if limit := parseDownloadSpeed(envconfig.Var("OLLAMA_MAX_DOWNLOAD_SPEED")); limit > 0 {
		// Use a burst size of 512KB to allow smooth throughput with multiple chunks.
		burst := int(limit)
		if burst > 512*int(format.KiloByte) {
			burst = 512 * int(format.KiloByte)
		}
		downloadLimiter = rate.NewLimiter(rate.Limit(limit), burst)
	}
}

// parseDownloadSpeed parses a human-friendly speed string into bytes per second.
// Accepted formats: plain bytes ("1048576"), with suffix ("10m", "100k", "1g"),
// or with full suffix ("10mb", "100kb", "1gb"). Case-insensitive.
// Returns 0 for empty or invalid input.
func parseDownloadSpeed(s string) int64 {
	s = strings.TrimSpace(strings.ToLower(s))
	if s == "" || s == "0" {
		return 0
	}

	multiplier := int64(1)
	// Strip trailing "b" or "b/s" if present (e.g. "10mb" -> "10m", "10mb/s" -> "10m")
	s = strings.TrimSuffix(s, "/s")
	s = strings.TrimSuffix(s, "b")

	switch {
	case strings.HasSuffix(s, "k"):
		multiplier = int64(format.KiloByte)
		s = strings.TrimSuffix(s, "k")
	case strings.HasSuffix(s, "m"):
		multiplier = int64(format.MegaByte)
		s = strings.TrimSuffix(s, "m")
	case strings.HasSuffix(s, "g"):
		multiplier = int64(format.GigaByte)
		s = strings.TrimSuffix(s, "g")
	}

	var value float64
	for i, c := range s {
		if (c >= '0' && c <= '9') || c == '.' {
			continue
		}
		// unknown character at position i
		_ = i
		return 0
	}
	// Parse the numeric part
	n, err := parseFloat(s)
	if err != nil || n <= 0 {
		return 0
	}
	value = n

	return int64(value * float64(multiplier))
}

func parseFloat(s string) (float64, error) {
	var result float64
	var decimal float64
	var inDecimal bool
	var decimalPlace float64 = 0.1

	for _, c := range s {
		switch {
		case c >= '0' && c <= '9':
			if inDecimal {
				decimal += float64(c-'0') * decimalPlace
				decimalPlace *= 0.1
			} else {
				result = result*10 + float64(c-'0')
			}
		case c == '.':
			if inDecimal {
				return 0, io.ErrUnexpectedEOF
			}
			inDecimal = true
		default:
			return 0, io.ErrUnexpectedEOF
		}
	}
	return result + decimal, nil
}

// rateLimitedReader wraps an io.Reader and enforces a global rate limit.
type rateLimitedReader struct {
	r       io.Reader
	ctx     context.Context //nolint:containedctx // ctx required for rate.Limiter.WaitN in io.Reader.Read
	limiter *rate.Limiter
}

func newRateLimitedReader(ctx context.Context, r io.Reader, limiter *rate.Limiter) io.Reader {
	if limiter == nil {
		return r
	}
	return &rateLimitedReader{r: r, ctx: ctx, limiter: limiter}
}

func (r *rateLimitedReader) Read(p []byte) (int, error) {
	// Limit the read size to the burst size to ensure smooth rate limiting
	if len(p) > r.limiter.Burst() {
		p = p[:r.limiter.Burst()]
	}
	n, err := r.r.Read(p)
	if n > 0 {
		if waitErr := r.limiter.WaitN(r.ctx, n); waitErr != nil {
			return n, waitErr
		}
	}
	return n, err
}
