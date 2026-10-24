package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"sync"
	"time"
)

// Stream rates are delivery chunks, not tokens: detokenization and speculative
// decoding can combine tokens. A slowdown is evidence to correlate with host
// measurements, not by itself proof of memory pressure.
type streamOptions struct {
	output string
	drop   float64
	hold   time.Duration
	stall  time.Duration
}

func (o streamOptions) enabled() bool { return o.output != "" || o.drop != 0 || o.stall != 0 }

func (o streamOptions) validate() error {
	if math.IsNaN(o.drop) || o.drop < 0 || o.drop >= 100 {
		return errors.New("-abort-drop must be between 0 (disabled) and 100 (exclusive)")
	}
	if o.drop > 0 && o.hold <= 0 {
		return errors.New("-abort-after must be positive when -abort-drop is enabled")
	}
	if o.stall < 0 {
		return errors.New("-stream-stall must not be negative")
	}
	return nil
}

const streamWindow = 5 * time.Second

var errStreamSlow = errors.New("generation stream slowed down")

type streamPoint struct {
	at     time.Time
	chunks int64
}

type streamSample struct {
	Time       time.Time `json:"time"`
	Elapsed    float64   `json:"elapsed_seconds"`
	Chunks     int64     `json:"chunks"`
	Bytes      int64     `json:"bytes"`
	Rate       float64   `json:"chunks_per_second"`
	Baseline   float64   `json:"baseline_chunks_per_second"`
	Idle       float64   `json:"idle_seconds"`
	StopReason string    `json:"stop_reason,omitempty"`
}

// One mutex protects the stream callback and the timer. The timer must keep
// checking even when the HTTP reader is blocked waiting for another chunk.
type streamWatch struct {
	mu       sync.Mutex
	first    time.Time
	last     time.Time
	chunks   int64
	bytes    int64
	history  []streamPoint
	baseline float64
	slow     time.Time
}

func (w *streamWatch) add(content string) {
	if w == nil || content == "" {
		return
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	w.last = time.Now()
	if w.first.IsZero() {
		w.first = w.last
		w.history = append(w.history, streamPoint{at: w.first})
	}
	w.chunks++
	w.bytes += int64(len(content))
}

// sampleLocked requires mu to be held. Keeping the calculation separate from the
// timer lets tests exercise sustained drops and bursty delivery without sleeps.
func (w *streamWatch) sampleLocked(now time.Time, o streamOptions) streamSample {
	s := streamSample{Time: now, Chunks: w.chunks, Bytes: w.bytes}
	if w.first.IsZero() {
		return s
	}
	s.Elapsed = now.Sub(w.first).Seconds()
	s.Idle = now.Sub(w.last).Seconds()
	for len(w.history) > 1 && now.Sub(w.history[1].at) >= streamWindow {
		w.history = w.history[1:]
	}
	prev := w.history[0]
	if elapsed := now.Sub(prev.at).Seconds(); elapsed > 0 {
		s.Rate = float64(w.chunks-prev.chunks) / elapsed
	}
	if w.baseline == 0 && now.Sub(w.first) >= streamWindow {
		w.baseline = float64(w.chunks) / s.Elapsed
	}
	s.Baseline = w.baseline
	w.history = append(w.history, streamPoint{at: now, chunks: w.chunks})
	if o.stall > 0 && now.Sub(w.last) >= o.stall {
		s.StopReason = "no_progress"
	} else if o.drop > 0 && w.baseline > 0 && s.Rate < w.baseline*(1-o.drop/100) {
		if w.slow.IsZero() {
			w.slow = now
		}
		if now.Sub(w.slow) >= o.hold {
			s.StopReason = "sustained_drop"
		}
	} else {
		w.slow = time.Time{}
	}
	return s
}

func completeStream(ctx context.Context, backend benchBackend, p completionParams, o streamOptions) (completionResult, error) {
	if !o.enabled() {
		return backend.Complete(ctx, p)
	}
	var file *os.File
	var writer *bufio.Writer
	var enc *json.Encoder
	if o.output != "" {
		var err error
		file, err = os.OpenFile(o.output, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
		if err != nil {
			return completionResult{}, err
		}
		defer file.Close()
		writer = bufio.NewWriter(file)
		enc = json.NewEncoder(writer)
	}
	write := func(v any) error {
		if enc == nil {
			return nil
		}
		if err := enc.Encode(v); err != nil {
			return err
		}
		return writer.Flush()
	}
	start := time.Now()
	if err := write(map[string]any{"event": "start", "time": start, "requested_tokens": p.numPredict, "abort_drop_percent": o.drop, "abort_after_seconds": o.hold.Seconds(), "stall_seconds": o.stall.Seconds()}); err != nil {
		return completionResult{}, err
	}
	ctx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
	watch := &streamWatch{}
	p.stream = watch
	stop, stopped := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(stopped)
		ticker := time.NewTicker(time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-stop:
				return
			case <-ctx.Done():
				return
			case <-ticker.C:
				watch.mu.Lock()
				sample := watch.sampleLocked(time.Now(), o)
				watch.mu.Unlock()
				if err := write(sample); err != nil {
					cancel(fmt.Errorf("write stream samples: %w", err))
					return
				}
				if sample.StopReason != "" {
					cancel(fmt.Errorf("%w: %s (%.1f chunks/s, baseline %.1f)", errStreamSlow, sample.StopReason, sample.Rate, sample.Baseline))
					return
				}
			}
		}
	}()
	res, err := backend.Complete(ctx, p)
	close(stop)
	<-stopped
	if cause := context.Cause(ctx); cause != nil {
		err = cause
	}
	status := "complete"
	if err != nil {
		status = "error"
		if errors.Is(err, errStreamSlow) {
			status = "aborted_slow_stream"
		}
	}
	message := ""
	if err != nil {
		message = err.Error()
	}
	endErr := write(map[string]any{"event": "end", "time": time.Now(), "status": status, "error": message, "chunks": watch.chunks, "bytes": watch.bytes, "prompt_tokens": res.promptEvalCount, "generated_tokens": res.evalCount, "generate_seconds": res.evalDuration.Seconds(), "ttft_seconds": res.ttft.Seconds(), "stats": res.stats})
	return res, errors.Join(err, endErr)
}
