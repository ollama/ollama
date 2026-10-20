package mlxrunner

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"net/http"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/model"
)

// speechHandler streams 24 kHz 16-bit PCM as the model generates it.
func (r *Runner) speechHandler(w http.ResponseWriter, req *http.Request) {
	speaker, ok := r.Model.(model.SpeechModel)
	if !ok {
		http.Error(w, "this model does not support speech", http.StatusNotImplemented)
		return
	}
	var input llm.SpeechRequest
	if err := json.NewDecoder(req.Body).Decode(&input); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Generation queues audio for a separate writer instead of waiting on the
	// client, so a client reading at playback speed does not hold the MLX worker.
	ctx, cancel := context.WithCancel(req.Context())
	defer cancel()
	queue := &audioQueue{ready: make(chan struct{}, 1)}
	var streaming bool
	var writeErr error
	written := make(chan struct{})
	go func() {
		defer close(written)
		streaming, writeErr = writeAudio(w, queue)
		if writeErr != nil {
			cancel()
		}
	}()
	// The writer must not outlive the handler, even if generation panics.
	defer func() { queue.close(); <-written }()

	err := r.mlxThread.Do(ctx, func() error {
		defer mlx.ClearCache()
		return speaker.Speak(ctx, input, func(samples []float32) error {
			queue.push(pcm16(samples))
			return nil
		})
	})
	queue.close()
	<-written
	if err == nil {
		err = writeErr
	}
	switch {
	case err == nil:
	case streaming:
		// Truncate the stream so the client sees the failure.
		panic(http.ErrAbortHandler)
	default:
		status := http.StatusInternalServerError
		var statusErr api.StatusError
		if errors.As(err, &statusErr) {
			status = statusErr.StatusCode
		}
		http.Error(w, err.Error(), status)
	}
}

// writeAudio streams queued audio until the queue is closed and reports
// whether any audio was written.
func writeAudio(w http.ResponseWriter, queue *audioQueue) (streaming bool, err error) {
	controller := http.NewResponseController(w)
	defer controller.SetWriteDeadline(time.Time{})
	for {
		chunks, ok := queue.next()
		if !ok {
			return streaming, nil
		}
		for _, pcm := range chunks {
			// A stalled reader must not hold the connection forever.
			if err := controller.SetWriteDeadline(time.Now().Add(30 * time.Second)); err != nil {
				return streaming, err
			}
			if !streaming {
				w.Header().Set("Content-Type", "audio/pcm")
				streaming = true
			}
			if _, err := w.Write(pcm); err != nil {
				return streaming, err
			}
		}
		if err := controller.Flush(); err != nil {
			return streaming, err
		}
	}
}

// audioQueue hands audio from generation to the writer without ever blocking
// generation.
type audioQueue struct {
	mu      sync.Mutex
	pending [][]byte
	closed  bool
	ready   chan struct{}
}

func (q *audioQueue) push(pcm []byte) {
	q.mu.Lock()
	q.pending = append(q.pending, pcm)
	q.mu.Unlock()
	q.wake()
}

func (q *audioQueue) close() {
	q.mu.Lock()
	q.closed = true
	q.mu.Unlock()
	q.wake()
}

func (q *audioQueue) wake() {
	select {
	case q.ready <- struct{}{}:
	default:
	}
}

// next waits for queued audio. It returns false once the queue is closed and
// empty.
func (q *audioQueue) next() ([][]byte, bool) {
	for {
		q.mu.Lock()
		pending, closed := q.pending, q.closed
		q.pending = nil
		q.mu.Unlock()
		if len(pending) > 0 || closed {
			return pending, len(pending) > 0
		}
		<-q.ready
	}
}

func pcm16(samples []float32) []byte {
	b := make([]byte, 2*len(samples))
	for i, s := range samples {
		v := math.Round(float64(max(-1, min(1, s))) * math.MaxInt16)
		binary.LittleEndian.PutUint16(b[2*i:], uint16(int16(v)))
	}
	return b
}
