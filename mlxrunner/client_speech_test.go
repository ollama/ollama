package mlxrunner

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/model"
)

func TestClientSpeech(t *testing.T) {
	for _, status := range []int{200, 400} {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != "/v1/audio/speech" || r.Method != "POST" {
				t.Errorf("unexpected request: %s %s", r.Method, r.URL)
			}
			if status != 200 {
				http.Error(w, "unknown voice", status)
				return
			}
			for _, b := range [][]byte{{1}, {0, 2}, {0}} {
				_, _ = w.Write(b)
				w.(http.Flusher).Flush()
			}
		}))
		client := &Client{port: srv.Listener.Addr().(*net.TCPAddr).Port, client: srv.Client()}
		var got bytes.Buffer
		err := client.Speech(t.Context(), llm.SpeechRequest{Input: "hello", Voice: "Freya"}, func(b []byte) error { _, err := got.Write(b); return err })
		srv.Close()
		if status == 200 {
			if err != nil || !bytes.Equal(got.Bytes(), []byte{1, 0, 2, 0}) {
				t.Fatalf("audio: %x, %v", got.Bytes(), err)
			}
		} else {
			var se api.StatusError
			if !errors.As(err, &se) || se.StatusCode != 400 || se.ErrorMessage != "unknown voice" {
				t.Fatalf("error lost: %v", err)
			}
		}
	}
}

func TestClientSpeechCancellation(t *testing.T) {
	started := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		close(started)
		<-r.Context().Done()
	}))
	defer srv.Close()
	client := &Client{port: srv.Listener.Addr().(*net.TCPAddr).Port, client: srv.Client()}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- client.Speech(ctx, llm.SpeechRequest{}, func([]byte) error { return nil }) }()
	<-started
	cancel()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation lost: %v", err)
	}
}

func TestPCM16(t *testing.T) {
	got := pcm16([]float32{-2, -1, 0, 1, 2})
	want := []byte{1, 128, 1, 128, 0, 0, 255, 127, 255, 127}
	if !bytes.Equal(got, want) {
		t.Fatalf("PCM: %x, want %x", got, want)
	}
}

type speechTestModel struct {
	model.Model
	chunks    int
	generated chan struct{}
}

func (m speechTestModel) NewSpeech(context.Context, llm.SpeechRequest) (model.SpeechState, error) {
	return &speechTestState{model: m}, nil
}

type speechTestState struct {
	model     speechTestModel
	generated int
}

func (s *speechTestState) Close() { close(s.model.generated) }

func (m speechTestModel) DecodeSpeech(states []model.SpeechState) ([]model.SpeechOutput, error) {
	out := make([]model.SpeechOutput, len(states))
	for i, state := range states {
		s := state.(*speechTestState)
		s.generated++
		out[i] = model.SpeechOutput{Samples: []float32{-1, 0, 1}, Done: s.generated == m.chunks}
	}
	return out, nil
}

type blockedSpeechWriter struct {
	*controlledSpeechWriter
	read <-chan struct{}
}

func (w *blockedSpeechWriter) Write(p []byte) (int, error) {
	<-w.read
	return w.ResponseRecorder.Write(p)
}

// A client reading at playback speed must not hold the MLX worker.
func TestSpeechDoesNotWaitForClient(t *testing.T) {
	m := speechTestModel{chunks: 3, generated: make(chan struct{})}
	r := &Runner{Model: m, mlxThread: mlxtest.Worker(t), parallel: 2}
	defer startTestScheduler(t, r)()
	// Hold every write until released: generation can only finish here if
	// neither the handler nor the MLX worker waits for the client to read.
	read := make(chan struct{})
	unblock := sync.OnceFunc(func() { close(read) })
	w := &blockedSpeechWriter{controlledSpeechWriter: &controlledSpeechWriter{ResponseRecorder: httptest.NewRecorder()}, read: read}
	done := make(chan struct{})
	go func() {
		defer close(done)
		r.speechHandler(w, httptest.NewRequestWithContext(t.Context(), "POST", "/v1/audio/speech", strings.NewReader("{}")))
	}()
	defer func() { unblock(); <-done }()
	select {
	case <-m.generated:
	case <-time.After(10 * time.Second):
		t.Fatal("generation waited for the client to read")
	}
	unblock()
	<-done
	if want := bytes.Repeat([]byte{1, 128, 0, 0, 255, 127}, m.chunks); !bytes.Equal(w.Body.Bytes(), want) {
		t.Fatalf("audio = %x, want %x", w.Body.Bytes(), want)
	}
}

func TestClientSpeechInterruptedStream(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte{1, 0})
		w.(http.Flusher).Flush()
		panic(http.ErrAbortHandler)
	}))
	defer srv.Close()
	client := &Client{port: srv.Listener.Addr().(*net.TCPAddr).Port, client: srv.Client()}
	var got bytes.Buffer
	err := client.Speech(t.Context(), llm.SpeechRequest{}, func(b []byte) error { _, err := got.Write(b); return err })
	if !errors.Is(err, io.ErrUnexpectedEOF) || !bytes.Equal(got.Bytes(), []byte{1, 0}) {
		t.Fatalf("interrupted audio: %x, %v", got.Bytes(), err)
	}
}

func TestClientSpeechConsumerFailure(t *testing.T) {
	canceled := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		w.Write([]byte{1, 0})
		w.(http.Flusher).Flush()
		<-r.Context().Done()
		close(canceled)
	}))
	defer srv.Close()
	client := &Client{port: srv.Listener.Addr().(*net.TCPAddr).Port, client: srv.Client()}
	stop := errors.New("consumer stopped reading")
	err := client.Speech(t.Context(), llm.SpeechRequest{}, func([]byte) error { return stop })
	if !errors.Is(err, stop) {
		t.Fatalf("consumer error lost: %v", err)
	}
	select {
	case <-canceled:
	case <-time.After(5 * time.Second):
		t.Fatal("runner request was not canceled")
	}
}

// The runner's logging wrapper must preserve HTTP controls for streaming.
type controlledSpeechWriter struct {
	*httptest.ResponseRecorder
	deadline time.Time
	flushErr error
}

func (w *controlledSpeechWriter) SetWriteDeadline(t time.Time) error {
	w.deadline = t
	return nil
}

func (w *controlledSpeechWriter) FlushError() error { return w.flushErr }

func TestSpeechResponseControls(t *testing.T) {
	want := errors.New("write failed")
	w := &controlledSpeechWriter{ResponseRecorder: httptest.NewRecorder(), flushErr: want}
	controller := http.NewResponseController(&statusRecorder{ResponseWriter: w})
	deadline := time.Now().Add(time.Second)
	if err := controller.SetWriteDeadline(deadline); err != nil || !w.deadline.Equal(deadline) {
		t.Fatalf("write deadline lost: %v, %v", w.deadline, err)
	}
	if err := controller.Flush(); !errors.Is(err, want) {
		t.Fatalf("flush error lost: %v", err)
	}
}

func TestSpeechModelRejectsTextTokenization(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "this model does not support text tokenization", http.StatusBadRequest)
	}))
	defer srv.Close()
	client := &Client{port: srv.Listener.Addr().(*net.TCPAddr).Port, client: srv.Client()}
	_, err := client.Tokenize(t.Context(), "hello")
	var status api.StatusError
	if !errors.As(err, &status) || status.StatusCode != 400 || status.ErrorMessage != "this model does not support text tokenization" {
		t.Fatalf("tokenization error lost: %v", err)
	}
}
