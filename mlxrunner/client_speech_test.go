package mlxrunner

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
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
		err := client.Speech(t.Context(), llm.SpeechRequest{Input: "hello", Voice: "alba"}, func(b []byte) error { _, err := got.Write(b); return err })
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
