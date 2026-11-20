package server

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/manifest"
)

type speechFunc func(context.Context, llm.SpeechRequest, func([]byte) error) error

func (f speechFunc) Speech(ctx context.Context, req llm.SpeechRequest, emit func([]byte) error) error {
	return f(ctx, req, emit)
}

func TestSpeechValidation(t *testing.T) {
	for _, body := range []string{
		`{`, `{}`, `{"model":"test","input":"  \n"}`,
		`{"input":"hello"}`,
		`{"model":"test","input":"hello","response_format":"mp3"}`,
		`{"model":"test","input":"hello","speed":0}`,
		`{"model":"test","input":"hello","speed":2}`,
		`{"model":"test","input":"hello","stream_format":"sse"}`,
		`{"model":"test","input":"hello","instructions":"whisper"}`,
		`{"model":"test","input":"` + strings.Repeat("é", 4097) + `"}`,
	} {
		t.Run(body[:min(len(body), 90)], func(t *testing.T) {
			w := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(w)
			c.Request = httptest.NewRequest(http.MethodPost, "/v1/audio/speech", strings.NewReader(body))
			// Invalid requests must fail before touching the scheduler.
			(&Server{}).SpeechHandler(c)
			var got struct {
				Error struct {
					Message string `json:"message"`
				} `json:"error"`
			}
			if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil || w.Code != 400 || got.Error.Message == "" {
				t.Fatalf("invalid error response: %d %s", w.Code, w.Body)
			}
		})
	}
}

func TestSpeechStreamsBeforeCompletion(t *testing.T) {
	for _, format := range []string{"pcm", "wav"} {
		t.Run(format, func(t *testing.T) {
			release := make(chan struct{})
			defer close(release)
			pcm := []byte{1, 0, 2, 0}
			speaker := speechFunc(func(ctx context.Context, req llm.SpeechRequest, emit func([]byte) error) error {
				if err := emit(pcm); err != nil {
					return err
				}
				select {
				case <-release:
				case <-ctx.Done():
					return ctx.Err()
				}
				return nil
			})
			router := gin.New()
			router.Use(gin.Recovery())
			router.POST("/", func(c *gin.Context) { streamSpeech(c, speaker, llm.SpeechRequest{}, format) })
			srv := httptest.NewServer(router)
			defer srv.Close()
			req, _ := http.NewRequestWithContext(t.Context(), http.MethodPost, srv.URL, nil)
			res, err := srv.Client().Do(req)
			if err != nil {
				t.Fatal(err)
			}
			defer res.Body.Close()
			if res.Header.Get("Content-Type") != "audio/"+format {
				t.Fatal(res.Header)
			}
			if format == "wav" {
				h := make([]byte, 44)
				if _, err := io.ReadFull(res.Body, h); err != nil {
					t.Fatal(err)
				}
				if string(h[:4]) != "RIFF" || string(h[8:16]) != "WAVEfmt " || string(h[36:40]) != "data" ||
					binary.LittleEndian.Uint32(h[24:]) != 24000 || binary.LittleEndian.Uint16(h[22:]) != 1 ||
					binary.LittleEndian.Uint16(h[34:]) != 16 {
					t.Fatalf("bad WAV header: %x", h)
				}
			}
			got := make([]byte, len(pcm))
			if _, err := io.ReadFull(res.Body, got); err != nil || !bytes.Equal(got, pcm) {
				t.Fatalf("first audio: %x, %v", got, err)
			}
		})
	}
}

func TestSpeechErrors(t *testing.T) {
	for _, tc := range []struct {
		name    string
		emitted bool
		err     error
		status  int
	}{
		{"invalid voice", false, api.StatusError{StatusCode: 400, ErrorMessage: "invalid voice"}, 400},
		{"empty output", false, nil, 500},
		{"generation failed", true, errors.New("generation failed"), 200},
	} {
		t.Run(tc.name, func(t *testing.T) {
			router := gin.New()
			router.Use(gin.Recovery())
			router.POST("/", func(c *gin.Context) {
				streamSpeech(c, speechFunc(func(_ context.Context, _ llm.SpeechRequest, emit func([]byte) error) error {
					if tc.emitted {
						if err := emit([]byte{1, 0}); err != nil {
							return err
						}
					}
					return tc.err
				}), llm.SpeechRequest{}, "pcm")
			})
			srv := httptest.NewServer(router)
			defer srv.Close()
			res, err := srv.Client().Post(srv.URL, "application/json", nil)
			if err != nil {
				t.Fatal(err)
			}
			defer res.Body.Close()
			body, err := io.ReadAll(res.Body)
			if res.StatusCode != tc.status {
				t.Fatalf("status %d: %s", res.StatusCode, body)
			}
			if tc.emitted {
				if !errors.Is(err, io.ErrUnexpectedEOF) {
					t.Fatalf("partial audio looked complete: %v", err)
				}
			} else if err != nil || !bytes.Contains(body, []byte(`"error"`)) {
				t.Fatalf("error response: %s, %v", body, err)
			}
		})
	}
}

func TestSpeechScheduleErrors(t *testing.T) {
	for _, tc := range []struct {
		err     error
		status  int
		message string
	}{
		{fmt.Errorf("model %w", errRequired), 400, "model is required"},
		{errCapabilities, 400, errCapabilities.Error()},
		{manifest.ErrNoCompatibleManifest, 400, manifest.ErrNoCompatibleManifest.Error()},
		{context.Canceled, 499, "request canceled"},
		{ErrMaxQueue, 503, ErrMaxQueue.Error()},
		{os.ErrNotExist, 404, `model "phonon" not found, try pulling it first`},
		{errors.New("load failed"), 500, "load failed"},
	} {
		t.Run(tc.err.Error(), func(t *testing.T) {
			w := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(w)
			status, message := scheduleError("phonon", tc.err)
			openAIError(c, status, message)
			var body struct {
				Error struct {
					Message string
					Type    string
				}
			}
			if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
				t.Fatal(err)
			}
			if w.Code != tc.status || body.Error.Message != tc.message || body.Error.Type == "" {
				t.Fatalf("got %d %s; want %d %q", w.Code, w.Body, tc.status, tc.message)
			}
		})
	}
}

// Gin's Flush method drops errors; speech must reach the underlying writer's
// FlushError so a failed write stops the producer immediately.
type speechFlushErrorWriter struct {
	*httptest.ResponseRecorder
	err error
}

func (w *speechFlushErrorWriter) SetWriteDeadline(time.Time) error { return nil }
func (w *speechFlushErrorWriter) FlushError() error                { return w.err }
func (w *speechFlushErrorWriter) Flush()                           { _ = w.FlushError() }

func TestSpeechFlushErrorStopsProducer(t *testing.T) {
	want := errors.New("connection write failed")
	w := &speechFlushErrorWriter{ResponseRecorder: httptest.NewRecorder(), err: want}
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/audio/speech", nil)
	var got error
	streamSpeech(c, speechFunc(func(_ context.Context, _ llm.SpeechRequest, emit func([]byte) error) error {
		got = emit([]byte{1, 0})
		return got
	}), llm.SpeechRequest{}, "pcm")
	if !errors.Is(got, want) {
		t.Fatalf("flush error did not reach producer: got %v, want %v", got, want)
	}
}
