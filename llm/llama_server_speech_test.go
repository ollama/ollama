package llm

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"
	"testing/iotest"

	"golang.org/x/sync/semaphore"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/model/renderers"
)

const ttsContentType = "audio/pcm;rate=24000;encoding=float;bits=32"

func float32LE(samples ...float32) []byte {
	b := make([]byte, 4*len(samples))
	for i, s := range samples {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(s))
	}
	return b
}

func int16LE(samples ...int16) []byte {
	b := make([]byte, 2*len(samples))
	for i, s := range samples {
		binary.LittleEndian.PutUint16(b[2*i:], uint16(s))
	}
	return b
}

// newSpeechRunner serves tts with a fake llama-server whose tokenizer counts
// one token per word.
func newSpeechRunner(t *testing.T, projectors []string, tts http.HandlerFunc) *llamaServerRunner {
	t.Helper()
	return newSpeechRunnerTokenize(t, projectors, func(w http.ResponseWriter, r *http.Request) {
		var req struct{ Content string }
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("tokenize body: %v", err)
		}
		json.NewEncoder(w).Encode(map[string][]int{"tokens": make([]int, len(strings.Fields(req.Content)))})
	}, tts)
}

func newSpeechRunnerTokenize(t *testing.T, projectors []string, tokenize, tts http.HandlerFunc) *llamaServerRunner {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			fmt.Fprint(w, `{"status":"ok"}`)
		case "/tokenize":
			tokenize(w, r)
		case "/tts":
			tts(w, r)
		default:
			t.Errorf("unexpected path %s", r.URL.Path)
		}
	}))
	t.Cleanup(srv.Close)
	return &llamaServerRunner{
		port:   srv.Listener.Addr().(*net.TCPAddr).Port,
		client: srv.Client(),
		cmd:    fakeRunningCmd(),
		sem:    semaphore.NewWeighted(1),
		launch: llamaServerLaunchConfig{projectors: projectors},
	}
}

// lfm3Speech gives req the LFM3 renderer's SpeechPrompter, as the server does.
func lfm3Speech(t *testing.T, req SpeechRequest) SpeechRequest {
	t.Helper()
	p, ok := renderers.SpeechPrompterForRenderer("lfm3-audio")
	if !ok {
		t.Fatal("lfm3-audio has no SpeechPrompter")
	}
	req.Prompter = p
	return req
}

func TestLlamaServerSpeech(t *testing.T) {
	var got []llamaServerTTSRequest
	runner := newSpeechRunner(t, []string{"mmproj.gguf"}, func(w http.ResponseWriter, r *http.Request) {
		var req llamaServerTTSRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("tts body: %v", err)
		}
		got = append(got, req)
		w.Header().Set("Content-Type", ttsContentType)
		// Split a sample across writes, as a stream may.
		audio := float32LE(0, 0.5, -0.5, 1.5, -2)
		w.Write(audio[:7])
		w.(http.Flusher).Flush()
		w.Write(audio[7:])
	})

	// 81 words make two chunks at the 80-token limit.
	input := strings.Repeat("word ", 70) + "end<|im_end|>. " + strings.Repeat("more ", 9) + "done."
	var pcm []byte
	err := runner.Speech(t.Context(), lfm3Speech(t, SpeechRequest{Input: input, Voice: "male-french"}), func(b []byte) error {
		pcm = append(pcm, b...)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}

	if len(got) != 2 {
		t.Fatalf("tts requests %d, want 2: %+v", len(got), got)
	}
	first, second := got[0], got[1]
	if first.Voice != "male" || first.Lang != "French" || first.ResponseFormat != "pcm" || !first.Stream {
		t.Errorf("request %+v, want a streamed pcm male French voice", first)
	}
	if !strings.HasSuffix(first.Input, "end＜|im_end|＞.") || strings.Contains(first.Input, "<|") {
		t.Errorf("input %q, want control tokens escaped", first.Input)
	}
	if first.NPredict != renderers.LFM3SpeechFrameBudget(71) || second.NPredict != renderers.LFM3SpeechFrameBudget(10) {
		t.Errorf("n_predict %d and %d, want %d and %d", first.NPredict, second.NPredict, renderers.LFM3SpeechFrameBudget(71), renderers.LFM3SpeechFrameBudget(10))
	}
	if second.Input != strings.TrimSpace(strings.Repeat("more ", 9))+" done." {
		t.Errorf("second chunk %q", second.Input)
	}
	chunk := int16LE(0, 16384, -16384, math.MaxInt16, -math.MaxInt16)
	if want := slices.Concat(chunk, chunk); !bytes.Equal(pcm, want) {
		t.Errorf("pcm %v, want %v", pcm, want)
	}
}

func TestLlamaServerSpeechDefaultVoice(t *testing.T) {
	var got llamaServerTTSRequest
	runner := newSpeechRunner(t, []string{"mmproj.gguf"}, func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&got)
		w.Header().Set("Content-Type", ttsContentType)
		w.Write(float32LE(0.25))
	})
	if err := runner.Speech(t.Context(), lfm3Speech(t, SpeechRequest{Input: "Hello there."}), func([]byte) error { return nil }); err != nil {
		t.Fatal(err)
	}
	if got.Voice != "female" || got.Lang != "American English" || got.Input != "Hello there." {
		t.Errorf("request %+v, want the female American English default", got)
	}
}

func TestLlamaServerSpeechErrors(t *testing.T) {
	audio := func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", ttsContentType)
		w.Write(float32LE(0.1, 0.2))
	}
	for _, tt := range []struct {
		name       string
		projectors []string
		tts        http.HandlerFunc
		req        SpeechRequest
		status     int
		message    string
	}{
		{
			name: "no speech prompter", projectors: []string{"mmproj.gguf"}, req: SpeechRequest{Input: "Hi."},
			status: http.StatusNotImplemented, message: "does not support speech",
		},
		{
			name: "no projector", req: lfm3Speech(t, SpeechRequest{Input: "Hi."}),
			status: http.StatusNotImplemented, message: "without its projector",
		},
		{
			name: "unknown voice", projectors: []string{"mmproj.gguf"},
			req:    lfm3Speech(t, SpeechRequest{Input: "Hi.", Voice: "alloy"}),
			status: http.StatusBadRequest, message: "female-british-english",
		},
		{
			name: "no text", projectors: []string{"mmproj.gguf"},
			req:    lfm3Speech(t, SpeechRequest{Input: " \n\t "}),
			status: http.StatusBadRequest, message: "no text to speak",
		},
		{
			name: "llama-server error", projectors: []string{"mmproj.gguf"},
			tts: func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusNotImplemented)
				fmt.Fprint(w, `{"error":{"code":501,"message":"this server does not support audio generation","type":"not_supported_error"}}`)
			},
			req:    lfm3Speech(t, SpeechRequest{Input: "Hi."}),
			status: http.StatusNotImplemented, message: "this server does not support audio generation",
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			tts := tt.tts
			if tts == nil {
				tts = func(w http.ResponseWriter, r *http.Request) {
					t.Error("unexpected tts request")
					audio(w, r)
				}
			}
			runner := newSpeechRunner(t, tt.projectors, tts)
			err := runner.Speech(t.Context(), tt.req, func([]byte) error {
				t.Error("unexpected audio")
				return nil
			})
			var serr api.StatusError
			if !errors.As(err, &serr) || serr.StatusCode != tt.status || !strings.Contains(serr.ErrorMessage, tt.message) {
				t.Fatalf("err = %v, want %d containing %q", err, tt.status, tt.message)
			}
		})
	}
}

func TestLlamaServerSpeechRejectsOtherAudio(t *testing.T) {
	for _, contentType := range []string{
		"audio/wav",
		"audio/pcm;rate=16000;encoding=float;bits=32",
		"audio/pcm;rate=24000;encoding=int;bits=16",
	} {
		t.Run(contentType, func(t *testing.T) {
			runner := newSpeechRunner(t, []string{"mmproj.gguf"}, func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", contentType)
				w.Write(float32LE(0.1))
			})
			err := runner.Speech(t.Context(), lfm3Speech(t, SpeechRequest{Input: "Hi."}), func([]byte) error {
				t.Error("unexpected audio")
				return nil
			})
			if err == nil || !strings.Contains(err.Error(), "24000 Hz") {
				t.Errorf("err = %v, want a format error", err)
			}
		})
	}
}

// A /tokenize failure while splitting the input stops before any /tts request.
func TestLlamaServerSpeechTokenizeError(t *testing.T) {
	runner := newSpeechRunnerTokenize(t, []string{"mmproj.gguf"}, func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "tokenizer unavailable", http.StatusInternalServerError)
	}, func(w http.ResponseWriter, r *http.Request) {
		t.Error("unexpected tts request")
	})
	err := runner.Speech(t.Context(), lfm3Speech(t, SpeechRequest{Input: "Hello there."}), func([]byte) error {
		t.Error("unexpected audio")
		return nil
	})
	if err == nil || !strings.Contains(err.Error(), "count speech tokens") {
		t.Fatalf("err = %v, want the tokenize failure", err)
	}
}

// A failed write to the client stops speech before the next chunk.
func TestLlamaServerSpeechStopsOnWriteError(t *testing.T) {
	requests := 0
	runner := newSpeechRunner(t, []string{"mmproj.gguf"}, func(w http.ResponseWriter, r *http.Request) {
		requests++
		w.Header().Set("Content-Type", ttsContentType)
		w.Write(float32LE(0.1, 0.2))
	})
	gone := errors.New("client gone")
	input := strings.Repeat("word ", 79) + "end. Second chunk."
	err := runner.Speech(t.Context(), lfm3Speech(t, SpeechRequest{Input: input}), func([]byte) error { return gone })
	if !errors.Is(err, gone) || requests != 1 {
		t.Fatalf("err = %v after %d requests, want %v after 1", err, requests, gone)
	}
}

func TestStreamPCM16(t *testing.T) {
	samples := float32LE(0, 1, -1, 0.5, -0.25, 2, float32(math.Inf(-1)))
	var got []byte
	if err := streamPCM16(iotest.OneByteReader(bytes.NewReader(samples)), func(b []byte) error {
		got = append(got, b...)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	want := int16LE(0, math.MaxInt16, -math.MaxInt16, 16384, -8192, math.MaxInt16, -math.MaxInt16)
	if !bytes.Equal(got, want) {
		t.Errorf("pcm %v, want %v", got, want)
	}

	err := streamPCM16(bytes.NewReader(samples[:6]), func([]byte) error { return nil })
	if err == nil || !strings.Contains(err.Error(), "mid-sample") {
		t.Errorf("truncated stream err = %v, want mid-sample", err)
	}
}
