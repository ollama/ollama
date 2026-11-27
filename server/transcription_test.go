package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"golang.org/x/text/language"

	"github.com/ollama/ollama/api"
	gguftest "github.com/ollama/ollama/internal/testutil/gguf"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/mlxrunner/model/audio"
)

// testSpeech returns rate Hz audio of tone bursts separated by short pauses.
func testSpeech(seconds, rate int) []byte {
	samples := make([]float32, seconds*rate)
	for i := range samples {
		if i%(3*rate) < rate/2 {
			continue
		}
		samples[i] = 0.3 * float32(math.Sin(2*math.Pi*440*float64(i)/float64(rate)))
	}
	return audio.EncodeWAV(samples, rate)
}

func audioTextRequest(t *testing.T, path string, fields map[string]string, file []byte) *http.Request {
	t.Helper()
	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	for k, v := range fields {
		if err := mw.WriteField(k, v); err != nil {
			t.Fatal(err)
		}
	}
	if file != nil {
		fw, err := mw.CreateFormFile("file", "speech.wav")
		if err != nil {
			t.Fatal(err)
		}
		if _, err := fw.Write(file); err != nil {
			t.Fatal(err)
		}
	}
	if err := mw.Close(); err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, path, &body)
	req.Header.Set("Content-Type", mw.FormDataContentType())
	return req
}

func serveAudioText(s *Server, req *http.Request) *httptest.ResponseRecorder {
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = req
	if strings.HasSuffix(req.URL.Path, "translations") {
		s.TranslationHandler(c)
	} else {
		s.TranscriptionHandler(c)
	}
	return w
}

func TestAudioTextValidation(t *testing.T) {
	clip := testSpeech(1, 16000)
	cases := []struct {
		name   string
		path   string
		fields map[string]string
		file   []byte
		want   string
	}{
		{"no model", "/v1/audio/transcriptions", map[string]string{}, clip, "model is required"},
		{"no file", "/v1/audio/transcriptions", map[string]string{"model": "m"}, nil, "file is required"},
		{"empty file", "/v1/audio/transcriptions", map[string]string{"model": "m"}, []byte{}, "audio file is empty"},
		{"format", "/v1/audio/transcriptions", map[string]string{"model": "m", "response_format": "srt"}, clip, "response_format"},
		{"language", "/v1/audio/transcriptions", map[string]string{"model": "m", "language": "not a language"}, clip, "unsupported language"},
		{"private-use language", "/v1/audio/transcriptions", map[string]string{"model": "m", "language": "qaa"}, clip, "unsupported language"},
		{"uncoded language", "/v1/audio/transcriptions", map[string]string{"model": "m", "language": "mis"}, clip, "unsupported language"},
		{"private tag", "/v1/audio/transcriptions", map[string]string{"model": "m", "language": "x-foo"}, clip, "unsupported language"},
		{"to_language on transcription", "/v1/audio/transcriptions", map[string]string{"model": "m", "to_language": "de"}, clip, "only supported for translations"},
		{"to_language", "/v1/audio/translations", map[string]string{"model": "m", "to_language": "??"}, clip, "unsupported to_language"},
		{"temperature", "/v1/audio/transcriptions", map[string]string{"model": "m", "temperature": "warm"}, clip, "invalid temperature"},
		{"NaN temperature", "/v1/audio/transcriptions", map[string]string{"model": "m", "temperature": "NaN"}, clip, "invalid temperature"},
		{"infinite temperature", "/v1/audio/transcriptions", map[string]string{"model": "m", "temperature": "Inf"}, clip, "invalid temperature"},
		{"negative infinite temperature", "/v1/audio/translations", map[string]string{"model": "m", "temperature": "-Inf"}, clip, "invalid temperature"},
		{"stream", "/v1/audio/transcriptions", map[string]string{"model": "m", "stream": "maybe"}, clip, "invalid stream"},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			// Invalid requests must fail before touching the scheduler.
			w := serveAudioText(&Server{}, audioTextRequest(t, tt.path, tt.fields, tt.file))
			var got struct {
				Error struct {
					Message string `json:"message"`
				} `json:"error"`
			}
			if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil || w.Code != http.StatusBadRequest {
				t.Fatalf("got %d %s, want a 400 error", w.Code, w.Body)
			}
			if !strings.Contains(got.Error.Message, tt.want) {
				t.Fatalf("error %q, want one mentioning %q", got.Error.Message, tt.want)
			}
		})
	}
}

func TestSplitAudio(t *testing.T) {
	t.Run("short clip passes through", func(t *testing.T) {
		clip := testSpeech(10, 16000)
		clips, err := splitAudio(clip, 30*time.Second)
		if err != nil {
			t.Fatal(err)
		}
		if len(clips) != 1 || !bytes.Equal(clips[0].data, clip) || clips[0].duration != 10*time.Second {
			t.Fatalf("got %d clips, duration %v", len(clips), clips[0].duration)
		}
	})
	t.Run("long clip splits", func(t *testing.T) {
		clips, err := splitAudio(testSpeech(70, 16000), 30*time.Second)
		if err != nil {
			t.Fatal(err)
		}
		if len(clips) != 3 {
			t.Fatalf("got %d clips, want 3", len(clips))
		}
		var total time.Duration
		for _, c := range clips {
			if c.duration > 30*time.Second {
				t.Errorf("clip of %v exceeds the limit", c.duration)
			}
			samples, rate, err := audio.Decode(c.wav())
			if err != nil || rate != 16000 || time.Duration(len(samples))*time.Second/16000 != c.duration {
				t.Errorf("clip decodes to %d samples at %d Hz (%v), want %v", len(samples), rate, err, c.duration)
			}
			total += c.duration
		}
		if total != 70*time.Second {
			t.Errorf("clips total %v, want 70s", total)
		}
	})
	t.Run("unrecognized container passes through", func(t *testing.T) {
		data := []byte("ID3 not a wav file")
		clips, err := splitAudio(data, 30*time.Second)
		if err != nil || len(clips) != 1 || !bytes.Equal(clips[0].data, data) || clips[0].duration != 0 {
			t.Fatalf("got %v, %v", clips, err)
		}
	})
	t.Run("corrupt wav", func(t *testing.T) {
		if _, err := splitAudio([]byte("RIFF\x00\x00\x00\x00WAVEjunk"), 30*time.Second); err == nil {
			t.Fatal("expected an error")
		}
	})
}

func TestNeedsSpace(t *testing.T) {
	for _, tt := range []struct {
		prev, next string
		want       bool
	}{
		{"hello", "world", true},
		{"我每天早上喝咖啡。", "然後去上班", false},
		{"今日は", "いい天気", false},
		{"hello.", "然後", true},
		{"안녕하세요.", "반갑습니다", true},
	} {
		if got := needsSpace(tt.prev, tt.next); got != tt.want {
			t.Errorf("needsSpace(%q, %q) = %v, want %v", tt.prev, tt.next, got, tt.want)
		}
	}
}

// audioTestServer serves one model created with renderer and audio input;
// the runner answers each completion with respond's text.
func audioTestServer(t *testing.T, renderer string, audioInput bool, respond func(n int, prompt string) string) (*Server, *[]llm.CompletionRequest) {
	t.Helper()
	return audioTestServerNamed(t, "speech-test", renderer, audioInput, respond)
}

func audioTestServerNamed(t *testing.T, name, renderer string, audioInput bool, respond func(n int, prompt string) string) (*Server, *[]llm.CompletionRequest) {
	t.Helper()
	return audioTestServerFunc(t, name, renderer, audioInput, func(n int, prompt string) (string, error) {
		return respond(n, prompt), nil
	})
}

// audioTestServerFunc is audioTestServerNamed with a runner that can fail.
func audioTestServerFunc(t *testing.T, name, renderer string, audioInput bool, respond func(n int, prompt string) (string, error)) (*Server, *[]llm.CompletionRequest) {
	t.Helper()
	t.Setenv("OLLAMA_CONTEXT_LENGTH", "4096")
	gin.SetMode(gin.TestMode)

	var mu sync.Mutex
	var requests []llm.CompletionRequest
	mock := mockRunner{
		CompletionFn: func(_ context.Context, r llm.CompletionRequest, fn func(llm.CompletionResponse)) error {
			mu.Lock()
			requests = append(requests, r)
			n := len(requests)
			mu.Unlock()
			content, err := respond(n, r.Prompt)
			if err != nil {
				return err
			}
			fn(llm.CompletionResponse{Content: content, Done: true, DoneReason: llm.DoneReasonStop})
			return nil
		},
	}
	s := &Server{
		sched: &Scheduler{
			pendingReqCh:    make(chan *LlmRequest, 1),
			finishedReqCh:   make(chan *LlmRequest, 1),
			expiredCh:       make(chan *runnerRef, 1),
			unloadedCh:      make(chan any, 1),
			loaded:          make(map[string]*runnerRef),
			newServerFn:     newMockServer(&mock),
			getGpuFn:        getGpuFn,
			getSystemInfoFn: getSystemInfoFn,
			waitForRecovery: 250 * time.Millisecond,
			loadFn: func(req *LlmRequest, _ ml.SystemInfo, _ []ml.DeviceInfo, _ bool) bool {
				req.successCh <- &runnerRef{llama: &mock}
				return false
			},
		},
	}
	go s.sched.Run(t.Context())

	kv := gguftest.KV{
		"general.architecture":          "qwen3",
		"qwen3.block_count":             uint32(1),
		"qwen3.context_length":          uint32(8192),
		"qwen3.embedding_length":        uint32(4096),
		"qwen3.attention.head_count":    uint32(32),
		"qwen3.attention.head_count_kv": uint32(8),
		"tokenizer.ggml.tokens":         []string{""},
		"tokenizer.ggml.scores":         []float32{0},
		"tokenizer.ggml.token_type":     []int32{0},
	}
	if audioInput {
		kv["qwen3.audio.block_count"] = uint32(1)
	}
	_, digest := createBinFile(t, kv, []*gguftest.Tensor{
		{Name: "token_embd.weight", Shape: []uint64{1}, WriterTo: bytes.NewReader(make([]byte, 4))},
		{Name: "output.weight", Shape: []uint64{1}, WriterTo: bytes.NewReader(make([]byte, 4))},
	})
	w := createRequest(t, s.CreateHandler, api.CreateRequest{
		Model:    name,
		Files:    map[string]string{"file.gguf": digest},
		Renderer: renderer,
		Stream:   &stream,
	})
	if w.Code != http.StatusOK {
		t.Fatalf("create: %d %s", w.Code, w.Body)
	}
	return s, &requests
}

func TestTranscriptionSplitsLongAudio(t *testing.T) {
	s, requests := audioTestServer(t, "lfm3-audio", true, func(n int, _ string) string {
		return []string{"", " one ", "two", "three\n"}[n]
	})
	w := serveAudioText(s, audioTextRequest(t, "/v1/audio/transcriptions",
		map[string]string{"model": "speech-test", "language": "fr"}, testSpeech(70, 16000)))
	if w.Code != http.StatusOK {
		t.Fatalf("status %d: %s", w.Code, w.Body)
	}
	var got struct{ Text string }
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil || got.Text != "one two three" {
		t.Fatalf("got %q (%v), want %q", got.Text, err, "one two three")
	}
	if len(*requests) != 3 {
		t.Fatalf("got %d generations, want 3", len(*requests))
	}
	for _, r := range *requests {
		if !strings.Contains(r.Prompt, "Perform ASR in French.") || len(r.Media) != 1 {
			t.Errorf("prompt %q with %d clips", r.Prompt, len(r.Media))
		}
		if r.Options.NumPredict <= 0 || r.Options.NumPredict > maxAudioTextTokens(30*time.Second) {
			t.Errorf("num_predict = %d", r.Options.NumPredict)
		}
		if r.Options.Temperature != 0 {
			t.Errorf("temperature = %v", r.Options.Temperature)
		}
	}
}

func TestTranslationTargets(t *testing.T) {
	for _, tt := range []struct {
		fields map[string]string
		want   string
	}{
		{map[string]string{}, "Translate the following speech segment into English."},
		{map[string]string{"language": "de", "to_language": "es"}, "Translate the following speech segment in German into Spanish."},
		{map[string]string{"to_language": "pt-BR"}, "Translate the following speech segment into Portuguese."},
	} {
		s, requests := audioTestServer(t, "gemma4", true, func(int, string) string { return " Buenos días\n" })
		tt.fields["model"] = "speech-test"
		tt.fields["response_format"] = "text"
		w := serveAudioText(s, audioTextRequest(t, "/v1/audio/translations", tt.fields, testSpeech(5, 16000)))
		if w.Code != http.StatusOK || w.Body.String() != "Buenos días" {
			t.Fatalf("%v: status %d: %q", tt.fields, w.Code, w.Body)
		}
		if p := (*requests)[0].Prompt; !strings.Contains(p, tt.want) {
			t.Errorf("%v: prompt %q, want %q", tt.fields, p, tt.want)
		}
	}
}

func TestTranslationResolvesGemma4Size(t *testing.T) {
	// A stored "gemma4" renderer resolves by model size, as chat does; the
	// large models get the transcribe-then-translate format.
	s, requests := audioTestServerNamed(t, "speech-12b", "gemma4", true, func(int, string) string { return "Hola\nEnglish: Hello" })
	w := serveAudioText(s, audioTextRequest(t, "/v1/audio/translations",
		map[string]string{"model": "speech-12b", "response_format": "text"}, testSpeech(5, 16000)))
	if w.Code != http.StatusOK || w.Body.String() != "Hello" {
		t.Fatalf("status %d: %q", w.Code, w.Body)
	}
	if p := (*requests)[0].Prompt; !strings.Contains(p, "then translate it into English") {
		t.Errorf("prompt %q", p)
	}
}

func TestTranscriptionVerboseJSON(t *testing.T) {
	s, _ := audioTestServer(t, "lfm3-audio", true, func(n int, _ string) string {
		return fmt.Sprintf("part%d", n)
	})
	w := serveAudioText(s, audioTextRequest(t, "/v1/audio/transcriptions",
		map[string]string{"model": "speech-test", "language": "de", "response_format": "verbose_json"}, testSpeech(40, 16000)))
	if w.Code != http.StatusOK {
		t.Fatalf("status %d: %s", w.Code, w.Body)
	}
	var got struct {
		Task     string
		Language string
		Duration float64
		Text     string
		Segments []struct {
			ID         int
			Start, End float64
			Text       string
		}
	}
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Task != "transcribe" || got.Language != "german" || got.Duration != 40 || got.Text != "part1 part2" || len(got.Segments) != 2 {
		t.Fatalf("got %+v", got)
	}
	if s0, s1 := got.Segments[0], got.Segments[1]; s0.Start != 0 || s0.End != s1.Start || s1.End != 40 || s1.ID != 1 || s1.Text != "part2" {
		t.Fatalf("segments %+v", got.Segments)
	}
}

func TestTranslationVerboseJSONLanguage(t *testing.T) {
	// Translations report the output language, as OpenAI does.
	for _, tt := range []struct {
		fields map[string]string
		want   string
	}{
		{map[string]string{}, "english"},
		{map[string]string{"language": "de"}, "english"},
		{map[string]string{"language": "de", "to_language": "es"}, "spanish"},
	} {
		s, _ := audioTestServer(t, "lfm3-audio", true, func(int, string) string { return "Hola" })
		tt.fields["model"] = "speech-test"
		tt.fields["response_format"] = "verbose_json"
		w := serveAudioText(s, audioTextRequest(t, "/v1/audio/translations", tt.fields, testSpeech(5, 16000)))
		if w.Code != http.StatusOK {
			t.Fatalf("%v: status %d: %s", tt.fields, w.Code, w.Body)
		}
		var got struct{ Task, Language, Text string }
		if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
			t.Fatal(err)
		}
		if got.Task != "translate" || got.Language != tt.want || got.Text != "Hola" {
			t.Errorf("%v: got %+v, want language %q", tt.fields, got, tt.want)
		}
	}
}

func TestTranscriptionRejectsImplausibleSampleRate(t *testing.T) {
	s, requests := audioTestServer(t, "lfm3-audio", true, func(int, string) string { return "x" })
	w := serveAudioText(s, audioTextRequest(t, "/v1/audio/transcriptions",
		map[string]string{"model": "speech-test"}, audio.EncodeWAV(make([]float32, 16), 1_000_000_000)))
	if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), "1000000000 Hz") {
		t.Fatalf("status %d: %s", w.Code, w.Body)
	}
	if len(*requests) != 0 {
		t.Fatalf("ran %d generations", len(*requests))
	}
}

func TestAudioTextUploadTooLarge(t *testing.T) {
	req := audioTextRequest(t, "/v1/audio/transcriptions", map[string]string{"model": "m"}, testSpeech(1, 16000))
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = req
	req.Body = http.MaxBytesReader(w, req.Body, 100)
	(&Server{}).TranscriptionHandler(c)
	if w.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("status %d: %s", w.Code, w.Body)
	}
}

func TestParseAudioLanguage(t *testing.T) {
	for in, want := range map[string]string{"fr": "fr", "pt-BR": "pt-BR", "French": "fr", " chinese ": "zh", "iw": "he"} {
		got, err := parseAudioLanguage("language", in, language.Und)
		if err != nil || got.String() != want {
			t.Errorf("parseAudioLanguage(%q) = %v, %v; want %s", in, got, err, want)
		}
	}
}

func TestTranscriptionStreamErrorAfterStart(t *testing.T) {
	// The second piece fails after the first has streamed: the client gets
	// the first delta, then an error event, and no done event.
	s, _ := audioTestServerFunc(t, "speech-test", "lfm3-audio", true, func(n int, _ string) (string, error) {
		if n == 2 {
			return "", errors.New("runner crashed")
		}
		return "part1", nil
	})
	w := serveAudioText(s, audioTextRequest(t, "/v1/audio/transcriptions",
		map[string]string{"model": "speech-test", "stream": "true"}, testSpeech(40, 16000)))
	body := w.Body.String()
	if !strings.HasPrefix(body, `data: {"delta":"part1","type":"transcript.text.delta"}`) ||
		!strings.Contains(body, "runner crashed") || strings.Contains(body, "transcript.text.done") {
		t.Fatalf("got %q", body)
	}
}

func TestChatResponseWriterSplitLines(t *testing.T) {
	var got []string
	w := &chatResponseWriter{header: http.Header{}, status: http.StatusOK, cancel: func() {}, fn: func(r api.ChatResponse) error {
		got = append(got, r.Message.Content)
		return nil
	}}
	stream := `{"message":{"content":"one"}}` + "\n" + `{"message":{"content":"two"}}` + "\n" + `{"message":{"content":"three"},"done":true}`
	for i := 0; i < len(stream); i += 7 {
		if _, err := w.Write([]byte(stream[i:min(i+7, len(stream))])); err != nil {
			t.Fatal(err)
		}
	}
	if err := w.result(); err != nil {
		t.Fatal(err)
	}
	if strings.Join(got, ",") != "one,two,three" {
		t.Fatalf("got %v", got)
	}

	w = &chatResponseWriter{header: http.Header{}, status: http.StatusOK, cancel: func() {}, fn: func(api.ChatResponse) error { return nil }}
	w.Write([]byte(`{"message":{"content":"a"}}` + "\n" + `{"error":"out of mem`))
	w.Write([]byte(`ory"}` + "\n"))
	if err := w.result(); err == nil || !strings.Contains(err.Error(), "out of memory") {
		t.Fatalf("error %v", err)
	}
}

func TestTranscriptionStreams(t *testing.T) {
	s, _ := audioTestServer(t, "lfm3-audio", true, func(n int, _ string) string {
		return fmt.Sprintf("part%d", n)
	})
	w := serveAudioText(s, audioTextRequest(t, "/v1/audio/transcriptions",
		map[string]string{"model": "speech-test", "stream": "true"}, testSpeech(40, 16000)))
	if w.Code != http.StatusOK || w.Header().Get("Content-Type") != "text/event-stream" {
		t.Fatalf("status %d, content type %q", w.Code, w.Header().Get("Content-Type"))
	}
	want := `data: {"delta":"part1","type":"transcript.text.delta"}` + "\n\n" +
		`data: {"delta":" part2","type":"transcript.text.delta"}` + "\n\n" +
		`data: {"text":"part1 part2","type":"transcript.text.done"}` + "\n\n"
	if w.Body.String() != want {
		t.Fatalf("got %q, want %q", w.Body, want)
	}
}

func TestTranscriptionRequiresAudioInput(t *testing.T) {
	s, requests := audioTestServer(t, "lfm3", false, func(int, string) string { return "x" })
	w := serveAudioText(s, audioTextRequest(t, "/v1/audio/transcriptions",
		map[string]string{"model": "speech-test"}, testSpeech(1, 16000)))
	if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), "does not support audio input") {
		t.Fatalf("status %d: %s", w.Code, w.Body)
	}
	if len(*requests) != 0 {
		t.Fatalf("ran %d generations", len(*requests))
	}
}

func TestTranscriptionModelNotFound(t *testing.T) {
	s, _ := audioTestServer(t, "lfm3-audio", true, func(int, string) string { return "x" })
	w := serveAudioText(s, audioTextRequest(t, "/v1/audio/transcriptions",
		map[string]string{"model": "missing"}, testSpeech(1, 16000)))
	if w.Code != http.StatusNotFound {
		t.Fatalf("status %d: %s", w.Code, w.Body)
	}
}
