package server

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"strconv"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/gin-gonic/gin"
	"golang.org/x/text/language"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/mlxrunner/model/audio"
	"github.com/ollama/ollama/model/renderers"
	"github.com/ollama/ollama/openai"
	"github.com/ollama/ollama/types/model"
)

// Long recordings are split into pieces, so these bound memory rather than
// output: the decoded float32 samples of a request are held while it runs.
// The duration cap binds for compressed-rate WAV (16 kHz 8-bit mono); higher
// rates reach the upload cap first.
const (
	maxAudioUploadBytes = 512 << 20
	maxAudioSeconds     = 2 * 60 * 60
)

// TranscriptionHandler serves the OpenAI-compatible POST /v1/audio/transcriptions.
func (s *Server) TranscriptionHandler(c *gin.Context) {
	s.audioTextHandler(c, renderers.AudioTranscribe)
}

// TranslationHandler serves the OpenAI-compatible POST /v1/audio/translations.
// Translations default to English like OpenAI's; to_language picks another
// target.
func (s *Server) TranslationHandler(c *gin.Context) {
	s.audioTextHandler(c, renderers.AudioTranslate)
}

// audioTextHandler turns speech into text with one chat generation per clip,
// so long recordings stay within what the model was trained on and each
// generation's output stays bounded.
func (s *Server) audioTextHandler(c *gin.Context, task renderers.AudioTask) {
	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, maxAudioUploadBytes)
	if err := c.Request.ParseMultipartForm(32 << 20); err != nil {
		if tooLarge := new(http.MaxBytesError); errors.As(err, &tooLarge) {
			openAIError(c, http.StatusRequestEntityTooLarge, fmt.Sprintf("audio uploads are limited to %d MB", maxAudioUploadBytes>>20))
			return
		}
		openAIError(c, http.StatusBadRequest, "failed to parse multipart form: "+err.Error())
		return
	}
	form, err := parseAudioTextForm(c.Request, task)
	if err != nil {
		openAIError(c, http.StatusBadRequest, err.Error())
		return
	}

	ref, err := parseAndValidateModelRef(form.model)
	if err != nil {
		openAIError(c, http.StatusBadRequest, err.Error())
		return
	}
	// Cloud and remote models are proxied by the chat handler, which knows
	// nothing of their renderer here; they get the generic prompt.
	var rendererName string
	if ref.Source != modelSourceCloud {
		m, err := s.getExistingModelForRunner(ref.Name.String(), "")
		if err != nil {
			status, message := scheduleError(form.model, err)
			openAIError(c, status, message)
			return
		}
		if m.Config.RemoteHost == "" {
			if err := m.CheckCapabilities(model.CapabilityAudio); err != nil {
				openAIError(c, http.StatusBadRequest, fmt.Sprintf("%q does not support audio input", form.model))
				return
			}
			rendererName = resolveRendererName(m)
		}
	}
	prompter := renderers.AudioPrompterForRenderer(rendererName)

	clips, err := splitAudio(form.audio, prompter.AudioClipLimit())
	if err != nil {
		openAIError(c, http.StatusBadRequest, err.Error())
		return
	}
	// Split pieces hold their own samples; drop the upload.
	form.audio = nil

	out := audioTextWriter{c: c, req: form.request, format: form.responseFormat, stream: form.stream}
	for i, clip := range clips {
		req := form.request
		req.Audio = clip.wav()
		prompt, err := prompter.AudioPrompt(req)
		if err != nil {
			out.fail(http.StatusBadRequest, err.Error())
			return
		}

		options := map[string]any{"temperature": form.temperature}
		if clip.duration > 0 {
			options["num_predict"] = maxAudioTextTokens(clip.duration)
		}
		var text strings.Builder
		var doneReason string
		err = s.chat(c.Request.Context(), api.ChatRequest{
			Model:    form.model,
			Messages: prompt.Messages,
			Think:    prompt.Think,
			Options:  options,
		}, func(r api.ChatResponse) error {
			text.WriteString(r.Message.Content)
			doneReason = cmp.Or(r.DoneReason, doneReason)
			return nil
		})
		if err == nil && doneReason == "length" {
			slog.Warn("speech-to-text output hit its length limit", "model", form.model, "clip", i, "clip_seconds", clip.duration.Seconds(), "num_predict", options["num_predict"])
		}
		if err != nil {
			status, message := http.StatusInternalServerError, err.Error()
			var statusErr api.StatusError
			if errors.As(err, &statusErr) {
				status, message = statusErr.StatusCode, statusErr.ErrorMessage
			}
			out.fail(status, message)
			return
		}
		if err := out.add(prompter.AudioText(req, text.String()), clip.duration); err != nil {
			return
		}
	}
	out.done()
}

type audioTextForm struct {
	model          string
	audio          []byte
	request        renderers.AudioRequest
	responseFormat string
	temperature    float64
	stream         bool
}

func parseAudioTextForm(r *http.Request, task renderers.AudioTask) (audioTextForm, error) {
	f := audioTextForm{
		model:          r.FormValue("model"),
		responseFormat: cmp.Or(r.FormValue("response_format"), "json"),
		request:        renderers.AudioRequest{Task: task, Prompt: r.FormValue("prompt")},
	}
	if f.model == "" {
		return f, errors.New("model is required")
	}
	switch f.responseFormat {
	case "json", "text", "verbose_json":
	default:
		return f, fmt.Errorf("response_format %q is not supported, use json, text or verbose_json", f.responseFormat)
	}

	var err error
	if v := r.FormValue("temperature"); v != "" {
		if f.temperature, err = strconv.ParseFloat(v, 64); err != nil || math.IsNaN(f.temperature) || math.IsInf(f.temperature, 0) {
			return f, fmt.Errorf("invalid temperature %q", v)
		}
	}
	if v := r.FormValue("stream"); v != "" {
		if f.stream, err = strconv.ParseBool(v); err != nil {
			return f, fmt.Errorf("invalid stream %q", v)
		}
	}
	if f.request.Source, err = parseAudioLanguage("language", r.FormValue("language"), language.Und); err != nil {
		return f, err
	}
	switch to := r.FormValue("to_language"); {
	case task == renderers.AudioTranslate:
		if f.request.Target, err = parseAudioLanguage("to_language", to, language.English); err != nil {
			return f, err
		}
	case to != "":
		return f, errors.New("to_language is only supported for translations")
	}

	file, _, err := r.FormFile("file")
	if err != nil {
		return f, fmt.Errorf("file is required: %w", err)
	}
	defer file.Close()
	if f.audio, err = io.ReadAll(file); err != nil {
		return f, fmt.Errorf("failed to read audio file: %w", err)
	}
	if len(f.audio) == 0 {
		return f, errors.New("audio file is empty")
	}
	return f, nil
}

// parseAudioLanguage accepts ISO 639-1 codes and other BCP 47 tags ("pt-BR"),
// and English language names ("French") as earlier releases passed the field
// through to the prompt.
func parseAudioLanguage(field, value string, fallback language.Tag) (language.Tag, error) {
	if value == "" {
		return fallback, nil
	}
	if tag, err := language.Parse(value); err == nil && renderers.LanguageName(tag) != "" {
		return tag, nil
	}
	if tag, ok := renderers.LanguageForName(value); ok {
		return tag, nil
	}
	return language.Und, fmt.Errorf("unsupported %s %q, use an ISO 639-1 code such as \"en\"", field, value)
}

type audioClip struct {
	// data is the upload when it isn't split; split pieces keep samples and
	// are encoded only when sent, so one piece is encoded at a time.
	data    []byte
	samples []float32
	rate    int
	// duration is zero when the container isn't one the server decodes.
	duration time.Duration
}

func (c audioClip) wav() []byte {
	if c.data != nil {
		return c.data
	}
	return audio.EncodeWAV(c.samples, c.rate)
}

// splitAudio cuts audio longer than limit into clips at quiet points. Audio
// in containers the server doesn't decode passes through whole for the
// runner to handle.
func splitAudio(data []byte, limit time.Duration) ([]audioClip, error) {
	samples, rate, err := audio.DecodeLimit(data, maxAudioSeconds)
	if errors.Is(err, audio.ErrUnrecognized) {
		return []audioClip{{data: data}}, nil
	}
	if err != nil {
		return nil, err
	}
	duration := func(n int) time.Duration { return time.Duration(n) * time.Second / time.Duration(rate) }
	if limit <= 0 || duration(len(samples)) <= limit {
		return []audioClip{{data: data, duration: duration(len(samples))}}, nil
	}
	var clips []audioClip
	for _, piece := range audio.Split(samples, rate, int(limit/time.Second)) {
		clips = append(clips, audioClip{samples: piece, rate: rate, duration: duration(len(piece))})
	}
	return clips, nil
}

// maxAudioTextTokens bounds a generation over d of audio. Speech runs a few
// tokens per second in any script, so this only stops runaway repetition.
func maxAudioTextTokens(d time.Duration) int {
	return 64 + 32*int((d+time.Second-1)/time.Second)
}

// audioTextWriter writes clip texts as one OpenAI transcription response,
// streamed as server-sent events when requested.
type audioTextWriter struct {
	c        *gin.Context
	req      renderers.AudioRequest
	format   string
	stream   bool
	text     string
	segments []openai.TranscriptionSegment
	elapsed  time.Duration
	started  bool
}

// add appends the text of the next clip, which lasts d.
func (w *audioTextWriter) add(text string, d time.Duration) error {
	start := w.elapsed
	w.elapsed += d
	if text == "" {
		return nil
	}
	w.segments = append(w.segments, openai.TranscriptionSegment{
		ID: len(w.segments), Start: start.Seconds(), End: w.elapsed.Seconds(), Text: text,
	})
	delta := text
	if w.text != "" && needsSpace(w.text, text) {
		delta = " " + text
	}
	w.text += delta
	if !w.stream {
		return nil
	}
	return w.event(map[string]string{"type": "transcript.text.delta", "delta": delta})
}

func (w *audioTextWriter) done() {
	switch {
	case w.stream:
		_ = w.event(map[string]string{"type": "transcript.text.done", "text": w.text})
	case w.format == "text":
		w.c.Data(http.StatusOK, "text/plain; charset=utf-8", []byte(w.text))
	case w.format == "verbose_json":
		resp := openai.TranscriptionVerboseResponse{
			Task:     "transcribe",
			Duration: w.elapsed.Seconds(),
			Text:     w.text,
			Segments: w.segments,
		}
		// OpenAI reports languages by lowercase English name, and the output
		// language for translations.
		if w.req.Task == renderers.AudioTranslate {
			resp.Task = "translate"
			resp.Language = strings.ToLower(renderers.LanguageName(w.req.Target))
		} else if w.req.Source != language.Und {
			resp.Language = strings.ToLower(renderers.LanguageName(w.req.Source))
		}
		w.c.JSON(http.StatusOK, resp)
	default:
		w.c.JSON(http.StatusOK, openai.TranscriptionResponse{Text: w.text})
	}
}

func (w *audioTextWriter) fail(status int, message string) {
	if !w.started {
		openAIError(w.c, status, message)
		return
	}
	_ = w.event(openai.NewError(status, message))
}

func (w *audioTextWriter) event(v any) error {
	if !w.started {
		w.started = true
		w.c.Header("Content-Type", "text/event-stream")
		w.c.Header("Cache-Control", "no-cache")
		w.c.Status(http.StatusOK)
	}
	data, err := json.Marshal(v)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w.c.Writer, "data: %s\n\n", data); err != nil {
		return err
	}
	w.c.Writer.Flush()
	return nil
}

// needsSpace reports whether text joining prev needs a separating space;
// scripts written without spaces between words join directly.
func needsSpace(prev, next string) bool {
	last, _ := utf8.DecodeLastRuneInString(prev)
	first, _ := utf8.DecodeRuneInString(next)
	return !unspacedScript(last) || !unspacedScript(first)
}

func unspacedScript(r rune) bool {
	switch {
	case unicode.In(r, unicode.Han, unicode.Hiragana, unicode.Katakana, unicode.Thai, unicode.Lao, unicode.Khmer, unicode.Myanmar):
		return true
	case r >= 0x3000 && r <= 0x303f, r >= 0xff00 && r <= 0xffef:
		// CJK and fullwidth punctuation
		return true
	}
	return false
}

// chat runs req through the chat handler and passes each streamed response
// to fn, so internal callers get the same templating, scheduling and parsing
// as /api/chat.
func (s *Server) chat(ctx context.Context, req api.ChatRequest, fn func(api.ChatResponse) error) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	body, err := json.Marshal(req)
	if err != nil {
		return err
	}
	w := &chatResponseWriter{header: http.Header{}, status: http.StatusOK, fn: fn, cancel: cancel}
	c, _ := gin.CreateTestContext(w)
	c.Request, err = http.NewRequestWithContext(ctx, http.MethodPost, "/api/chat", bytes.NewReader(body))
	if err != nil {
		return err
	}
	c.Request.Header.Set("Content-Type", "application/json")
	s.ChatHandler(c)
	return w.result()
}

// chatResponseWriter decodes the chat handler's NDJSON stream.
type chatResponseWriter struct {
	header http.Header
	status int
	fn     func(api.ChatResponse) error
	cancel context.CancelFunc
	buf    []byte
	err    error
}

func (w *chatResponseWriter) Header() http.Header      { return w.header }
func (w *chatResponseWriter) WriteHeader(status int)   { w.status = status }
func (w *chatResponseWriter) Flush()                   {}
func (w *chatResponseWriter) CloseNotify() <-chan bool { return nil }
func (w *chatResponseWriter) Write(b []byte) (int, error) {
	w.buf = append(w.buf, b...)
	if w.status != http.StatusOK {
		return len(b), nil
	}
	for {
		i := bytes.IndexByte(w.buf, '\n')
		if i < 0 {
			return len(b), nil
		}
		line := w.buf[:i]
		w.buf = w.buf[i+1:]
		if w.err == nil && len(bytes.TrimSpace(line)) > 0 {
			w.err = w.line(line)
			if w.err != nil {
				w.cancel()
			}
		}
	}
}

func (w *chatResponseWriter) line(line []byte) error {
	var r struct {
		api.ChatResponse
		Error string `json:"error"`
	}
	if err := json.Unmarshal(line, &r); err != nil {
		return err
	}
	if r.Error != "" {
		return api.StatusError{StatusCode: http.StatusInternalServerError, ErrorMessage: r.Error}
	}
	return w.fn(r.ChatResponse)
}

func (w *chatResponseWriter) result() error {
	if w.status != http.StatusOK {
		var r struct {
			Error string `json:"error"`
		}
		message := strings.TrimSpace(string(w.buf))
		if json.Unmarshal(w.buf, &r) == nil && r.Error != "" {
			message = r.Error
		}
		return api.StatusError{StatusCode: w.status, ErrorMessage: message}
	}
	if w.err == nil && len(bytes.TrimSpace(w.buf)) > 0 {
		w.err = w.line(w.buf)
	}
	return w.err
}
