package llm

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"mime"
	"net/http"
	"strconv"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/mlxrunner/model/audio"
	"github.com/ollama/ollama/model/renderers"
)

var _ Speaker = (*llamaServerRunner)(nil)

// Speech reads text aloud through llama-server's /tts, one generation per
// chunk. The request's SpeechPrompter, from the model's renderer, names the
// voices and splits the input the way the publisher documents.
func (s *llamaServerRunner) Speech(ctx context.Context, req SpeechRequest, fn func(pcm []byte) error) error {
	if req.Prompter == nil {
		return api.StatusError{StatusCode: http.StatusNotImplemented, ErrorMessage: "this model does not support speech"}
	}
	if len(s.launch.projectors) == 0 {
		return api.StatusError{StatusCode: http.StatusNotImplemented, ErrorMessage: "this model was created without its projector, which holds the speech codec; recreate it with the mmproj to use speech"}
	}
	voice, err := req.Prompter.SpeechVoice(req.Voice)
	if err != nil {
		return api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: err.Error()}
	}

	if err := s.sem.Acquire(ctx, 1); err != nil {
		return err
	}
	defer s.sem.Release(1)

	status, err := s.getServerStatusRetry(ctx)
	if err != nil {
		return err
	} else if status != ServerStatusReady {
		return fmt.Errorf("unexpected server status: %s", status)
	}

	// Chunking counts many overlapping pieces; each count is a /tokenize call.
	counts := map[string]int{}
	var countErr error
	count := func(text string) int {
		if n, ok := counts[text]; ok || countErr != nil {
			return n
		}
		tokens, err := s.tokenize(ctx, text, false)
		if err != nil {
			countErr = err
			return 0
		}
		counts[text] = len(tokens)
		return len(tokens)
	}
	chunks := req.Prompter.SpeechChunks(req.Input, count)
	if countErr != nil {
		return fmt.Errorf("count speech tokens: %w", countErr)
	}
	if len(chunks) == 0 {
		return api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: "input has no text to speak"}
	}
	for _, chunk := range chunks {
		if err := s.speakChunk(ctx, voice, chunk, fn); err != nil {
			return err
		}
	}
	return nil
}

type llamaServerTTSRequest struct {
	Input          string `json:"input"`
	Voice          string `json:"voice"`
	Lang           string `json:"lang"`
	ResponseFormat string `json:"response_format"`
	Stream         bool   `json:"stream"`
	NPredict       int    `json:"n_predict"`
}

// speakChunk generates one chunk; llama-server renders the model's "tts"
// template from the voice and language.
func (s *llamaServerRunner) speakChunk(ctx context.Context, voice renderers.SpeechVoice, chunk renderers.SpeechChunk, fn func([]byte) error) error {
	var body bytes.Buffer
	enc := json.NewEncoder(&body)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(llamaServerTTSRequest{
		Input:          chunk.Text,
		Voice:          voice.Voice,
		Lang:           voice.Language,
		ResponseFormat: "pcm",
		Stream:         true,
		NPredict:       chunk.MaxFrames,
	}); err != nil {
		return err
	}
	r, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d/tts", s.port), &body)
	if err != nil {
		return err
	}
	r.Header.Set("Content-Type", "application/json")
	res, err := s.httpClient().Do(r)
	if err != nil {
		if errors.Is(err, context.Canceled) {
			return err
		}
		slog.Error("llama-server tts error", "error", err)
		if msg := s.lastErrMsg(); msg != "" {
			return fmt.Errorf("model runner has unexpectedly stopped, this may be due to resource limitations or an internal error, check ollama server logs for details: %s", msg)
		}
		return errors.New("model runner has unexpectedly stopped, this may be due to resource limitations or an internal error, check ollama server logs for details")
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		data, err := io.ReadAll(res.Body)
		if err != nil {
			return fmt.Errorf("failed reading llama-server error response: %w", err)
		}
		return api.StatusError{StatusCode: res.StatusCode, ErrorMessage: s.statusErrorMessage(data)}
	}
	if err := checkLlamaServerPCM(res.Header.Get("Content-Type")); err != nil {
		return err
	}
	return streamPCM16(res.Body, fn)
}

// checkLlamaServerPCM accepts the float32 mono PCM llama-server streams for
// response_format pcm, at the rate Speaker promises.
func checkLlamaServerPCM(contentType string) error {
	mediaType, params, err := mime.ParseMediaType(contentType)
	if err != nil {
		return fmt.Errorf("llama-server tts content type %q: %w", contentType, err)
	}
	rate, _ := strconv.Atoi(params["rate"])
	if mediaType != "audio/pcm" || params["encoding"] != "float" || params["bits"] != "32" || rate != SpeechSampleRate {
		return fmt.Errorf("llama-server tts returned %q, want 32-bit float PCM at %d Hz", contentType, SpeechSampleRate)
	}
	return nil
}

// streamPCM16 converts little-endian float32 samples to 16-bit PCM as they
// arrive, carrying a sample split across reads to the next one.
func streamPCM16(r io.Reader, fn func([]byte) error) error {
	buf := make([]byte, 32<<10)
	pending := 0
	for {
		n, err := r.Read(buf[pending:])
		n += pending
		whole := n - n%4
		if whole > 0 {
			pcm := make([]byte, whole/2)
			for i := 0; i < whole; i += 4 {
				sample := math.Float32frombits(binary.LittleEndian.Uint32(buf[i:]))
				binary.LittleEndian.PutUint16(pcm[i/2:], uint16(audio.PCM16(sample)))
			}
			if err := fn(pcm); err != nil {
				return err
			}
		}
		pending = copy(buf, buf[whole:n])
		if errors.Is(err, io.EOF) {
			if pending != 0 {
				return fmt.Errorf("llama-server tts stream ended mid-sample")
			}
			return nil
		} else if err != nil {
			return err
		}
	}
}
