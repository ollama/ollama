package server

import (
	"cmp"
	"encoding/binary"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/openai"
	"github.com/ollama/ollama/types/model"
)

const speechSampleRate = 24000

// SpeechHandler serves the OpenAI-compatible POST /v1/audio/speech, streaming
// audio as the model generates it.
func (s *Server) SpeechHandler(c *gin.Context) {
	var req struct {
		Model          string   `json:"model"`
		Input          string   `json:"input"`
		Voice          string   `json:"voice"`
		ResponseFormat string   `json:"response_format"`
		Speed          *float64 `json:"speed"`
		Instructions   string   `json:"instructions"`
		StreamFormat   string   `json:"stream_format"`
	}
	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, 1<<20)
	if err := c.ShouldBindJSON(&req); err != nil {
		speechError(c, http.StatusBadRequest, err.Error())
		return
	}
	format := cmp.Or(req.ResponseFormat, "wav")
	switch {
	case strings.TrimSpace(req.Model) == "":
		speechError(c, http.StatusBadRequest, "model is required")
		return
	case strings.TrimSpace(req.Input) == "":
		speechError(c, http.StatusBadRequest, "input is required")
		return
	case utf8.RuneCountInString(req.Input) > 4096:
		speechError(c, http.StatusBadRequest, "input must be at most 4096 characters")
		return
	case format != "wav" && format != "pcm":
		speechError(c, http.StatusBadRequest, fmt.Sprintf("response_format %q is not supported, use wav or pcm", format))
		return
	case req.Speed != nil && *req.Speed != 1:
		speechError(c, http.StatusBadRequest, "only speed 1 is supported")
		return
	case req.Instructions != "":
		speechError(c, http.StatusBadRequest, "instructions are not supported")
		return
	case req.StreamFormat != "" && req.StreamFormat != "audio":
		speechError(c, http.StatusBadRequest, "only stream_format audio is supported")
		return
	}

	r, _, _, err := s.scheduleRunner(c.Request.Context(), req.Model, "", []model.Capability{model.CapabilitySpeech}, nil, nil, nil)
	speaker, ok := r.(llm.Speaker)
	switch {
	case errors.Is(err, errCapabilities), err == nil && !ok:
		speechError(c, http.StatusBadRequest, fmt.Sprintf("%q does not support speech", req.Model))
		return
	case err != nil:
		status, message := scheduleError(req.Model, err)
		speechError(c, status, message)
		return
	}

	streamSpeech(c, speaker, llm.SpeechRequest{Input: req.Input, Voice: req.Voice}, format)
}

func streamSpeech(c *gin.Context, speaker llm.Speaker, req llm.SpeechRequest, format string) {
	// Write through Gin for status/byte accounting, but bypass its Flush
	// method so ResponseController can return the underlying flush error.
	var writer http.ResponseWriter = c.Writer
	if w, ok := writer.(interface{ Unwrap() http.ResponseWriter }); ok {
		writer = w.Unwrap()
	}
	controller := http.NewResponseController(writer)
	defer controller.SetWriteDeadline(time.Time{})
	started, clientGone := false, false
	err := speaker.Speech(c.Request.Context(), req, func(pcm []byte) (err error) {
		// Every error here comes from writing to the client.
		defer func() { clientGone = err != nil }()
		if len(pcm) == 0 {
			return nil
		}
		// Bound a stalled client's writes so it cannot hold the model and
		// back up the runner indefinitely. Reset for each chunk, not request.
		if err := controller.SetWriteDeadline(time.Now().Add(30 * time.Second)); err != nil {
			return err
		}
		if !started {
			started = true
			c.Header("Content-Type", "audio/"+format)
			c.Status(http.StatusOK)
			if format == "wav" {
				if _, err := c.Writer.Write(wavHeader()); err != nil {
					return err
				}
			}
		}
		if _, err := c.Writer.Write(pcm); err != nil {
			return err
		}
		return controller.Flush()
	})
	if err == nil && !started {
		err = errors.New("speech model produced no audio")
	}
	if err == nil {
		return
	}
	if started {
		// Close the connection rather than end the stream, so clients don't
		// mistake partial audio for a complete response. A client that hangs
		// up is not a failure worth a warning.
		if !clientGone && c.Request.Context().Err() == nil {
			slog.Warn("speech stream failed", "error", err)
		}
		if conn, _, err := controller.Hijack(); err == nil {
			conn.Close()
		}
		return
	}
	status, message := http.StatusInternalServerError, err.Error()
	var statusErr api.StatusError
	if errors.As(err, &statusErr) {
		status, message = statusErr.StatusCode, statusErr.ErrorMessage
	}
	speechError(c, status, message)
}

func speechError(c *gin.Context, status int, message string) {
	c.AbortWithStatusJSON(status, openai.NewError(status, message))
}

// wavHeader describes 16-bit mono PCM of unknown length, as streamed WAV is.
func wavHeader() []byte {
	h := make([]byte, 44)
	copy(h[0:], "RIFF")
	binary.LittleEndian.PutUint32(h[4:], 0xffffffff)
	copy(h[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(h[16:], 16)
	binary.LittleEndian.PutUint16(h[20:], 1)
	binary.LittleEndian.PutUint16(h[22:], 1)
	binary.LittleEndian.PutUint32(h[24:], speechSampleRate)
	binary.LittleEndian.PutUint32(h[28:], speechSampleRate*2)
	binary.LittleEndian.PutUint16(h[32:], 2)
	binary.LittleEndian.PutUint16(h[34:], 16)
	copy(h[36:], "data")
	binary.LittleEndian.PutUint32(h[40:], 0xffffffff)
	return h
}
