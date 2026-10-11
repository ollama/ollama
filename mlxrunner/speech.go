package mlxrunner

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"net/http"

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

	streaming := false
	err := r.mlxThread.Do(req.Context(), func() error {
		defer mlx.ClearCache()
		return speaker.Speak(req.Context(), input, func(samples []float32) error {
			if !streaming {
				w.Header().Set("Content-Type", "audio/pcm")
				streaming = true
			}
			if _, err := w.Write(pcm16(samples)); err != nil {
				return err
			}
			return http.NewResponseController(w).Flush()
		})
	})
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

func pcm16(samples []float32) []byte {
	b := make([]byte, 2*len(samples))
	for i, s := range samples {
		v := math.Round(float64(max(-1, min(1, s))) * math.MaxInt16)
		binary.LittleEndian.PutUint16(b[2*i:], uint16(int16(v)))
	}
	return b
}
