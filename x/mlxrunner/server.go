//go:build mlx

package mlxrunner

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strconv"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	"github.com/ollama/ollama/x/mlxrunner/sample"
	"github.com/ollama/ollama/x/models/qwen3_5"
)

type samplingConfig struct {
	temperature      float32
	topP             float32
	minP             float32
	topK             int
	repeatLastN      int
	repeatPenalty    float32
	presencePenalty  float32
	frequencyPenalty float32
}

func defaultSamplingConfig(m base.Model, think *bool) samplingConfig {
	if _, ok := m.(*qwen3_5.Model); ok {
		cfg := samplingConfig{
			temperature:      1.0,
			topP:             0.95,
			minP:             0.0,
			topK:             20,
			repeatLastN:      64,
			repeatPenalty:    1.0,
			presencePenalty:  1.5,
			frequencyPenalty: 0.0,
		}
		if think != nil && !*think {
			cfg.temperature = 0.7
			cfg.topP = 0.8
		}
		return cfg
	}

	opts := api.DefaultOptions()
	return samplingConfig{
		temperature:      opts.Temperature,
		topP:             opts.TopP,
		minP:             opts.MinP,
		topK:             opts.TopK,
		repeatLastN:      opts.RepeatLastN,
		repeatPenalty:    opts.RepeatPenalty,
		presencePenalty:  opts.PresencePenalty,
		frequencyPenalty: opts.FrequencyPenalty,
	}
}

func resolveSamplingConfig(m base.Model, req Request) samplingConfig {
	cfg := defaultSamplingConfig(m, req.Think)

	if req.Options.Temperature != nil {
		cfg.temperature = *req.Options.Temperature
	}
	if req.Options.TopP != nil {
		cfg.topP = *req.Options.TopP
	}
	if req.Options.MinP != nil {
		cfg.minP = *req.Options.MinP
	}
	if req.Options.TopK != nil {
		cfg.topK = *req.Options.TopK
	}
	if req.Options.RepeatLastN != nil {
		cfg.repeatLastN = *req.Options.RepeatLastN
	}
	if req.Options.RepeatPenalty != nil {
		cfg.repeatPenalty = *req.Options.RepeatPenalty
	}
	if req.Options.PresencePenalty != nil {
		cfg.presencePenalty = *req.Options.PresencePenalty
	}
	if req.Options.FrequencyPenalty != nil {
		cfg.frequencyPenalty = *req.Options.FrequencyPenalty
	}

	return cfg
}

func Execute(args []string) error {
	slog.SetDefault(logutil.NewLogger(os.Stderr, envconfig.LogLevel()))

	if err := mlx.CheckInit(); err != nil {
		return fmt.Errorf("MLX not available: %w", err)
	}

	var (
		modelName string
		port      int
	)

	flagSet := flag.NewFlagSet("mlxrunner", flag.ExitOnError)
	flagSet.StringVar(&modelName, "model", "", "Model name")
	flagSet.IntVar(&port, "port", 0, "Port to listen on")
	_ = flagSet.Bool("verbose", false, "Enable debug logging")
	flagSet.Parse(args)

	runner := Runner{
		Requests: make(chan Request),
	}

	if err := runner.Load(modelName); err != nil {
		return err
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/status", func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewEncoder(w).Encode(statusResponse{
			Status:        0,
			Progress:      100,
			ContextLength: runner.contextLength,
			Memory:        uint64(mlx.ActiveMemory() + mlx.CacheMemory()),
		}); err != nil {
			slog.Error("Failed to encode response", "error", err)
			http.Error(w, "Internal Server Error", http.StatusInternalServerError)
			return
		}
	})

	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case "POST":
			fallthrough
		case "GET":
			if err := json.NewEncoder(w).Encode(map[string]any{
				"Success": true,
			}); err != nil {
				slog.Error("Failed to encode response", "error", err)
				http.Error(w, "Internal Server Error", http.StatusInternalServerError)
				return
			}
		case "DELETE":
			// TODO: cleanup model and cache
		}
	})

	mux.HandleFunc("POST /v1/completions", func(w http.ResponseWriter, r *http.Request) {
		request := Request{Responses: make(chan CompletionResponse)}

		if err := json.NewDecoder(r.Body).Decode(&request.TextCompletionsRequest); err != nil {
			slog.Error("Failed to decode request", "error", err)
			http.Error(w, "Bad Request", http.StatusBadRequest)
			return
		}

		request.Options.MaxTokens = cmp.Or(request.Options.MaxTokens, request.Options.NumPredict)

		sampling := resolveSamplingConfig(runner.Model, request)

		request.Pipeline = runner.TextGenerationPipeline
		request.Sampler = sample.New(
			sampling.temperature,
			sampling.topP,
			sampling.minP,
			sampling.topK,
			sampling.repeatLastN,
			sampling.repeatPenalty,
			sampling.presencePenalty,
			sampling.frequencyPenalty,
		)

		var cancel context.CancelFunc
		request.Ctx, cancel = context.WithCancel(r.Context())
		defer cancel()

		select {
		case <-r.Context().Done():
			return
		case runner.Requests <- request:
		}

		w.Header().Set("Content-Type", "application/jsonl")
		w.WriteHeader(http.StatusOK)
		enc := json.NewEncoder(w)
		for {
			select {
			case <-r.Context().Done():
				return
			case response, ok := <-request.Responses:
				if !ok {
					return
				}

				if err := enc.Encode(response); err != nil {
					slog.Error("Failed to encode response", "error", err)
					return
				}

				if f, ok := w.(http.Flusher); ok {
					f.Flush()
				}
			}
		}
	})

	mux.HandleFunc("POST /v1/tokenize", func(w http.ResponseWriter, r *http.Request) {
		var b bytes.Buffer
		if _, err := io.Copy(&b, r.Body); err != nil {
			slog.Error("Failed to read request body", "error", err)
			http.Error(w, "Bad Request", http.StatusBadRequest)
			return
		}

		tokens := runner.Tokenizer.Encode(b.String(), true)

		if err := json.NewEncoder(w).Encode(tokens); err != nil {
			slog.Error("Failed to encode response", "error", err)
			http.Error(w, "Internal Server Error", http.StatusInternalServerError)
			return
		}
	})

	for source, target := range map[string]string{
		"GET /health":      "/v1/status",
		"POST /load":       "/v1/models",
		"POST /completion": "/v1/completions",
	} {
		mux.Handle(source, http.RedirectHandler(target, http.StatusPermanentRedirect))
	}

	return runner.Run("127.0.0.1", strconv.Itoa(port), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		recorder := &statusRecorder{ResponseWriter: w, code: http.StatusOK}
		t := time.Now()
		mux.ServeHTTP(recorder, r)

		var level slog.Level
		switch {
		case recorder.code >= 500:
			level = slog.LevelError
		case recorder.code >= 400:
			level = slog.LevelWarn
		case recorder.code >= 300:
			return
		}

		slog.Log(r.Context(), level, "ServeHTTP", "method", r.Method, "path", r.URL.Path, "took", time.Since(t), "status", recorder.Status())
	}))
}

type statusRecorder struct {
	http.ResponseWriter
	code int
}

func (w *statusRecorder) WriteHeader(code int) {
	w.code = code
	w.ResponseWriter.WriteHeader(code)
}

func (w *statusRecorder) Status() string {
	return strconv.Itoa(w.code) + " " + http.StatusText(w.code)
}

func (w *statusRecorder) Flush() {
	if f, ok := w.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}
