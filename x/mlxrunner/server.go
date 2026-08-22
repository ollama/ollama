package mlxrunner

import (
	"bytes"
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

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/x/imagegen/manifest"
	"github.com/ollama/ollama/x/internal/mlxthread"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/sample"
)

func Execute(args []string) error {
	slog.SetDefault(logutil.NewLogger(os.Stderr, envconfig.LogLevel()))

	var (
		modelName string
		port      int
	)

	flagSet := flag.NewFlagSet("mlxrunner", flag.ExitOnError)
	flagSet.StringVar(&modelName, "model", "", "Model name")
	flagSet.IntVar(&port, "port", 0, "Port to listen on")
	_ = flagSet.Bool("verbose", false, "Enable debug logging")
	flagSet.Parse(args)

	worker, err := mlxthread.Start("mlxrunner", func() error {
		if err := mlx.CheckInit(); err != nil {
			return fmt.Errorf("MLX not available: %w", err)
		}

		if mlx.GPUIsAvailable() {
			mlx.SetDefaultDeviceGPU()
			slog.Info("MLX engine initialized", "MLX version", mlx.Version(), "device", "gpu")
		} else {
			slog.Info("MLX engine initialized", "MLX version", mlx.Version(), "device", "cpu")
		}

		return nil
	})
	if err != nil {
		return err
	}
	defer worker.Stop(context.Background(), func() {
		mlx.Sweep()
		mlx.ClearCache()
	})
	runnerCtx, cancelRunner := context.WithCancel(context.Background())
	defer cancelRunner()

	runner := Runner{
		Requests:  make(chan Request),
		mlxThread: worker,
	}

	state := newLoadState()

	load := func(ctx context.Context) error {
		if err := worker.Do(ctx, func() error {
			return runner.Load(modelName, state.SetProgress)
		}); err != nil {
			return err
		}
		state.MarkReady(runner.contextLength)
		return nil
	}

	readMemory := func() (uint64, error) {
		return uint64(mlx.ActiveMemory() + mlx.CacheMemory()), nil
	}
	// Weight bytes from the manifest stand in until the model is loaded and the
	// MLX worker is free to report real usage. Reporting zero instead would
	// clobber the parent's own estimate, which it reads before the load ends.
	modelManifest, err := manifest.LoadManifest(modelName)
	if err != nil {
		return err
	}
	loadingMemory := uint64(modelManifest.TotalTensorSize())

	memoryCache := newStatusMemoryCache(
		runnerCtx,
		loadingMemory,
		time.Time{},
		statusMemoryRefreshWait,
		func() (uint64, error) {
			return mlxthread.Call(runnerCtx, worker, readMemory)
		},
	)

	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/status", func(w http.ResponseWriter, r *http.Request) {
		status := state.Status()

		memory := loadingMemory
		if status == llm.ServerStatusReady {
			memory = memoryCache.Memory()
		}

		if err := json.NewEncoder(w).Encode(statusResponse{
			Status:        status,
			Progress:      state.Progress(),
			ContextLength: state.ContextLength(),
			Memory:        memory,
		}); err != nil {
			slog.Error("Failed to encode response", "error", err)
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
		// Acquires MarkReady: everything below reads what the load wrote.
		if state.Status() != llm.ServerStatusReady {
			http.Error(w, "model is still loading", http.StatusServiceUnavailable)
			return
		}

		request := Request{Responses: make(chan CompletionResponse)}

		if err := json.NewDecoder(r.Body).Decode(&request.CompletionRequest); err != nil {
			slog.Error("Failed to decode request", "error", err)
			http.Error(w, "Bad Request", http.StatusBadRequest)
			return
		}

		request.Pipeline = runner.TextGenerationPipeline
		request.SamplerOpts = sample.Options{
			Temperature:      request.Options.Temperature,
			TopP:             request.Options.TopP,
			MinP:             request.Options.MinP,
			TopK:             request.Options.TopK,
			RepeatLastN:      request.Options.RepeatLastN,
			RepeatPenalty:    request.Options.RepeatPenalty,
			PresencePenalty:  request.Options.PresencePenalty,
			FrequencyPenalty: request.Options.FrequencyPenalty,
			Seed:             request.Options.Seed,
			UseSeed:          request.Options.Seed >= 0,
			Logprobs:         request.Logprobs,
			TopLogprobs:      request.TopLogprobs,
		}

		if err := runner.Prepare(&request); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

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
		if state.Status() != llm.ServerStatusReady {
			http.Error(w, "model is still loading", http.StatusServiceUnavailable)
			return
		}

		var b bytes.Buffer
		if _, err := io.Copy(&b, r.Body); err != nil {
			slog.Error("Failed to read request body", "error", err)
			http.Error(w, "Bad Request", http.StatusBadRequest)
			return
		}

		tokens := runner.Tokenizer.Encode(b.String(), runner.Tokenizer.AddBOS())

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
		case r.URL.Path == "/v1/status":
			// Polled several times a second for the duration of a model
			// load. Failures are already covered by the cases above.
			level = logutil.LevelTrace
		}

		slog.Log(r.Context(), level, "ServeHTTP", "method", r.Method, "path", r.URL.Path, "took", time.Since(t), "status", recorder.Status())
	}), load)
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
