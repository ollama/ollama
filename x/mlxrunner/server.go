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

	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/x/internal/mlxthread"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/sample"
)

// planWired returns the MLX wired (resident) cap in bytes for the given free RAM,
// held below free memory so a model larger than free RAM pages from disk instead
// of OOM-killing the runner.
func planWired(free int) int {
	return free * 4 / 5
}

// planCache returns the MLX cache limit in bytes for a model of modelSize against
// the free RAM the host had at startup. The pool is tightened only when the model
// did not fit under the wired cap (so it pages); a return of 0 leaves MLX's
// default, so a resident model keeps full buffer reuse.
func planCache(modelSize, startupFree int) int {
	if modelSize > planWired(startupFree) {
		return startupFree / 5
	}
	return 0
}

// configureMLXMemory caps MLX's wired footprint to the host's free RAM at startup
// so that a model the scheduler has already admitted pages instead of MLX wiring
// its whole resident set and OOM-killing the runner. This is a post-admission
// safeguard only; it does not change the Client.Load fit check that governs
// whether an oversized model is loaded at all. It returns the free RAM read
// (0 if unavailable) for the post-load cache decision. Must run on the MLX thread.
func configureMLXMemory() int {
	mem, err := discover.GetCPUMem()
	free := int(mem.FreeMemory)
	if err != nil || free <= 0 {
		return 0 // without a free-memory reading, keep MLX's defaults
	}
	wired := planWired(free)
	if wired > 0 {
		mlx.SetWiredLimit(wired)
	}
	slog.Info("capped MLX wired memory",
		"free", format.HumanBytes2(uint64(free)),
		"wired", format.HumanBytes2(uint64(wired)))
	return free
}

// tuneMLXMemory tightens MLX's buffer cache once the model is loaded and its size
// is known — but only when the model did not fit in the startup free RAM (so it
// pages). A model that fit keeps MLX's default pool. Must run on the MLX thread.
func tuneMLXMemory(startupFree int) {
	if startupFree <= 0 {
		return
	}
	modelSize := mlx.ActiveMemory()
	cache := planCache(modelSize, startupFree)
	if cache > 0 {
		mlx.SetCacheLimit(cache)
		slog.Info("tightened MLX cache (model exceeds free RAM)",
			"model", format.HumanBytes2(uint64(modelSize)),
			"free", format.HumanBytes2(uint64(startupFree)),
			"cache", format.HumanBytes2(uint64(cache)))
	}
}

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

	var startupFree int // free RAM at engine init, for the post-load cache decision
	worker, err := mlxthread.Start("mlxrunner", func() error {
		if err := mlx.CheckInit(); err != nil {
			return fmt.Errorf("MLX not available: %w", err)
		}

		if mlx.GPUIsAvailable() {
			mlx.SetDefaultDeviceGPU()
			startupFree = configureMLXMemory()
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

	if err := worker.Do(context.Background(), func() error {
		if err := runner.Load(modelName); err != nil {
			return err
		}
		tuneMLXMemory(startupFree)
		return nil
	}); err != nil {
		return err
	}

	readMemory := func() (uint64, error) {
		return uint64(mlx.ActiveMemory() + mlx.CacheMemory()), nil
	}
	initialMemory, err := mlxthread.Call(context.Background(), worker, readMemory)
	if err != nil {
		return err
	}
	memoryCache := newStatusMemoryCache(
		runnerCtx,
		initialMemory,
		time.Now(),
		statusMemoryRefreshWait,
		func() (uint64, error) {
			return mlxthread.Call(runnerCtx, worker, readMemory)
		},
	)

	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/status", func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewEncoder(w).Encode(statusResponse{
			Status:        0,
			Progress:      100,
			ContextLength: runner.contextLength,
			Memory:        memoryCache.Memory(),
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
