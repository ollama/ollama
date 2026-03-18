package ortrunner

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"strconv"
	"time"

	"golang.org/x/sync/errgroup"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/x/ortrunner/oga"
)

// Execute is the subprocess entry point for the ORT GenAI runner.
func Execute(args []string) error {
	slog.SetDefault(logutil.NewLogger(os.Stderr, envconfig.LogLevel()))

	if err := oga.CheckInit(); err != nil {
		return fmt.Errorf("ORT GenAI not available: %w", err)
	}

	var (
		modelDir string
		port     int
	)

	flagSet := flag.NewFlagSet("ortrunner", flag.ExitOnError)
	flagSet.StringVar(&modelDir, "model", "", "Path to ONNX model directory")
	flagSet.IntVar(&port, "port", 0, "Port to listen on")
	_ = flagSet.Bool("verbose", false, "Enable debug logging")
	flagSet.Parse(args)

	runner := Runner{
		Requests: make(chan Request),
	}
	defer runner.Close()

	slog.Info("loading ORT GenAI model", "dir", modelDir)
	if err := runner.Load(modelDir); err != nil {
		return fmt.Errorf("failed to load model: %w", err)
	}

	mux := http.NewServeMux()

	// Health / status endpoint
	mux.HandleFunc("GET /v1/status", func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewEncoder(w).Encode(statusResponse{
			Status:   0,
			Progress: 100,
		}); err != nil {
			slog.Error("failed to encode status", "error", err)
			http.Error(w, "Internal Server Error", http.StatusInternalServerError)
		}
	})

	// Model management endpoint
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{"Success": true})
	})

	// Completion endpoint — streaming JSONL
	completionHandler := func(w http.ResponseWriter, r *http.Request) {
		request := Request{Responses: make(chan CompletionResponse)}

		if err := json.NewDecoder(r.Body).Decode(&request.TextCompletionsRequest); err != nil {
			slog.Error("failed to decode request", "error", err)
			http.Error(w, "Bad Request", http.StatusBadRequest)
			return
		}

		request.Options.MaxTokens = cmp.Or(request.Options.MaxTokens, request.Options.NumPredict)

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
					slog.Error("failed to encode response", "error", err)
					return
				}
				if f, ok := w.(http.Flusher); ok {
					f.Flush()
				}
			}
		}
	}
	mux.HandleFunc("POST /v1/completions", completionHandler)
	mux.HandleFunc("POST /completion", completionHandler)

	// Tokenize endpoint
	mux.HandleFunc("POST /v1/tokenize", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Content string `json:"content"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, "Bad Request", http.StatusBadRequest)
			return
		}

		tokens, err := runner.tokenizer.Encode(body.Content)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		intTokens := make([]int, len(tokens))
		for i, t := range tokens {
			intTokens[i] = int(t)
		}

		json.NewEncoder(w).Encode(intTokens)
	})

	// Redirects for compatibility with the standard runner protocol
	for source, target := range map[string]string{
		"GET /health": "/v1/status",
		"POST /load":  "/v1/models",
	} {
		mux.Handle(source, http.RedirectHandler(target, http.StatusPermanentRedirect))
	}

	// Run the request processing loop and HTTP server
	g, ctx := errgroup.WithContext(context.Background())

	g.Go(func() error {
		for {
			select {
			case <-ctx.Done():
				return nil
			case request := <-runner.Requests:
				if err := runner.Generate(request.Ctx, request); err != nil {
					slog.Error("generation failed", "error", err)
					var statusErr api.StatusError
					if !errors.As(err, &statusErr) {
						statusErr = api.StatusError{
							StatusCode:   http.StatusInternalServerError,
							ErrorMessage: err.Error(),
						}
					}
					select {
					case request.Responses <- CompletionResponse{Error: &statusErr}:
					case <-request.Ctx.Done():
					}
				}
				close(request.Responses)
			}
		}
	})

	g.Go(func() error {
		addr := net.JoinHostPort("127.0.0.1", strconv.Itoa(port))
		slog.Info("ORT GenAI runner listening", "addr", addr)
		return http.ListenAndServe(addr, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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
	})

	return g.Wait()
}

type statusResponse struct {
	Status   int    `json:"status"`
	Progress int    `json:"progress"`
	Memory   uint64 `json:"memory,omitempty"`
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
