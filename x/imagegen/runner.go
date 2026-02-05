//go:build mlx

// Package imagegen provides a unified MLX runner for both LLM and image generation models.
package imagegen

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/x/imagegen/mlx"
)

// Execute is the entry point for the unified MLX runner subprocess.
func Execute(args []string) error {
	// Set up logging with appropriate level from environment
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: envconfig.LogLevel()})))

	fs := flag.NewFlagSet("mlx-runner", flag.ExitOnError)
	modelName := fs.String("model", "", "path to model")
	port := fs.Int("port", 0, "port to listen on")

	if err := fs.Parse(args); err != nil {
		return err
	}

	if *modelName == "" {
		return fmt.Errorf("--model is required")
	}
	if *port == 0 {
		return fmt.Errorf("--port is required")
	}

	// Initialize MLX
	if err := mlx.InitMLX(); err != nil {
		slog.Error("unable to initialize MLX", "error", err)
		return err
	}
	slog.Info("MLX library initialized")

	// Detect model type from capabilities
	mode := detectModelMode(*modelName)
	slog.Info("starting mlx runner", "model", *modelName, "port", *port, "mode", mode)

	// Create and start server
	server, err := newServer(*modelName, *port, mode)
	if err != nil {
		return fmt.Errorf("failed to create server: %w", err)
	}

	// Set up HTTP handlers
	mux := http.NewServeMux()
	mux.HandleFunc("/health", server.healthHandler)
	mux.HandleFunc("/completion", server.completionHandler)

	// LLM-specific endpoints
	if mode == ModeLLM {
		mux.HandleFunc("/tokenize", server.tokenizeHandler)
		mux.HandleFunc("/embedding", server.embeddingHandler)
	}

	httpServer := &http.Server{
		Addr:    fmt.Sprintf("127.0.0.1:%d", *port),
		Handler: mux,
	}

	// Handle shutdown
	done := make(chan struct{})
	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
		<-sigCh
		slog.Info("shutting down mlx runner")
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		httpServer.Shutdown(ctx)
		close(done)
	}()

	slog.Info("mlx runner listening", "addr", httpServer.Addr)
	if err := httpServer.ListenAndServe(); err != http.ErrServerClosed {
		return err
	}

	<-done
	return nil
}

// detectModelMode determines whether a model is an LLM or image generation model.
func detectModelMode(modelName string) ModelMode {
	// Check for image generation model by looking at model_index.json
	modelType := DetectModelType(modelName)
	if modelType != "" {
		// Known image generation model types
		switch modelType {
		case "ZImagePipeline", "FluxPipeline", "Flux2KleinPipeline":
			return ModeImageGen
		}
	}

	// Default to LLM mode for safetensors models without known image gen types
	return ModeLLM
}

// server holds the model and handles HTTP requests.
type server struct {
	mode      ModelMode
	modelName string
	port      int

	// Image generation model (when mode == ModeImageGen)
	imageModel ImageModel

	// LLM model (when mode == ModeLLM)
	llmModel *llmState
}

// newServer creates a new server instance and loads the appropriate model.
func newServer(modelName string, port int, mode ModelMode) (*server, error) {
	s := &server{
		mode:      mode,
		modelName: modelName,
		port:      port,
	}

	switch mode {
	case ModeImageGen:
		if err := s.loadImageModel(); err != nil {
			return nil, fmt.Errorf("failed to load image model: %w", err)
		}
	case ModeLLM:
		if err := s.loadLLMModel(); err != nil {
			return nil, fmt.Errorf("failed to load LLM model: %w", err)
		}
	}

	return s, nil
}

func (s *server) healthHandler(w http.ResponseWriter, r *http.Request) {
	resp := HealthResponse{Status: "ok"}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *server) completionHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req Request
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	switch s.mode {
	case ModeImageGen:
		s.handleImageCompletion(w, r, req)
	case ModeLLM:
		s.handleLLMCompletion(w, r, req)
	}
}

func (s *server) tokenizeHandler(w http.ResponseWriter, r *http.Request) {
	if s.llmModel == nil {
		http.Error(w, "LLM model not loaded", http.StatusInternalServerError)
		return
	}

	var req struct {
		Content string `json:"content"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	tok := s.llmModel.model.Tokenizer()
	tokens := tok.Encode(req.Content, false)

	// Convert int32 to int for JSON response
	intTokens := make([]int, len(tokens))
	for i, t := range tokens {
		intTokens[i] = int(t)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string][]int{"tokens": intTokens})
}

func (s *server) embeddingHandler(w http.ResponseWriter, r *http.Request) {
	http.Error(w, "embeddings not yet implemented for MLX models", http.StatusNotImplemented)
}
