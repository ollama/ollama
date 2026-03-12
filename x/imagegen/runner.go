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

	// Detect model type from capabilities
	mode := detectModelMode(*modelName)
	slog.Info("starting mlx runner", "model", *modelName, "port", *port, "mode", mode)

	if mode != ModeImageGen {
		return fmt.Errorf("imagegen runner only supports image generation models")
	}

	// Initialize MLX only for image generation mode.
	if err := mlx.InitMLX(); err != nil {
		slog.Error("unable to initialize MLX", "error", err)
		return err
	}
	slog.Info("MLX library initialized")

	// Create and start server
	server, err := newServer(*modelName, *port)
	if err != nil {
		return fmt.Errorf("failed to create server: %w", err)
	}

	// Set up HTTP handlers
	mux := http.NewServeMux()
	mux.HandleFunc("/health", server.healthHandler)
	mux.HandleFunc("/completion", server.completionHandler)

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
	modelName string
	port      int

	// Image generation model.
	imageModel ImageModel
}

// newServer creates a new server instance for image generation models.
func newServer(modelName string, port int) (*server, error) {
	s := &server{
		modelName: modelName,
		port:      port,
	}

	if err := s.loadImageModel(); err != nil {
		return nil, fmt.Errorf("failed to load image model: %w", err)
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

	s.handleImageCompletion(w, r, req)
}
