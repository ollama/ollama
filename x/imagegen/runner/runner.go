//go:build mlx

// Package runner provides a subprocess server for image generation.
// It listens on a port and handles HTTP requests for image generation.
package runner

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/ollama/ollama/x/imagegen"
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/models/glm_image"
	"github.com/ollama/ollama/x/imagegen/models/zimage"
)

// ImageModel is the interface for image generation models
type ImageModel interface {
	GenerateImage(ctx context.Context, prompt string, width, height int32, steps int, seed int64) (*mlx.Array, error)
}

// Request is the image generation request format
type Request struct {
	Prompt string `json:"prompt"`
	Width  int32  `json:"width,omitempty"`
	Height int32  `json:"height,omitempty"`
	Steps  int    `json:"steps,omitempty"`
	Seed   int64  `json:"seed,omitempty"`
}

// Response is streamed back for each progress update
type Response struct {
	Content string `json:"content,omitempty"`
	Image   string `json:"image,omitempty"` // Base64-encoded PNG
	Done    bool   `json:"done"`
}

// Server holds the model and handles requests
type Server struct {
	mu        sync.Mutex
	model     ImageModel
	modelName string
	modelType string // "zimage" or "glm_image"
}

// Execute is the entry point for the image runner subprocess
func Execute(args []string) error {
	fs := flag.NewFlagSet("image-runner", flag.ExitOnError)
	modelName := fs.String("model", "", "path to image model")
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

	slog.Info("starting image runner", "model", *modelName, "port", *port)

	// Check memory requirements before loading
	requiredMemory := imagegen.EstimateVRAM(*modelName)
	availableMemory := mlx.GetMemoryLimit()
	if availableMemory > 0 && availableMemory < requiredMemory {
		return fmt.Errorf("insufficient memory for image generation: need %d GB, have %d GB",
			requiredMemory/(1024*1024*1024), availableMemory/(1024*1024*1024))
	}

	// Detect model type and load appropriate model
	modelType, err := detectModelType(*modelName)
	if err != nil {
		return fmt.Errorf("failed to detect model type: %w", err)
	}

	var model ImageModel
	switch modelType {
	case "GlmImagePipeline":
		slog.Info("loading GLM-Image model")
		m := &glm_image.Model{}
		if err := m.Load(*modelName); err != nil {
			return fmt.Errorf("failed to load GLM-Image model: %w", err)
		}
		model = m
	default:
		// Default to zimage for ZImagePipeline, FluxPipeline, and unknown types
		slog.Info("loading Z-Image model")
		m := &zimage.Model{}
		if err := m.Load(*modelName); err != nil {
			return fmt.Errorf("failed to load Z-Image model: %w", err)
		}
		model = m
	}

	server := &Server{
		model:     model,
		modelName: *modelName,
		modelType: modelType,
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
		slog.Info("shutting down image runner")
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		httpServer.Shutdown(ctx)
		close(done)
	}()

	slog.Info("image runner listening", "addr", httpServer.Addr)
	if err := httpServer.ListenAndServe(); err != http.ErrServerClosed {
		return err
	}

	<-done
	return nil
}

func (s *Server) healthHandler(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (s *Server) completionHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req Request
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Serialize generation requests - MLX model may not handle concurrent generation
	s.mu.Lock()
	defer s.mu.Unlock()

	// Apply defaults
	if req.Width <= 0 {
		req.Width = 1024
	}
	if req.Height <= 0 {
		req.Height = 1024
	}
	if req.Steps <= 0 {
		// Default steps depend on model type
		switch s.modelType {
		case "GlmImagePipeline":
			req.Steps = 50 // GLM-Image default
		default:
			req.Steps = 9 // Z-Image turbo default
		}
	}
	if req.Seed <= 0 {
		req.Seed = time.Now().UnixNano()
	}

	// Set up streaming response
	w.Header().Set("Content-Type", "application/x-ndjson")
	w.Header().Set("Transfer-Encoding", "chunked")
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	// Generate image using interface method
	ctx := r.Context()
	img, err := s.model.GenerateImage(ctx, req.Prompt, req.Width, req.Height, req.Steps, req.Seed)

	if err != nil {
		// Don't send error for cancellation
		if ctx.Err() != nil {
			return
		}
		resp := Response{Content: fmt.Sprintf("error: %v", err), Done: true}
		data, _ := json.Marshal(resp)
		w.Write(data)
		w.Write([]byte("\n"))
		return
	}

	// Encode image as base64 PNG
	imageData, err := imagegen.EncodeImageBase64(img)
	if err != nil {
		resp := Response{Content: fmt.Sprintf("error encoding: %v", err), Done: true}
		data, _ := json.Marshal(resp)
		w.Write(data)
		w.Write([]byte("\n"))
		return
	}

	// Free the generated image array and clean up MLX state
	img.Free()
	mlx.ClearCache()
	mlx.MetalResetPeakMemory()

	// Send final response with image data
	resp := Response{
		Image: imageData,
		Done:  true,
	}
	data, _ := json.Marshal(resp)
	w.Write(data)
	w.Write([]byte("\n"))
	flusher.Flush()
}

// detectModelType reads the model manifest and returns the pipeline class name
func detectModelType(modelName string) (string, error) {
	manifest, err := imagegen.LoadManifest(modelName)
	if err != nil {
		return "", err
	}

	data, err := manifest.ReadConfig("model_index.json")
	if err != nil {
		return "ZImagePipeline", nil // Default to Z-Image
	}

	// Try both _class_name (diffusers format) and architecture (ollama format)
	var index struct {
		ClassName    string `json:"_class_name"`
		Architecture string `json:"architecture"`
	}
	if err := json.Unmarshal(data, &index); err != nil {
		return "ZImagePipeline", nil
	}

	// Prefer _class_name, fall back to architecture
	className := index.ClassName
	if className == "" {
		className = index.Architecture
	}
	if className == "" {
		return "ZImagePipeline", nil
	}
	return className, nil
}
