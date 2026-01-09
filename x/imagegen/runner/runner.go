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
	"path/filepath"
	"sync"
	"syscall"
	"time"

	"github.com/ollama/ollama/x/imagegen"
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/models/zimage"
)

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
	Content string `json:"content"`
	Done    bool   `json:"done"`
}

// Server holds the model and handles requests
type Server struct {
	mu        sync.Mutex
	model     *zimage.Model
	modelName string
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

	// Load model
	model := &zimage.Model{}
	if err := model.Load(*modelName); err != nil {
		return fmt.Errorf("failed to load model: %w", err)
	}

	server := &Server{
		model:     model,
		modelName: *modelName,
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
		req.Steps = 9
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

	// Generate image
	ctx := r.Context()
	img, err := s.model.GenerateFromConfig(&zimage.GenerateConfig{
		Ctx:    ctx,
		Prompt: req.Prompt,
		Width:  req.Width,
		Height: req.Height,
		Steps:  req.Steps,
		Seed:   req.Seed,
		Progress: func(step, total int) {
			resp := Response{
				Content: fmt.Sprintf("\rGenerating: step %d/%d", step, total),
				Done:    false,
			}
			data, _ := json.Marshal(resp)
			w.Write(data)
			w.Write([]byte("\n"))
			flusher.Flush()
		},
	})

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

	// Save image
	outPath := filepath.Join(os.TempDir(), fmt.Sprintf("ollama-image-%d.png", time.Now().UnixNano()))
	if err := imagegen.SaveImage(img, outPath); err != nil {
		resp := Response{Content: fmt.Sprintf("error saving: %v", err), Done: true}
		data, _ := json.Marshal(resp)
		w.Write(data)
		w.Write([]byte("\n"))
		return
	}

	// Free the generated image array and clean up MLX state
	img.Free()
	mlx.ClearCache()

	// Send final response
	resp := Response{
		Content: fmt.Sprintf("\n\nImage saved to: %s\n", outPath),
		Done:    true,
	}
	data, _ := json.Marshal(resp)
	w.Write(data)
	w.Write([]byte("\n"))
	flusher.Flush()
}
