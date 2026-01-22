//go:build mlx

// Package runner provides a subprocess server for image generation.
// It listens on a port and handles HTTP requests for image generation.
package runner

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/ollama/ollama/x/imagegen"
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/models/flux2"
	"github.com/ollama/ollama/x/imagegen/models/zimage"
)

// Request is the image generation request format
type Request struct {
	Prompt string   `json:"prompt"`
	Width  int32    `json:"width,omitempty"`
	Height int32    `json:"height,omitempty"`
	Steps  int      `json:"steps,omitempty"`
	Seed   int64    `json:"seed,omitempty"`
	Images [][]byte `json:"images,omitempty"` // Input images for image editing/conditioning
}

// Response is streamed back for each progress update
type Response struct {
	Content string `json:"content,omitempty"`
	Image   string `json:"image,omitempty"` // Base64-encoded PNG
	Done    bool   `json:"done"`
	Step    int    `json:"step,omitempty"`
	Total   int    `json:"total,omitempty"`
}

// ImageModel is the interface for image generation models
type ImageModel interface {
	GenerateImage(ctx context.Context, prompt string, width, height int32, steps int, seed int64, progress func(step, total int)) (*mlx.Array, error)
}

// ImageEditModel extends ImageModel with image editing/conditioning capability.
// Models that support input images for editing should implement this interface.
type ImageEditModel interface {
	ImageModel
	GenerateImageWithInputs(ctx context.Context, prompt string, width, height int32, steps int, seed int64, inputImages []image.Image, progress func(step, total int)) (*mlx.Array, error)
}

// Server holds the model and handles requests
type Server struct {
	mu        sync.Mutex
	model     ImageModel
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

	err := mlx.InitMLX()
	if err != nil {
		slog.Error("unable to initialize MLX", "error", err)
		return err
	}
	slog.Info("MLX library initialized")
	slog.Info("starting image runner", "model", *modelName, "port", *port)

	// Check memory requirements before loading
	requiredMemory := imagegen.EstimateVRAM(*modelName)
	availableMemory := mlx.GetMemoryLimit()
	if availableMemory > 0 && availableMemory < requiredMemory {
		return fmt.Errorf("insufficient memory for image generation: need %d GB, have %d GB",
			requiredMemory/(1024*1024*1024), availableMemory/(1024*1024*1024))
	}

	// Detect model type and load appropriate model
	modelType := imagegen.DetectModelType(*modelName)
	slog.Info("detected model type", "type", modelType)

	var model ImageModel
	switch modelType {
	case "Flux2KleinPipeline":
		m := &flux2.Model{}
		if err := m.Load(*modelName); err != nil {
			return fmt.Errorf("failed to load model: %w", err)
		}
		model = m
	default:
		// Default to Z-Image for ZImagePipeline, FluxPipeline, etc.
		m := &zimage.Model{}
		if err := m.Load(*modelName); err != nil {
			return fmt.Errorf("failed to load model: %w", err)
		}
		model = m
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

	// Validate and decode input images
	const maxInputImages = 2
	if len(req.Images) > maxInputImages {
		http.Error(w, fmt.Sprintf("too many input images, maximum is %d", maxInputImages), http.StatusBadRequest)
		return
	}

	var inputImages []image.Image
	if len(req.Images) > 0 {
		// TODO: add memory check for input images

		inputImages = make([]image.Image, len(req.Images))
		for i, imgBytes := range req.Images {
			img, err := imagegen.DecodeImage(imgBytes)
			if err != nil {
				http.Error(w, fmt.Sprintf("invalid image %d: %v", i, err), http.StatusBadRequest)
				return
			}
			inputImages[i] = img
		}
		slog.Info("decoded input images", "count", len(inputImages))

		// Default width/height to first input image dimensions, scaled to max 1024
		bounds := inputImages[0].Bounds()
		w, h := bounds.Dx(), bounds.Dy()
		if w > 1024 || h > 1024 {
			if w > h {
				h = h * 1024 / w
				w = 1024
			} else {
				w = w * 1024 / h
				h = 1024
			}
		}
		req.Width = int32(w)
		req.Height = int32(h)
	}

	// Serialize generation requests - MLX model may not handle concurrent generation
	s.mu.Lock()
	defer s.mu.Unlock()

	// Model applies its own defaults for width/height/steps
	// Only seed needs to be set here if not provided
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

	// Generate image using the common interface
	ctx := r.Context()
	enc := json.NewEncoder(w)

	// Progress callback streams step updates
	progress := func(step, total int) {
		resp := Response{Step: step, Total: total}
		enc.Encode(resp)
		w.Write([]byte("\n"))
		flusher.Flush()
	}

	// Use ImageEditModel if available and images provided, otherwise use basic ImageModel
	var img *mlx.Array
	var err error
	if len(inputImages) > 0 {
		editModel, ok := s.model.(ImageEditModel)
		if !ok {
			http.Error(w, "model does not support image editing", http.StatusBadRequest)
			return
		}
		img, err = editModel.GenerateImageWithInputs(ctx, req.Prompt, req.Width, req.Height, req.Steps, req.Seed, inputImages, progress)
	} else {
		img, err = s.model.GenerateImage(ctx, req.Prompt, req.Width, req.Height, req.Steps, req.Seed, progress)
	}

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
