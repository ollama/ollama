//go:build mlx

package imagegen

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"sync"
	"time"

	"github.com/ollama/ollama/x/imagegen/manifest"
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/models/flux2"
	"github.com/ollama/ollama/x/imagegen/models/zimage"
)

// ImageModel is the interface for image generation models.
type ImageModel interface {
	GenerateImage(ctx context.Context, prompt string, width, height int32, steps int, seed int64, progress func(step, total int)) (*mlx.Array, error)
}

var imageGenMu sync.Mutex

// loadImageModel loads an image generation model.
func (s *server) loadImageModel() error {
	// Check memory requirements before loading
	var requiredMemory uint64
	if modelManifest, err := manifest.LoadManifest(s.modelName); err == nil {
		requiredMemory = uint64(modelManifest.TotalTensorSize())
	}
	availableMemory := mlx.GetMemoryLimit()
	if availableMemory > 0 && requiredMemory > 0 && availableMemory < requiredMemory {
		return fmt.Errorf("insufficient memory for image generation: need %d GB, have %d GB",
			requiredMemory/(1024*1024*1024), availableMemory/(1024*1024*1024))
	}

	// Detect model type and load appropriate model
	modelType := DetectModelType(s.modelName)
	slog.Info("detected image model type", "type", modelType)

	var model ImageModel
	switch modelType {
	case "Flux2KleinPipeline":
		m := &flux2.Model{}
		if err := m.Load(s.modelName); err != nil {
			return fmt.Errorf("failed to load flux2 model: %w", err)
		}
		model = m
	default:
		// Default to Z-Image for ZImagePipeline, FluxPipeline, etc.
		m := &zimage.Model{}
		if err := m.Load(s.modelName); err != nil {
			return fmt.Errorf("failed to load zimage model: %w", err)
		}
		model = m
	}

	s.imageModel = model
	return nil
}

// handleImageCompletion handles image generation requests.
func (s *server) handleImageCompletion(w http.ResponseWriter, r *http.Request, req Request) {
	// Serialize generation requests - MLX model may not handle concurrent generation
	imageGenMu.Lock()
	defer imageGenMu.Unlock()

	// Set seed if not provided
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

	ctx := r.Context()
	enc := json.NewEncoder(w)

	// Progress callback streams step updates
	progress := func(step, total int) {
		resp := Response{Step: step, Total: total}
		enc.Encode(resp)
		w.Write([]byte("\n"))
		flusher.Flush()
	}

	// Generate image
	img, err := s.imageModel.GenerateImage(ctx, req.Prompt, req.Width, req.Height, req.Steps, req.Seed, progress)
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
	imageData, err := EncodeImageBase64(img)
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
