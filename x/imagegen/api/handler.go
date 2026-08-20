package api

import (
	"encoding/base64"
	"fmt"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/x/imagegen"
)

// RunnerScheduler is the interface for scheduling a model runner.
// This is implemented by server.Server to avoid circular imports.
type RunnerScheduler interface {
	ScheduleImageGenRunner(ctx *gin.Context, modelName string, opts api.Options, keepAlive *api.Duration) (llm.LlamaServer, error)
}

// RegisterRoutes registers the image generation API routes.
func RegisterRoutes(r gin.IRouter, scheduler RunnerScheduler) {
	r.POST("/v1/images/generations", func(c *gin.Context) {
		ImageGenerationHandler(c, scheduler)
	})
}

// ImageGenerationHandler handles OpenAI-compatible image generation requests.
func ImageGenerationHandler(c *gin.Context, scheduler RunnerScheduler) {
	var req ImageGenerationRequest
	if err := c.BindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": gin.H{"message": err.Error()}})
		return
	}

	// Validate required fields
	if req.Model == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": gin.H{"message": "model is required"}})
		return
	}
	if req.Prompt == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": gin.H{"message": "prompt is required"}})
		return
	}

	// Apply defaults
	if req.N == 0 {
		req.N = 1
	}
	if req.Size == "" {
		req.Size = "1024x1024"
	}
	if req.ResponseFormat == "" {
		req.ResponseFormat = "b64_json"
	}

	// Verify model exists
	if imagegen.ResolveModelName(req.Model) == "" {
		c.JSON(http.StatusNotFound, gin.H{"error": gin.H{"message": fmt.Sprintf("model %q not found", req.Model)}})
		return
	}

	// Parse size
	width, height := parseSize(req.Size)

	// Build options - we repurpose NumCtx/NumGPU for width/height
	opts := api.Options{}
	opts.NumCtx = int(width)
	opts.NumGPU = int(height)

	// Schedule runner
	runner, err := scheduler.ScheduleImageGenRunner(c, req.Model, opts, nil)
	if err != nil {
		status := http.StatusInternalServerError
		if strings.Contains(err.Error(), "not found") {
			status = http.StatusNotFound
		}
		c.JSON(status, gin.H{"error": gin.H{"message": err.Error()}})
		return
	}

	// Build completion request
	completionReq := llm.CompletionRequest{
		Prompt:  req.Prompt,
		Options: &opts,
	}

	if req.Stream {
		handleStreamingResponse(c, runner, completionReq, req.ResponseFormat)
	} else {
		handleNonStreamingResponse(c, runner, completionReq, req.ResponseFormat)
	}
}

func handleStreamingResponse(c *gin.Context, runner llm.LlamaServer, req llm.CompletionRequest, format string) {
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")

	var imagePath string
	err := runner.Completion(c.Request.Context(), req, func(resp llm.CompletionResponse) {
		if resp.Done {
			imagePath = extractPath(resp.Content)
		} else {
			progress := parseProgress(resp.Content)
			if progress.Total > 0 {
				c.SSEvent("progress", progress)
				c.Writer.Flush()
			}
		}
	})
	if err != nil {
		c.SSEvent("error", gin.H{"error": err.Error()})
		return
	}

	c.SSEvent("done", buildResponse(imagePath, format))
}

func handleNonStreamingResponse(c *gin.Context, runner llm.LlamaServer, req llm.CompletionRequest, format string) {
	var imagePath string
	err := runner.Completion(c.Request.Context(), req, func(resp llm.CompletionResponse) {
		if resp.Done {
			imagePath = extractPath(resp.Content)
		}
	})
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": gin.H{"message": err.Error()}})
		return
	}

	c.JSON(http.StatusOK, buildResponse(imagePath, format))
}

func parseSize(size string) (int32, int32) {
	parts := strings.Split(size, "x")
	if len(parts) != 2 {
		return 1024, 1024
	}
	w, _ := strconv.Atoi(parts[0])
	h, _ := strconv.Atoi(parts[1])
	if w == 0 {
		w = 1024
	}
	if h == 0 {
		h = 1024
	}
	return int32(w), int32(h)
}

func extractPath(content string) string {
	if idx := strings.Index(content, "Image saved to: "); idx >= 0 {
		return strings.TrimSpace(content[idx+16:])
	}
	return ""
}

func parseProgress(content string) ImageProgressEvent {
	var step, total int
	fmt.Sscanf(content, "\rGenerating: step %d/%d", &step, &total)
	return ImageProgressEvent{Step: step, Total: total}
}

func buildResponse(imagePath, format string) ImageGenerationResponse {
	resp := ImageGenerationResponse{
		Created: time.Now().Unix(),
		Data:    make([]ImageData, 1),
	}

	if imagePath == "" {
		return resp
	}

	if format == "url" {
		resp.Data[0].URL = "file://" + imagePath
	} else {
		data, err := os.ReadFile(imagePath)
		if err == nil {
			resp.Data[0].B64JSON = base64.StdEncoding.EncodeToString(data)
		}
	}

	return resp
}

// HandleGenerateRequest handles Ollama /api/generate requests for image gen models.
// This allows routes.go to delegate image generation with minimal code.
func HandleGenerateRequest(c *gin.Context, scheduler RunnerScheduler, modelName, prompt string, keepAlive *api.Duration, streamFn func(c *gin.Context, ch chan any)) {
	opts := api.Options{}

	// Schedule runner
	runner, err := scheduler.ScheduleImageGenRunner(c, modelName, opts, keepAlive)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// Build completion request
	completionReq := llm.CompletionRequest{
		Prompt:  prompt,
		Options: &opts,
	}

	// Stream responses via channel
	ch := make(chan any)
	go func() {
		defer close(ch)
		err := runner.Completion(c.Request.Context(), completionReq, func(resp llm.CompletionResponse) {
			ch <- GenerateResponse{
				Model:     modelName,
				CreatedAt: time.Now().UTC(),
				Response:  resp.Content,
				Done:      resp.Done,
			}
		})
		if err != nil {
			// Log error but don't block - channel is already being consumed
			_ = err
		}
	}()

	streamFn(c, ch)
}

// GenerateResponse matches api.GenerateResponse structure for streaming.
type GenerateResponse struct {
	Model     string    `json:"model"`
	CreatedAt time.Time `json:"created_at"`
	Response  string    `json:"response"`
	Done      bool      `json:"done"`
}
