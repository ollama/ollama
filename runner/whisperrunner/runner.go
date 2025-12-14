package whisperrunner

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/stt"
	"github.com/ollama/ollama/whisper"
)

// Runner manages the whisper model execution as a subprocess
type Runner struct {
	server    stt.WhisperServer
	modelPath string
	port      int

	mu        sync.RWMutex
	loaded    bool
	modelInfo whisper.ModelInfo

	sem chan struct{} // Semaphore for concurrent requests
}

// Config holds runner configuration
type Config struct {
	ModelPath  string
	Port       int
	NumThreads int
	UseGPU     bool
	FlashAttn  bool
	GPUDevice  int
}

// Status represents the runner status
type Status int

const (
	StatusLaunched Status = iota
	StatusLoading
	StatusReady
	StatusError
)

func (s Status) String() string {
	switch s {
	case StatusLaunched:
		return "launched"
	case StatusLoading:
		return "loading"
	case StatusReady:
		return "ready"
	case StatusError:
		return "error"
	default:
		return "unknown"
	}
}

// StatusResponse for health check endpoint
type StatusResponse struct {
	Status   string  `json:"status"`
	Progress float32 `json:"progress,omitempty"`
}

// InternalTranscribeRequest for subprocess communication
type InternalTranscribeRequest struct {
	Samples       []float32      `json:"samples"`
	Language      string         `json:"language,omitempty"`
	Translate     bool           `json:"translate,omitempty"`
	InitialPrompt string         `json:"initial_prompt,omitempty"`
	Temperature   float32        `json:"temperature,omitempty"`
	NoTimestamps  bool           `json:"no_timestamps,omitempty"`
	Options       map[string]any `json:"options,omitempty"`
}

// NewRunner creates a new whisper runner
func NewRunner(config Config) (*Runner, error) {
	if config.ModelPath == "" {
		return nil, errors.New("model path is required")
	}

	if _, err := os.Stat(config.ModelPath); err != nil {
		return nil, fmt.Errorf("model not found: %s", config.ModelPath)
	}

	params := whisper.ContextParams{
		UseGPU:    config.UseGPU,
		FlashAttn: config.FlashAttn,
		GPUDevice: config.GPUDevice,
	}

	server, err := stt.NewWhisperServer(config.ModelPath, params)
	if err != nil {
		return nil, err
	}

	maxConcurrent := 4
	if config.UseGPU {
		maxConcurrent = 2
	}

	return &Runner{
		server:    server,
		modelPath: config.ModelPath,
		port:      config.Port,
		sem:       make(chan struct{}, maxConcurrent),
	}, nil
}

// Load loads the model onto hardware
func (r *Runner) Load(ctx context.Context, gpus []ml.DeviceInfo) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.loaded {
		return nil
	}

	if err := r.server.Load(ctx, gpus); err != nil {
		return err
	}

	r.modelInfo = r.server.GetModelInfo()
	r.loaded = true
	return nil
}

// Transcribe performs transcription
func (r *Runner) Transcribe(ctx context.Context, req InternalTranscribeRequest) (*api.TranscribeResponse, error) {
	select {
	case r.sem <- struct{}{}:
		defer func() { <-r.sem }()
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	r.mu.RLock()
	if !r.loaded {
		r.mu.RUnlock()
		return nil, errors.New("model not loaded")
	}
	r.mu.RUnlock()

	// Build STT request from internal request
	sttReq := stt.TranscribeRequest{
		Samples:       req.Samples,
		Language:      req.Language,
		Translate:     req.Translate,
		InitialPrompt: req.InitialPrompt,
		Temperature:   req.Temperature,
		NoTimestamps:  req.NoTimestamps,
		Options:       req.Options,
	}

	start := time.Now()
	resp, err := r.server.Transcribe(ctx, sttReq)
	if err != nil {
		return nil, err
	}

	result := &api.TranscribeResponse{
		Model:              r.modelPath,
		Text:               combineTexts(resp.Segments),
		Language:           resp.Language,
		Duration:           float64(len(req.Samples)) / float64(whisper.SampleRate),
		ProcessingDuration: time.Since(start),
		Done:               true,
	}

	result.Segments = make([]api.TranscribeSegment, len(resp.Segments))
	for i, seg := range resp.Segments {
		result.Segments[i] = api.TranscribeSegment{
			ID:    i,
			Start: seg.Start.Seconds(),
			End:   seg.End.Seconds(),
			Text:  seg.Text,
		}
		if len(seg.Tokens) > 0 {
			result.Segments[i].Tokens = make([]int, len(seg.Tokens))
			for j, tok := range seg.Tokens {
				result.Segments[i].Tokens[j] = tok.ID
			}
		}
	}

	if req.Translate {
		result.Task = "translate"
	} else {
		result.Task = "transcribe"
	}

	return result, nil
}

// Close releases resources
func (r *Runner) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.server != nil {
		return r.server.Close()
	}
	return nil
}

func combineTexts(segments []whisper.Segment) string {
	var builder strings.Builder
	for _, seg := range segments {
		text := strings.TrimSpace(seg.Text)
		if text != "" {
			if builder.Len() > 0 {
				builder.WriteString(" ")
			}
			builder.WriteString(text)
		}
	}
	return builder.String()
}

// ============================================================================
// HTTP Server for Subprocess Mode
// ============================================================================

func (r *Runner) StartServer() error {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", r.handleHealth)
	mux.HandleFunc("/load", r.handleLoad)
	mux.HandleFunc("/transcribe", r.handleTranscribe)
	mux.HandleFunc("/info", r.handleInfo)

	addr := fmt.Sprintf("127.0.0.1:%d", r.port)
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		return err
	}

	slog.Info("whisper runner started", "addr", addr, "model", r.modelPath)

	server := &http.Server{
		Handler:      mux,
		ReadTimeout:  10 * time.Minute,
		WriteTimeout: 10 * time.Minute,
	}

	go func() {
		sigChan := make(chan os.Signal, 1)
		signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
		<-sigChan
		slog.Info("shutting down whisper runner")
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		server.Shutdown(ctx)
	}()

	return server.Serve(listener)
}

func (r *Runner) handleHealth(w http.ResponseWriter, req *http.Request) {
	r.mu.RLock()
	loaded := r.loaded
	r.mu.RUnlock()

	status := StatusLaunched
	if loaded {
		status = StatusReady
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(StatusResponse{Status: status.String()})
}

func (r *Runner) handleLoad(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var loadReq struct {
		GPUs []ml.DeviceInfo `json:"gpus,omitempty"`
	}
	json.NewDecoder(req.Body).Decode(&loadReq)

	if err := r.Load(req.Context(), loadReq.GPUs); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]bool{"success": true})
}

func (r *Runner) handleTranscribe(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var transcribeReq InternalTranscribeRequest
	if err := json.NewDecoder(req.Body).Decode(&transcribeReq); err != nil {
		http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
		return
	}

	if len(transcribeReq.Samples) == 0 {
		http.Error(w, "samples are required", http.StatusBadRequest)
		return
	}

	resp, err := r.Transcribe(req.Context(), transcribeReq)
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (r *Runner) handleInfo(w http.ResponseWriter, req *http.Request) {
	r.mu.RLock()
	info := r.modelInfo
	multilingual := r.server != nil && r.server.IsMultilingual()
	loaded := r.loaded
	r.mu.RUnlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"model_path":   r.modelPath,
		"loaded":       loaded,
		"multilingual": multilingual,
		"vocab_size":   info.VocabSize,
	})
}

// Main entry point for subprocess mode
func Main() {
	var modelPath string
	var port int
	var useGPU bool
	var flashAttn bool
	var gpuDevice int
	var numThreads int

	flag.StringVar(&modelPath, "model", "", "Path to whisper model file")
	flag.IntVar(&port, "port", 0, "Port to listen on")
	flag.BoolVar(&useGPU, "gpu", true, "Use GPU acceleration")
	flag.BoolVar(&flashAttn, "flash-attn", false, "Enable flash attention")
	flag.IntVar(&gpuDevice, "gpu-device", 0, "GPU device ID")
	flag.IntVar(&numThreads, "threads", 0, "Number of CPU threads")
	flag.Parse()

	if modelPath == "" {
		log.Fatal("--model is required")
	}
	if port == 0 {
		log.Fatal("--port is required")
	}
	if numThreads == 0 {
		numThreads = runtime.NumCPU()
	}

	config := Config{
		ModelPath:  modelPath,
		Port:       port,
		NumThreads: numThreads,
		UseGPU:     useGPU,
		FlashAttn:  flashAttn,
		GPUDevice:  gpuDevice,
	}

	slog.Info("creating whisper runner", "model", modelPath, "port", port, "gpu", useGPU)

	runner, err := NewRunner(config)
	if err != nil {
		log.Fatalf("failed to create runner: %v", err)
	}
	defer runner.Close()

	if err := runner.Load(context.Background(), nil); err != nil {
		log.Fatalf("failed to load model: %v", err)
	}

	if err := runner.StartServer(); err != nil {
		log.Fatalf("server error: %v", err)
	}
}
