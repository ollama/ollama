package imagegen

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math/rand"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/x/imagegen/manifest"
)

// Server wraps an MLX runner subprocess to implement llm.LlamaServer.
//
// This implementation is compatible with Ollama's scheduler and can be loaded/unloaded
// like any other model. It is used for image generation models.
type Server struct {
	mu          sync.Mutex
	cmd         *exec.Cmd
	port        int
	modelName   string
	vramSize    uint64
	done        chan error
	client      *http.Client
	lastErr     string // Last stderr line for error reporting
	lastErrLock sync.Mutex
}

// NewServer prepares a new MLX runner server for image generation models.
// The subprocess is not started until Load() is called.
func NewServer(modelName string) (*Server, error) {
	// Validate platform support before attempting to start
	if err := CheckPlatformSupport(); err != nil {
		return nil, err
	}

	return &Server{
		modelName: modelName,
		done:      make(chan error, 1),
		client:    &http.Client{Timeout: 10 * time.Minute},
	}, nil
}

// ModelPath returns the path to the model.
func (s *Server) ModelPath() string {
	return s.modelName
}

// Load checks whether the model fits in GPU memory and starts the subprocess.
func (s *Server) Load(ctx context.Context, _ ml.SystemInfo, gpus []ml.DeviceInfo, requireFull bool) ([]ml.DeviceID, error) {
	// Estimate VRAM based on tensor size from manifest
	if modelManifest, err := manifest.LoadManifest(s.modelName); err == nil {
		s.vramSize = uint64(modelManifest.TotalTensorSize())
	} else {
		s.vramSize = 8 * 1024 * 1024 * 1024
	}

	if len(gpus) > 0 {
		available := gpus[0].FreeMemory
		overhead := gpus[0].MinimumMemory() + envconfig.GpuOverhead()
		if available > overhead {
			available -= overhead
		} else {
			available = 0
		}

		if s.vramSize > available {
			if requireFull {
				return nil, llm.ErrLoadRequiredFull
			}
			return nil, fmt.Errorf("model requires %s but only %s are available (after %s overhead)", format.HumanBytes2(s.vramSize), format.HumanBytes2(available), format.HumanBytes2(overhead))
		}
	}

	// Find a free port
	port := 0
	if a, err := net.ResolveTCPAddr("tcp", "localhost:0"); err == nil {
		if l, err := net.ListenTCP("tcp", a); err == nil {
			port = l.Addr().(*net.TCPAddr).Port
			l.Close()
		}
	}
	if port == 0 {
		port = rand.Intn(65535-49152) + 49152
	}
	s.port = port

	// Get the current executable path (we use the same binary with runner subcommand)
	exe, err := os.Executable()
	if err != nil {
		return nil, fmt.Errorf("unable to lookup executable path: %w", err)
	}
	if eval, err := filepath.EvalSymlinks(exe); err == nil {
		exe = eval
	}

	// Spawn subprocess: ollama runner --imagegen-engine --model <path> --port <port>
	cmd := exec.Command(exe, "runner", "--imagegen-engine", "--model", s.modelName, "--port", strconv.Itoa(port))
	cmd.Env = os.Environ()
	configureMLXSubprocessEnv(cmd, ml.LibraryPaths(gpus))

	s.cmd = cmd

	// Forward subprocess stdout/stderr to server logs
	stdout, _ := cmd.StdoutPipe()
	stderr, _ := cmd.StderrPipe()
	go func() {
		scanner := bufio.NewScanner(stdout)
		for scanner.Scan() {
			slog.Info("mlx-runner", "msg", scanner.Text())
		}
	}()
	go func() {
		scanner := bufio.NewScanner(stderr)
		for scanner.Scan() {
			line := scanner.Text()
			slog.Warn("mlx-runner", "msg", line)
			s.lastErrLock.Lock()
			s.lastErr = line
			s.lastErrLock.Unlock()
		}
	}()

	slog.Info("starting mlx runner subprocess", "model", s.modelName, "port", s.port)
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("failed to start mlx runner: %w", err)
	}

	// Reap subprocess when it exits
	go func() {
		err := cmd.Wait()
		s.done <- err
	}()

	return nil, nil
}

// Ping checks if the subprocess is healthy.
func (s *Server) Ping(ctx context.Context) error {
	url := fmt.Sprintf("http://127.0.0.1:%d/health", s.port)
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return err
	}
	resp, err := s.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("health check failed: %d", resp.StatusCode)
	}
	return nil
}

func mlxLibraryPathEnv() string {
	switch runtime.GOOS {
	case "windows":
		return "PATH"
	case "darwin":
		return "DYLD_LIBRARY_PATH"
	default:
		return "LD_LIBRARY_PATH"
	}
}

func configureMLXSubprocessEnv(cmd *exec.Cmd, libraryPaths []string) {
	if len(libraryPaths) == 0 {
		return
	}

	// Search order for the imagegen runner is:
	//   1. bundled lib/ollama root
	//   2. backend-specific library dirs selected during GPU discovery
	//   3. any existing caller-provided library path values
	pathEnv := mlxLibraryPathEnv()
	pathEnvPaths := append([]string{}, libraryPaths...)
	if existingPath, ok := os.LookupEnv(pathEnv); ok {
		pathEnvPaths = append(pathEnvPaths, filepath.SplitList(existingPath)...)
	}
	setSubprocessEnv(cmd, pathEnv, strings.Join(pathEnvPaths, string(filepath.ListSeparator)))
	slog.Debug("mlx subprocess library path", pathEnv, strings.Join(pathEnvPaths, string(filepath.ListSeparator)))

	ollamaLibraryPaths := append([]string{}, libraryPaths...)
	if existingPath, ok := os.LookupEnv("OLLAMA_LIBRARY_PATH"); ok {
		ollamaLibraryPaths = append(ollamaLibraryPaths, filepath.SplitList(existingPath)...)
	}
	setSubprocessEnv(cmd, "OLLAMA_LIBRARY_PATH", strings.Join(ollamaLibraryPaths, string(filepath.ListSeparator)))
	slog.Debug("mlx subprocess library path", "OLLAMA_LIBRARY_PATH", strings.Join(ollamaLibraryPaths, string(filepath.ListSeparator)))
}

func setSubprocessEnv(cmd *exec.Cmd, key, value string) {
	for i := range cmd.Env {
		name, _, ok := strings.Cut(cmd.Env[i], "=")
		if ok && strings.EqualFold(name, key) {
			cmd.Env[i] = key + "=" + value
			return
		}
	}
	cmd.Env = append(cmd.Env, key+"="+value)
}

// getLastErr returns the last stderr line.
func (s *Server) getLastErr() string {
	s.lastErrLock.Lock()
	defer s.lastErrLock.Unlock()
	return s.lastErr
}

// WaitUntilRunning waits for the subprocess to be ready.
func (s *Server) WaitUntilRunning(ctx context.Context) error {
	timeout := time.After(envconfig.LoadTimeout())
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case err := <-s.done:
			errMsg := s.getLastErr()
			if errMsg != "" {
				return fmt.Errorf("mlx runner failed: %s (exit: %v)", errMsg, err)
			}
			return fmt.Errorf("mlx runner exited unexpectedly: %w", err)
		case <-timeout:
			errMsg := s.getLastErr()
			if errMsg != "" {
				return fmt.Errorf("timeout waiting for mlx runner: %s", errMsg)
			}
			return errors.New("timeout waiting for mlx runner to start")
		case <-ticker.C:
			if err := s.Ping(ctx); err == nil {
				slog.Info("mlx runner is ready", "port", s.port)
				return nil
			}
		}
	}
}

// Completion handles both text and image generation requests.
func (s *Server) Completion(ctx context.Context, req llm.CompletionRequest, fn func(llm.CompletionResponse)) error {
	seed := req.Seed
	if seed == 0 {
		seed = time.Now().UnixNano()
	}

	// Extract raw image bytes from llm.ImageData slice
	var images [][]byte
	for _, img := range req.Images {
		images = append(images, img.Data)
	}

	// Build request for subprocess
	creq := Request{
		Prompt: req.Prompt,
		Width:  req.Width,
		Height: req.Height,
		Steps:  int(req.Steps),
		Seed:   seed,
		Images: images,
	}

	// Pass LLM options if present
	if req.Options != nil {
		creq.Options = &RequestOptions{
			NumPredict:  req.Options.NumPredict,
			Temperature: float64(req.Options.Temperature),
			TopP:        float64(req.Options.TopP),
			TopK:        req.Options.TopK,
			Stop:        req.Options.Stop,
		}
	}

	body, err := json.Marshal(creq)
	if err != nil {
		return err
	}

	url := fmt.Sprintf("http://127.0.0.1:%d/completion", s.port)
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := s.client.Do(httpReq)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("%s", strings.TrimSpace(string(body)))
	}

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 1024*1024), 16*1024*1024) // 16MB max
	for scanner.Scan() {
		// Parse subprocess response
		var raw struct {
			Image              string `json:"image,omitempty"`
			Content            string `json:"content,omitempty"`
			Done               bool   `json:"done"`
			Step               int    `json:"step,omitempty"`
			Total              int    `json:"total,omitempty"`
			StopReason         string `json:"stop_reason,omitempty"`
			PromptEvalCount    int    `json:"prompt_eval_count,omitempty"`
			PromptEvalDuration int    `json:"prompt_eval_duration,omitempty"`
			EvalCount          int    `json:"eval_count,omitempty"`
			EvalDuration       int    `json:"eval_duration,omitempty"`
		}
		if err := json.Unmarshal(scanner.Bytes(), &raw); err != nil {
			slog.Debug("mlx response parse error", "error", err, "line", string(scanner.Bytes()))
			continue
		}

		// Log stop reason when generation completes
		if raw.Done && raw.StopReason != "" {
			slog.Info("mlx generation completed", "stop_reason", raw.StopReason)
		}

		// Convert to llm.CompletionResponse
		cresp := llm.CompletionResponse{
			Content:            raw.Content,
			Done:               raw.Done,
			Step:               raw.Step,
			TotalSteps:         raw.Total,
			Image:              raw.Image,
			PromptEvalCount:    raw.PromptEvalCount,
			PromptEvalDuration: time.Duration(raw.PromptEvalDuration),
			EvalCount:          raw.EvalCount,
			EvalDuration:       time.Duration(raw.EvalDuration),
		}

		fn(cresp)
		if cresp.Done {
			return nil
		}
	}

	// Scanner exited without receiving Done - connection was likely closed
	scanErr := scanner.Err()
	if scanErr != nil {
		slog.Error("mlx scanner error", "error", scanErr)
	} else {
		slog.Warn("mlx scanner EOF without Done response - subprocess may have crashed")
	}

	// Check if subprocess is still alive
	if s.HasExited() {
		slog.Error("mlx subprocess has exited unexpectedly")
	}

	return scanErr
}

// Close terminates the subprocess.
func (s *Server) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.cmd != nil && s.cmd.Process != nil {
		slog.Info("stopping mlx runner subprocess", "pid", s.cmd.Process.Pid)
		s.cmd.Process.Signal(os.Interrupt)

		// Wait briefly for graceful shutdown
		select {
		case <-s.done:
		case <-time.After(5 * time.Second):
			s.cmd.Process.Kill()
		}
		s.cmd = nil
	}
	return nil
}

// MemorySize returns the total and VRAM memory usage.
func (s *Server) MemorySize() (total, vram uint64) {
	return s.vramSize, s.vramSize
}

// VRAMByGPU returns VRAM usage for a specific GPU.
func (s *Server) VRAMByGPU(id ml.DeviceID) uint64 {
	return s.vramSize
}

// ContextLength returns the context length (not applicable for image generation).
func (s *Server) ContextLength() int {
	return 0
}

// Embedding returns embeddings for the input.
func (s *Server) Embedding(ctx context.Context, input string) ([]float32, int, error) {
	return nil, 0, errors.New("embeddings not supported for MLX models")
}

// Tokenize tokenizes the input content.
func (s *Server) Tokenize(ctx context.Context, content string) ([]int, error) {
	return nil, errors.New("tokenization not supported for image generation models")
}

// Detokenize converts tokens back to text.
func (s *Server) Detokenize(ctx context.Context, tokens []int) (string, error) {
	return "", errors.New("detokenization not supported for MLX models")
}

// Pid returns the process ID of the subprocess.
func (s *Server) Pid() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.cmd != nil && s.cmd.Process != nil {
		return s.cmd.Process.Pid
	}
	return -1
}

// GetPort returns the port the subprocess is listening on.
func (s *Server) GetPort() int {
	return s.port
}

// GetDeviceInfos returns device information.
func (s *Server) GetDeviceInfos(ctx context.Context) []ml.DeviceInfo {
	return nil
}

// HasExited returns whether the subprocess has exited.
func (s *Server) HasExited() bool {
	select {
	case <-s.done:
		return true
	default:
		return false
	}
}

// Ensure Server implements llm.LlamaServer
var _ llm.LlamaServer = (*Server)(nil)
