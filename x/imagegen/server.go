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

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/x/imagegen/manifest"
)

// Server wraps an MLX runner subprocess to implement llm.LlamaServer.
//
// This implementation is compatible with Ollama's scheduler and can be loaded/unloaded
// like any other model. It supports both LLM (safetensors) and image generation models.
type Server struct {
	mu          sync.Mutex
	cmd         *exec.Cmd
	port        int
	modelName   string
	mode        ModelMode
	vramSize    uint64
	done        chan error
	client      *http.Client
	lastErr     string // Last stderr line for error reporting
	lastErrLock sync.Mutex
}

// NewServer spawns a new MLX runner subprocess and waits until it's ready.
func NewServer(modelName string, mode ModelMode) (*Server, error) {
	// Validate platform support before attempting to start
	if err := CheckPlatformSupport(); err != nil {
		return nil, err
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

	// Get the current executable path (we use the same binary with runner subcommand)
	exe, err := os.Executable()
	if err != nil {
		return nil, fmt.Errorf("unable to lookup executable path: %w", err)
	}
	if eval, err := filepath.EvalSymlinks(exe); err == nil {
		exe = eval
	}

	// Spawn subprocess: ollama runner --imagegen-engine --model <path> --port <port>
	cmd := exec.Command(exe, "runner", "--imagegen-engine", "--model", modelName, "--port", strconv.Itoa(port))
	cmd.Env = os.Environ()

	// On Linux, set LD_LIBRARY_PATH to include MLX library directories
	if runtime.GOOS == "linux" {
		// Build library paths: start with LibOllamaPath, then add any mlx_* subdirectories
		libraryPaths := []string{ml.LibOllamaPath}
		if mlxDirs, err := filepath.Glob(filepath.Join(ml.LibOllamaPath, "mlx_*")); err == nil {
			libraryPaths = append(libraryPaths, mlxDirs...)
		}

		// Append existing LD_LIBRARY_PATH if set
		if existingPath, ok := os.LookupEnv("LD_LIBRARY_PATH"); ok {
			libraryPaths = append(libraryPaths, filepath.SplitList(existingPath)...)
		}

		pathEnvVal := strings.Join(libraryPaths, string(filepath.ListSeparator))

		// Update or add LD_LIBRARY_PATH in cmd.Env
		found := false
		for i := range cmd.Env {
			if strings.HasPrefix(cmd.Env[i], "LD_LIBRARY_PATH=") {
				cmd.Env[i] = "LD_LIBRARY_PATH=" + pathEnvVal
				found = true
				break
			}
		}
		if !found {
			cmd.Env = append(cmd.Env, "LD_LIBRARY_PATH="+pathEnvVal)
		}
		slog.Debug("mlx subprocess library path", "LD_LIBRARY_PATH", pathEnvVal)
	}

	// Estimate VRAM based on tensor size from manifest
	var vramSize uint64
	if modelManifest, err := manifest.LoadManifest(modelName); err == nil {
		vramSize = uint64(modelManifest.TotalTensorSize())
	} else {
		// Fallback: default to 8GB if manifest can't be loaded
		vramSize = 8 * 1024 * 1024 * 1024
	}

	s := &Server{
		cmd:       cmd,
		port:      port,
		modelName: modelName,
		mode:      mode,
		vramSize:  vramSize,
		done:      make(chan error, 1),
		client:    &http.Client{Timeout: 10 * time.Minute},
	}

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

	slog.Info("starting mlx runner subprocess", "exe", exe, "model", modelName, "port", port, "mode", mode)
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("failed to start mlx runner: %w", err)
	}

	// Reap subprocess when it exits
	go func() {
		err := cmd.Wait()
		s.done <- err
	}()

	// Wait for subprocess to be ready
	if err := s.waitUntilRunning(); err != nil {
		s.Close()
		return nil, err
	}

	return s, nil
}

// ModelPath returns the path to the model.
func (s *Server) ModelPath() string {
	return s.modelName
}

// Load satisfies the LlamaServer interface. MLX models don't need GPU layer assignment.
func (s *Server) Load(ctx context.Context, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, requireFull bool) ([]ml.DeviceID, error) {
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

// waitUntilRunning waits for the subprocess to be ready.
func (s *Server) waitUntilRunning() error {
	ctx := context.Background()
	timeout := time.After(2 * time.Minute)
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case err := <-s.done:
			// Include recent stderr lines for better error context
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

// getLastErr returns the last stderr line.
func (s *Server) getLastErr() string {
	s.lastErrLock.Lock()
	defer s.lastErrLock.Unlock()
	return s.lastErr
}

// WaitUntilRunning satisfies the LlamaServer interface.
func (s *Server) WaitUntilRunning(ctx context.Context) error {
	return nil
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

// VRAMSize returns the estimated VRAM usage.
func (s *Server) VRAMSize() uint64 {
	return s.vramSize
}

// TotalSize returns the total memory usage.
func (s *Server) TotalSize() uint64 {
	return s.vramSize
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
	body, err := json.Marshal(map[string]string{"content": content})
	if err != nil {
		return nil, err
	}

	url := fmt.Sprintf("http://127.0.0.1:%d/tokenize", s.port)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("tokenize failed: %d", resp.StatusCode)
	}

	var result struct {
		Tokens []int `json:"tokens"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return result.Tokens, nil
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
