package imagegen

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
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
)

// Server wraps an image generation subprocess to implement llm.LlamaServer.
type Server struct {
	mu          sync.Mutex
	cmd         *exec.Cmd
	port        int
	modelName   string
	vramSize    uint64
	done        chan error
	client      *http.Client
	stderrLines []string // Recent stderr lines for error reporting (max 10)
	stderrLock  sync.Mutex
}

const maxStderrLines = 10

// completionRequest is sent to the subprocess
type completionRequest struct {
	Prompt string `json:"prompt"`
	Width  int32  `json:"width,omitempty"`
	Height int32  `json:"height,omitempty"`
	Steps  int    `json:"steps,omitempty"`
	Seed   int64  `json:"seed,omitempty"`
}

// completionResponse is received from the subprocess
type completionResponse struct {
	Content string `json:"content,omitempty"`
	Image   string `json:"image,omitempty"`
	Done    bool   `json:"done"`
}

// NewServer spawns a new image generation subprocess and waits until it's ready.
func NewServer(modelName string) (*Server, error) {
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

	// Get the ollama-mlx executable path (in same directory as current executable)
	exe, err := os.Executable()
	if err != nil {
		return nil, fmt.Errorf("unable to lookup executable path: %w", err)
	}
	if eval, err := filepath.EvalSymlinks(exe); err == nil {
		exe = eval
	}
	mlxExe := filepath.Join(filepath.Dir(exe), "ollama-mlx")

	// Spawn subprocess: ollama-mlx runner --image-engine --model <path> --port <port>
	cmd := exec.Command(mlxExe, "runner", "--image-engine", "--model", modelName, "--port", strconv.Itoa(port))
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

	s := &Server{
		cmd:       cmd,
		port:      port,
		modelName: modelName,
		vramSize:  EstimateVRAM(modelName),
		done:      make(chan error, 1),
		client:    &http.Client{Timeout: 10 * time.Minute},
	}

	// Forward subprocess stdout/stderr to server logs
	stdout, _ := cmd.StdoutPipe()
	stderr, _ := cmd.StderrPipe()
	go func() {
		scanner := bufio.NewScanner(stdout)
		for scanner.Scan() {
			slog.Info("image-runner", "msg", scanner.Text())
		}
	}()
	go func() {
		scanner := bufio.NewScanner(stderr)
		for scanner.Scan() {
			line := scanner.Text()
			slog.Warn("image-runner", "msg", line)
			// Capture recent stderr lines for error reporting
			s.stderrLock.Lock()
			s.stderrLines = append(s.stderrLines, line)
			if len(s.stderrLines) > maxStderrLines {
				s.stderrLines = s.stderrLines[1:]
			}
			s.stderrLock.Unlock()
		}
	}()

	slog.Info("starting ollama-mlx image runner subprocess", "exe", mlxExe, "model", modelName, "port", port)
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("failed to start image runner: %w", err)
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

// Load is a no-op for image generation models.
// Unlike LLM models, imagegen models are loaded by the subprocess at startup
// rather than through this interface method.
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
			stderrContext := s.getStderrContext()
			if stderrContext != "" {
				return fmt.Errorf("image runner failed: %s (exit: %v)", stderrContext, err)
			}
			return fmt.Errorf("image runner exited unexpectedly: %w", err)
		case <-timeout:
			stderrContext := s.getStderrContext()
			if stderrContext != "" {
				return fmt.Errorf("timeout waiting for image runner: %s", stderrContext)
			}
			return errors.New("timeout waiting for image runner to start")
		case <-ticker.C:
			if err := s.Ping(ctx); err == nil {
				slog.Info("image runner is ready", "port", s.port)
				return nil
			}
		}
	}
}

// getStderrContext returns recent stderr lines joined as a single string.
func (s *Server) getStderrContext() string {
	s.stderrLock.Lock()
	defer s.stderrLock.Unlock()
	if len(s.stderrLines) == 0 {
		return ""
	}
	return strings.Join(s.stderrLines, "; ")
}

// WaitUntilRunning is a no-op for image generation models.
// NewServer already blocks until the subprocess is ready, so this method
// returns immediately. Required by the llm.LlamaServer interface.
func (s *Server) WaitUntilRunning(ctx context.Context) error {
	return nil
}

// Completion generates an image from the prompt via the subprocess.
func (s *Server) Completion(ctx context.Context, req llm.CompletionRequest, fn func(llm.CompletionResponse)) error {
	// Build request - let the model apply its own defaults for unspecified values
	creq := completionRequest{
		Prompt: req.Prompt,
		Seed:   time.Now().UnixNano(),
	}

	// Parse size string (OpenAI format: "WxH") - only set if provided
	if req.Size != "" {
		if w, h := parseSize(req.Size); w > 0 && h > 0 {
			creq.Width = w
			creq.Height = h
		}
	}

	// Encode request body
	body, err := json.Marshal(creq)
	if err != nil {
		return err
	}

	// Send request to subprocess
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
		return fmt.Errorf("completion request failed: %d", resp.StatusCode)
	}

	// Stream responses - use large buffer for base64 image data
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 1024*1024), 16*1024*1024) // 16MB max
	for scanner.Scan() {
		var cresp completionResponse
		if err := json.Unmarshal(scanner.Bytes(), &cresp); err != nil {
			continue
		}

		content := cresp.Content
		// If this is the final response with an image, encode it in the content
		if cresp.Done && cresp.Image != "" {
			content = "IMAGE_BASE64:" + cresp.Image
		}

		fn(llm.CompletionResponse{
			Content: content,
			Done:    cresp.Done,
		})
		if cresp.Done {
			break
		}
	}

	return scanner.Err()
}

// Close terminates the subprocess.
func (s *Server) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.cmd != nil && s.cmd.Process != nil {
		slog.Info("stopping image runner subprocess", "pid", s.cmd.Process.Pid)
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

// Embedding returns an error as image generation models don't produce embeddings.
// Required by the llm.LlamaServer interface.
func (s *Server) Embedding(ctx context.Context, input string) ([]float32, int, error) {
	return nil, 0, errors.New("embedding not supported for image generation models")
}

// Tokenize returns an error as image generation uses internal tokenization.
// Required by the llm.LlamaServer interface.
func (s *Server) Tokenize(ctx context.Context, content string) ([]int, error) {
	return nil, errors.New("tokenize not supported for image generation models")
}

// Detokenize returns an error as image generation uses internal tokenization.
// Required by the llm.LlamaServer interface.
func (s *Server) Detokenize(ctx context.Context, tokens []int) (string, error) {
	return "", errors.New("detokenize not supported for image generation models")
}

// Pid returns the subprocess PID.
func (s *Server) Pid() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.cmd != nil && s.cmd.Process != nil {
		return s.cmd.Process.Pid
	}
	return -1
}

// GetPort returns the subprocess port.
func (s *Server) GetPort() int {
	return s.port
}

// GetDeviceInfos returns nil as GPU tracking is handled by the subprocess.
// Required by the llm.LlamaServer interface.
func (s *Server) GetDeviceInfos(ctx context.Context) []ml.DeviceInfo {
	return nil
}

// HasExited returns true if the subprocess has exited.
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

// parseSize parses an OpenAI-style size string "WxH" into width and height.
func parseSize(size string) (int32, int32) {
	parts := strings.Split(size, "x")
	if len(parts) != 2 {
		return 0, 0
	}
	w, _ := strconv.Atoi(parts[0])
	h, _ := strconv.Atoi(parts[1])
	return int32(w), int32(h)
}
