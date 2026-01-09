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
	"strconv"
	"sync"
	"time"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
)

// Server wraps an image generation subprocess to implement llm.LlamaServer.
type Server struct {
	mu        sync.Mutex
	cmd       *exec.Cmd
	port      int
	modelName string
	vramSize  uint64
	done      chan error
	client    *http.Client
}

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
	Content string `json:"content"`
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

	// Get the ollama executable path
	exe, err := os.Executable()
	if err != nil {
		return nil, fmt.Errorf("unable to lookup executable path: %w", err)
	}
	if eval, err := filepath.EvalSymlinks(exe); err == nil {
		exe = eval
	}

	// Spawn subprocess: ollama runner --image-engine --model <path> --port <port>
	cmd := exec.Command(exe, "runner", "--image-engine", "--model", modelName, "--port", strconv.Itoa(port))
	cmd.Env = os.Environ()

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
			slog.Warn("image-runner", "msg", scanner.Text())
		}
	}()

	slog.Info("starting image runner subprocess", "model", modelName, "port", port)
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("failed to start image runner: %w", err)
	}

	s := &Server{
		cmd:       cmd,
		port:      port,
		modelName: modelName,
		vramSize:  EstimateVRAM(modelName),
		done:      make(chan error, 1),
		client:    &http.Client{Timeout: 10 * time.Minute},
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

// Load is called by the scheduler after the server is created.
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
			return fmt.Errorf("subprocess exited: %w", err)
		case <-timeout:
			return errors.New("timeout waiting for image runner to start")
		case <-ticker.C:
			if err := s.Ping(ctx); err == nil {
				slog.Info("image runner is ready", "port", s.port)
				return nil
			}
		}
	}
}

// WaitUntilRunning implements the LlamaServer interface (no-op since NewServer waits).
func (s *Server) WaitUntilRunning(ctx context.Context) error {
	return nil
}

// Completion generates an image from the prompt via the subprocess.
func (s *Server) Completion(ctx context.Context, req llm.CompletionRequest, fn func(llm.CompletionResponse)) error {
	// Build request
	creq := completionRequest{
		Prompt: req.Prompt,
		Width:  1024,
		Height: 1024,
		Steps:  9,
		Seed:   time.Now().UnixNano(),
	}

	if req.Options != nil {
		if req.Options.NumCtx > 0 && req.Options.NumCtx <= 4096 {
			creq.Width = int32(req.Options.NumCtx)
		}
		if req.Options.NumGPU > 0 && req.Options.NumGPU <= 4096 {
			creq.Height = int32(req.Options.NumGPU)
		}
		if req.Options.NumPredict > 0 && req.Options.NumPredict <= 100 {
			creq.Steps = req.Options.NumPredict
		}
		if req.Options.Seed > 0 {
			creq.Seed = int64(req.Options.Seed)
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

	// Stream responses
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		var cresp completionResponse
		if err := json.Unmarshal(scanner.Bytes(), &cresp); err != nil {
			continue
		}
		fn(llm.CompletionResponse{
			Content: cresp.Content,
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

// Embedding is not supported for image generation models.
func (s *Server) Embedding(ctx context.Context, input string) ([]float32, int, error) {
	return nil, 0, errors.New("embedding not supported for image generation models")
}

// Tokenize is not supported for image generation models.
func (s *Server) Tokenize(ctx context.Context, content string) ([]int, error) {
	return nil, errors.New("tokenize not supported for image generation models")
}

// Detokenize is not supported for image generation models.
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

// GetDeviceInfos returns nil since we don't track GPU info.
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
