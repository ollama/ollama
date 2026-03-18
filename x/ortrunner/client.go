package ortrunner

import (
	"bufio"
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
	"sync/atomic"
	"time"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
)

// Client wraps an ORT GenAI runner subprocess to implement llm.LlamaServer.
type Client struct {
	port      int
	modelDir  string
	memory    atomic.Uint64
	done      chan error
	client    *http.Client
	lastErr   string
	lastErrLk sync.Mutex
	mu        sync.Mutex
	cmd       *exec.Cmd
}

// NewClient spawns a new ORT GenAI runner subprocess and waits until it's ready.
func NewClient(modelDir string) (*Client, error) {
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

	// Get the current executable path
	exe, err := os.Executable()
	if err != nil {
		return nil, fmt.Errorf("unable to lookup executable path: %w", err)
	}
	if eval, err := filepath.EvalSymlinks(exe); err == nil {
		exe = eval
	}

	// Spawn subprocess: ollama runner --ortgenai-engine --model <dir> --port <port>
	cmd := exec.Command(exe, "runner", "--ortgenai-engine", "--model", modelDir, "--port", strconv.Itoa(port))
	cmd.Env = os.Environ()

	// Add ORT GenAI library paths to the subprocess search path
	var libPathEnvVar string
	switch runtime.GOOS {
	case "linux":
		libPathEnvVar = "LD_LIBRARY_PATH"
	case "windows":
		libPathEnvVar = "PATH"
	}

	if libPathEnvVar != "" {
		var libraryPaths []string

		// Add paths from OLLAMA_ORT_PATH
		if ortPath, ok := os.LookupEnv("OLLAMA_ORT_PATH"); ok {
			libraryPaths = append(libraryPaths, filepath.SplitList(ortPath)...)
		}

		// Add lib/ollama paths
		if ml.LibOllamaPath != "" {
			libraryPaths = append(libraryPaths, ml.LibOllamaPath)
			if ortDirs, err := filepath.Glob(filepath.Join(ml.LibOllamaPath, "ortgenai*")); err == nil {
				libraryPaths = append(libraryPaths, ortDirs...)
			}
		}

		if existingPath, ok := os.LookupEnv(libPathEnvVar); ok {
			libraryPaths = append(libraryPaths, filepath.SplitList(existingPath)...)
		}

		pathEnvVal := strings.Join(libraryPaths, string(filepath.ListSeparator))

		found := false
		for i := range cmd.Env {
			envName := cmd.Env[i]
			if runtime.GOOS == "windows" {
				envName = strings.ToUpper(envName)
			}
			if strings.HasPrefix(envName, libPathEnvVar+"=") {
				cmd.Env[i] = libPathEnvVar + "=" + pathEnvVal
				found = true
				break
			}
		}
		if !found {
			cmd.Env = append(cmd.Env, libPathEnvVar+"="+pathEnvVal)
		}
		slog.Debug("ortgenai subprocess library path", libPathEnvVar, pathEnvVal)
	}

	c := &Client{
		port:     port,
		modelDir: modelDir,
		done:     make(chan error, 1),
		client:   &http.Client{Timeout: 10 * time.Minute},
		cmd:      cmd,
	}

	// Forward subprocess stdout/stderr to server logs
	stdout, _ := cmd.StdoutPipe()
	stderr, _ := cmd.StderrPipe()
	go func() {
		io.Copy(os.Stderr, stdout) //nolint:errcheck
	}()
	go func() {
		scanner := bufio.NewScanner(stderr)
		for scanner.Scan() {
			line := scanner.Text()
			fmt.Fprintln(os.Stderr, line)
			c.lastErrLk.Lock()
			c.lastErr = line
			c.lastErrLk.Unlock()
		}
	}()

	slog.Info("starting ortgenai runner subprocess", "exe", exe, "model", modelDir, "port", port)
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("failed to start ortgenai runner: %w", err)
	}

	go func() {
		err := cmd.Wait()
		c.done <- err
	}()

	if err := c.waitUntilRunning(); err != nil {
		c.Close()
		return nil, err
	}

	return c, nil
}

func (c *Client) getLastErr() string {
	c.lastErrLk.Lock()
	defer c.lastErrLk.Unlock()
	return c.lastErr
}

func (c *Client) waitUntilRunning() error {
	ctx := context.Background()
	timeout := time.After(5 * time.Minute) // ORT model loading can be slow
	ticker := time.NewTicker(200 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case err := <-c.done:
			errMsg := c.getLastErr()
			if errMsg != "" {
				return fmt.Errorf("ortgenai runner failed: %s (exit: %v)", errMsg, err)
			}
			return fmt.Errorf("ortgenai runner exited unexpectedly: %w", err)
		case <-timeout:
			errMsg := c.getLastErr()
			if errMsg != "" {
				return fmt.Errorf("timeout waiting for ortgenai runner: %s", errMsg)
			}
			return errors.New("timeout waiting for ortgenai runner to start")
		case <-ticker.C:
			if err := c.Ping(ctx); err == nil {
				slog.Info("ortgenai runner is ready", "port", c.port)
				return nil
			}
		}
	}
}

// Close terminates the subprocess.
func (c *Client) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.cmd != nil && c.cmd.Process != nil {
		slog.Info("stopping ortgenai runner subprocess", "pid", c.cmd.Process.Pid)
		c.cmd.Process.Signal(os.Interrupt)
		select {
		case <-c.done:
		case <-time.After(5 * time.Second):
			c.cmd.Process.Kill()
		}
		c.cmd = nil
	}
	return nil
}

// Completion implements llm.LlamaServer.
func (c *Client) Completion(ctx context.Context, req llm.CompletionRequest, fn func(llm.CompletionResponse)) error {
	creq := completionRequest{
		Prompt: req.Prompt,
	}
	if req.Options != nil {
		creq.Options = &completionOpts{
			Temperature: req.Options.Temperature,
			TopP:        req.Options.TopP,
			TopK:        req.Options.TopK,
			NumPredict:  req.Options.NumPredict,
		}
	}

	body, err := json.Marshal(creq)
	if err != nil {
		return err
	}

	httpURL := fmt.Sprintf("http://127.0.0.1:%d/completion", c.port)
	httpReq, err := http.NewRequestWithContext(ctx, "POST", httpURL, strings.NewReader(string(body)))
	if err != nil {
		return err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(httpReq)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("%s", strings.TrimSpace(string(respBody)))
	}

	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		var raw CompletionResponse
		if err := json.Unmarshal(scanner.Bytes(), &raw); err != nil {
			continue
		}
		if raw.Error != nil {
			return *raw.Error
		}

		cresp := llm.CompletionResponse{
			Content:            raw.Content,
			Done:               raw.Done,
			DoneReason:         llm.DoneReason(raw.DoneReason),
			PromptEvalCount:    raw.PromptEvalCount,
			PromptEvalDuration: raw.PromptEvalDuration,
			EvalCount:          raw.EvalCount,
			EvalDuration:       raw.EvalDuration,
		}

		fn(cresp)
		if cresp.Done {
			return nil
		}
	}

	return scanner.Err()
}

// Ping implements llm.LlamaServer.
func (c *Client) Ping(ctx context.Context) error {
	reqURL := fmt.Sprintf("http://127.0.0.1:%d/v1/status", c.port)
	req, err := http.NewRequestWithContext(ctx, "GET", reqURL, nil)
	if err != nil {
		return err
	}
	resp, err := c.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("health check failed: %d", resp.StatusCode)
	}
	return nil
}

// Tokenize implements llm.LlamaServer.
func (c *Client) Tokenize(ctx context.Context, content string) ([]int, error) {
	body, _ := json.Marshal(map[string]string{"content": content})
	reqURL := fmt.Sprintf("http://127.0.0.1:%d/v1/tokenize", c.port)
	req, err := http.NewRequestWithContext(ctx, "POST", reqURL, strings.NewReader(string(body)))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var tokens []int
	if err := json.NewDecoder(resp.Body).Decode(&tokens); err != nil {
		return nil, err
	}
	return tokens, nil
}

// Detokenize implements llm.LlamaServer.
func (c *Client) Detokenize(_ context.Context, _ []int) (string, error) {
	return "", errors.New("not supported")
}

// Embedding implements llm.LlamaServer.
func (c *Client) Embedding(_ context.Context, _ string) ([]float32, int, error) {
	return nil, 0, errors.New("not supported")
}

// GetDeviceInfos implements llm.LlamaServer.
func (c *Client) GetDeviceInfos(_ context.Context) []ml.DeviceInfo {
	return nil
}

// GetPort implements llm.LlamaServer.
func (c *Client) GetPort() int {
	return c.port
}

// HasExited implements llm.LlamaServer.
func (c *Client) HasExited() bool {
	select {
	case <-c.done:
		return true
	default:
		return false
	}
}

// Load implements llm.LlamaServer.
func (c *Client) Load(_ context.Context, _ ml.SystemInfo, _ []ml.DeviceInfo, _ bool) ([]ml.DeviceID, error) {
	return nil, nil
}

// ModelPath implements llm.LlamaServer.
func (c *Client) ModelPath() string {
	return c.modelDir
}

// Pid implements llm.LlamaServer.
func (c *Client) Pid() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.cmd != nil && c.cmd.Process != nil {
		return c.cmd.Process.Pid
	}
	return -1
}

// MemorySize implements llm.LlamaServer.
func (c *Client) MemorySize() (total, vram uint64) {
	mem := c.memory.Load()
	return mem, mem
}

// VRAMByGPU implements llm.LlamaServer.
func (c *Client) VRAMByGPU(_ ml.DeviceID) uint64 {
	return c.memory.Load()
}

// WaitUntilRunning implements llm.LlamaServer.
func (c *Client) WaitUntilRunning(_ context.Context) error {
	return nil
}

// ContextLength implements llm.LlamaServer.
func (c *Client) ContextLength() int {
	return 4096 // default context length for ORT GenAI models
}

var _ llm.LlamaServer = (*Client)(nil)

type completionRequest struct {
	Prompt  string          `json:"prompt"`
	Options *completionOpts `json:"options,omitempty"`
}

type completionOpts struct {
	Temperature float32 `json:"temperature,omitempty"`
	TopP        float32 `json:"top_p,omitempty"`
	TopK        int     `json:"top_k,omitempty"`
	NumPredict  int     `json:"num_predict,omitempty"`
}
