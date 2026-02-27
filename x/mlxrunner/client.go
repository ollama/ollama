package mlxrunner

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
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
	"github.com/ollama/ollama/x/imagegen"
	"github.com/ollama/ollama/x/imagegen/manifest"
)

// Client wraps an MLX runner subprocess to implement llm.LlamaServer for LLM models.
type Client struct {
	port        int
	modelName   string
	vramSize    uint64
	done        chan error
	client      *http.Client
	lastErr     string
	lastErrLock sync.Mutex
	mu          sync.Mutex
	cmd         *exec.Cmd
}

// NewClient spawns a new MLX runner subprocess for LLM models and waits until it's ready.
func NewClient(modelName string) (*Client, error) {
	if err := imagegen.CheckPlatformSupport(); err != nil {
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

	// Get the current executable path
	exe, err := os.Executable()
	if err != nil {
		return nil, fmt.Errorf("unable to lookup executable path: %w", err)
	}
	if eval, err := filepath.EvalSymlinks(exe); err == nil {
		exe = eval
	}

	// Spawn subprocess: ollama runner --mlx-engine --model <name> --port <port>
	cmd := exec.Command(exe, "runner", "--mlx-engine", "--model", modelName, "--port", strconv.Itoa(port))
	cmd.Env = os.Environ()

	// On Linux, set LD_LIBRARY_PATH to include MLX library directories
	if runtime.GOOS == "linux" {
		libraryPaths := []string{ml.LibOllamaPath}
		if mlxDirs, err := filepath.Glob(filepath.Join(ml.LibOllamaPath, "mlx_*")); err == nil {
			libraryPaths = append(libraryPaths, mlxDirs...)
		}

		if existingPath, ok := os.LookupEnv("LD_LIBRARY_PATH"); ok {
			libraryPaths = append(libraryPaths, filepath.SplitList(existingPath)...)
		}

		pathEnvVal := strings.Join(libraryPaths, string(filepath.ListSeparator))

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
		vramSize = 8 * 1024 * 1024 * 1024
	}

	c := &Client{
		port:      port,
		modelName: modelName,
		vramSize:  vramSize,
		done:      make(chan error, 1),
		client:    &http.Client{Timeout: 10 * time.Minute},
		cmd:       cmd,
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
			c.lastErrLock.Lock()
			c.lastErr = line
			c.lastErrLock.Unlock()
		}
	}()

	slog.Info("starting mlx runner subprocess", "exe", exe, "model", modelName, "port", port)
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("failed to start mlx runner: %w", err)
	}

	// Reap subprocess when it exits
	go func() {
		err := cmd.Wait()
		c.done <- err
	}()

	// Wait for subprocess to be ready
	if err := c.waitUntilRunning(); err != nil {
		c.Close()
		return nil, err
	}

	return c, nil
}

func (c *Client) getLastErr() string {
	c.lastErrLock.Lock()
	defer c.lastErrLock.Unlock()
	return c.lastErr
}

func (c *Client) waitUntilRunning() error {
	ctx := context.Background()
	timeout := time.After(2 * time.Minute)
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case err := <-c.done:
			errMsg := c.getLastErr()
			if errMsg != "" {
				return fmt.Errorf("mlx runner failed: %s (exit: %v)", errMsg, err)
			}
			return fmt.Errorf("mlx runner exited unexpectedly: %w", err)
		case <-timeout:
			errMsg := c.getLastErr()
			if errMsg != "" {
				return fmt.Errorf("timeout waiting for mlx runner: %s", errMsg)
			}
			return errors.New("timeout waiting for mlx runner to start")
		case <-ticker.C:
			if err := c.Ping(ctx); err == nil {
				slog.Info("mlx runner is ready", "port", c.port)
				return nil
			}
		}
	}
}

// completionRequest is a properly-tagged version of llm.CompletionRequest for JSON serialization.
type completionRequest struct {
	Prompt  string          `json:"prompt"`
	Options *completionOpts `json:"options,omitempty"`
}

type completionOpts struct {
	Temperature float32 `json:"temperature,omitempty"`
	TopP        float32 `json:"top_p,omitempty"`
	MinP        float32 `json:"min_p,omitempty"`
	TopK        int     `json:"top_k,omitempty"`
	NumPredict  int     `json:"num_predict,omitempty"`
}

// Close terminates the subprocess.
func (c *Client) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.cmd != nil && c.cmd.Process != nil {
		slog.Info("stopping mlx runner subprocess", "pid", c.cmd.Process.Pid)
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
			MinP:        req.Options.MinP,
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
		var raw struct {
			Content            string `json:"content,omitempty"`
			Done               bool   `json:"done"`
			DoneReason         int    `json:"done_reason,omitempty"`
			PromptEvalCount    int    `json:"prompt_eval_count,omitempty"`
			PromptEvalDuration int    `json:"prompt_eval_duration,omitempty"`
			EvalCount          int    `json:"eval_count,omitempty"`
			EvalDuration       int    `json:"eval_duration,omitempty"`
			PeakMemory         uint64 `json:"peak_memory,omitempty"`
		}
		if err := json.Unmarshal(scanner.Bytes(), &raw); err != nil {
			slog.Debug("mlx response parse error", "error", err, "line", string(scanner.Bytes()))
			continue
		}

		cresp := llm.CompletionResponse{
			Content:            raw.Content,
			Done:               raw.Done,
			DoneReason:         llm.DoneReason(raw.DoneReason),
			PromptEvalCount:    raw.PromptEvalCount,
			PromptEvalDuration: time.Duration(raw.PromptEvalDuration),
			EvalCount:          raw.EvalCount,
			EvalDuration:       time.Duration(raw.EvalDuration),
			PeakMemory:         raw.PeakMemory,
		}

		fn(cresp)
		if cresp.Done {
			return nil
		}
	}

	return scanner.Err()
}

func (c *Client) ContextLength() int {
	return math.MaxInt
}

// Detokenize implements llm.LlamaServer.
func (c *Client) Detokenize(ctx context.Context, tokens []int) (string, error) {
	return "", errors.New("not supported")
}

// Embedding implements llm.LlamaServer.
func (c *Client) Embedding(ctx context.Context, input string) ([]float32, int, error) {
	return nil, 0, errors.New("not supported")
}

// GetDeviceInfos implements llm.LlamaServer.
func (c *Client) GetDeviceInfos(ctx context.Context) []ml.DeviceInfo {
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
func (c *Client) Load(ctx context.Context, _ ml.SystemInfo, _ []ml.DeviceInfo, _ bool) ([]ml.DeviceID, error) {
	return nil, nil
}

// ModelPath implements llm.LlamaServer.
func (c *Client) ModelPath() string {
	return c.modelName
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

// Ping implements llm.LlamaServer.
func (c *Client) Ping(ctx context.Context) error {
	reqURL := fmt.Sprintf("http://127.0.0.1:%d/health", c.port)
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
	reqURL := fmt.Sprintf("http://127.0.0.1:%d/v1/tokenize", c.port)
	req, err := http.NewRequestWithContext(ctx, "POST", reqURL, strings.NewReader(content))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "text/plain")

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

// TotalSize implements llm.LlamaServer.
func (c *Client) TotalSize() uint64 {
	return c.vramSize
}

// VRAMByGPU implements llm.LlamaServer.
func (c *Client) VRAMByGPU(id ml.DeviceID) uint64 {
	return c.vramSize
}

// VRAMSize implements llm.LlamaServer.
func (c *Client) VRAMSize() uint64 {
	return c.vramSize
}

// WaitUntilRunning implements llm.LlamaServer.
func (c *Client) WaitUntilRunning(ctx context.Context) error {
	return nil
}

var _ llm.LlamaServer = (*Client)(nil)
