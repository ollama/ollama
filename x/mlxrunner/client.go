package mlxrunner

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
	"sync/atomic"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/x/imagegen"
	"github.com/ollama/ollama/x/imagegen/manifest"
)

// Client wraps an MLX runner subprocess to implement llm.LlamaServer for LLM models.
type Client struct {
	port          int
	modelName     string
	contextLength atomic.Int64
	memory        atomic.Uint64
	done          chan struct{}
	doneErr       error // valid after done is closed
	client        *http.Client
	status        *statusWriter
	mu            sync.Mutex
	cmd           *exec.Cmd
}

// statusWriter captures the last stderr line from the subprocess while
// forwarding all output to os.Stderr. Lines longer than maxStatusLen are
// truncated to the first maxStatusLen bytes.
type statusWriter struct {
	lastErrMsg string
	buf        []byte
	discarding bool
	mu         sync.Mutex
	out        *os.File
}

const maxStatusLen = 256

func (w *statusWriter) Write(b []byte) (int, error) {
	n, err := w.out.Write(b)

	w.mu.Lock()
	defer w.mu.Unlock()

	w.buf = append(w.buf, b...)
	for {
		i := bytes.IndexByte(w.buf, '\n')
		if i < 0 {
			break
		}
		if !w.discarding {
			line := bytes.TrimSpace(w.buf[:i])
			if len(line) > 0 {
				if len(line) > maxStatusLen {
					line = line[:maxStatusLen]
				}
				w.lastErrMsg = string(line)
			}
		}
		w.buf = w.buf[i+1:]
		w.discarding = false
	}
	// if the buffer grows past maxStatusLen without a newline, keep the front
	if len(w.buf) > maxStatusLen {
		if !w.discarding {
			w.lastErrMsg = string(bytes.TrimSpace(w.buf[:maxStatusLen]))
			w.discarding = true
		}
		w.buf = w.buf[:0]
	}

	return n, err
}

func (w *statusWriter) getLastErr() string {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.lastErrMsg
}

// NewClient prepares a new MLX runner client for LLM models.
// The subprocess is not started until Load() is called.
func NewClient(modelName string) (*Client, error) {
	if err := imagegen.CheckPlatformSupport(); err != nil {
		return nil, err
	}

	c := &Client{
		modelName: modelName,
		done:      make(chan struct{}),
		client:    http.DefaultClient,
	}

	modelManifest, err := manifest.LoadManifest(modelName)
	if err != nil {
		return nil, err
	}
	c.memory.Store(uint64(modelManifest.TotalTensorSize()))

	return c, nil
}

// WaitUntilRunning waits for the subprocess to be ready.
func (c *Client) WaitUntilRunning(ctx context.Context) error {
	timeout := time.After(2 * time.Minute)
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-c.done:
			if msg := c.status.getLastErr(); msg != "" {
				return fmt.Errorf("mlx runner failed: %s (exit: %v)", msg, c.doneErr)
			}
			return fmt.Errorf("mlx runner exited unexpectedly: %w", c.doneErr)
		case <-timeout:
			if msg := c.status.getLastErr(); msg != "" {
				return fmt.Errorf("timeout waiting for mlx runner: %s", msg)
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
	Temperature     float32 `json:"temperature,omitempty"`
	TopP            float32 `json:"top_p,omitempty"`
	MinP            float32 `json:"min_p,omitempty"`
	TopK            int     `json:"top_k,omitempty"`
	RepeatLastN     int     `json:"repeat_last_n,omitempty"`
	PresencePenalty float32 `json:"presence_penalty,omitempty"`
	NumPredict      int     `json:"num_predict,omitempty"`
}

type CompletionResponse struct {
	Content    string
	Done       bool
	DoneReason int

	PromptEvalCount    int
	PromptEvalDuration time.Duration
	EvalCount          int
	EvalDuration       time.Duration

	Error *api.StatusError
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
			Temperature:     req.Options.Temperature,
			TopP:            req.Options.TopP,
			MinP:            req.Options.MinP,
			TopK:            req.Options.TopK,
			RepeatLastN:     req.Options.RepeatLastN,
			PresencePenalty: req.Options.PresencePenalty,
			NumPredict:      req.Options.NumPredict,
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
		if errMsg := c.status.getLastErr(); errMsg != "" {
			return fmt.Errorf("mlx runner failed: %s", errMsg)
		}
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
			slog.Debug("mlx response parse error", "error", err, "line", string(scanner.Bytes()))
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

	if err := scanner.Err(); err != nil {
		if errMsg := c.status.getLastErr(); errMsg != "" {
			return fmt.Errorf("mlx runner failed: %s", errMsg)
		}
		return err
	}
	return nil
}

func (c *Client) ContextLength() int {
	return int(c.contextLength.Load())
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

// Load checks whether the model fits in GPU memory and starts the subprocess.
func (c *Client) Load(ctx context.Context, _ ml.SystemInfo, gpus []ml.DeviceInfo, requireFull bool) ([]ml.DeviceID, error) {
	if len(gpus) > 0 {
		modelSize := c.memory.Load()
		// We currently only use the first GPU with MLX
		available := gpus[0].FreeMemory
		overhead := gpus[0].MinimumMemory() + envconfig.GpuOverhead()
		if available > overhead {
			available -= overhead
		} else {
			available = 0
		}

		if modelSize > available {
			if requireFull {
				return nil, llm.ErrLoadRequiredFull
			}
			return nil, fmt.Errorf("model requires %s but only %s are available (after %s overhead)", format.HumanBytes2(modelSize), format.HumanBytes2(available), format.HumanBytes2(overhead))
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
	c.port = port

	// Get the current executable path
	exe, err := os.Executable()
	if err != nil {
		return nil, fmt.Errorf("unable to lookup executable path: %w", err)
	}
	if eval, err := filepath.EvalSymlinks(exe); err == nil {
		exe = eval
	}

	// Spawn subprocess: ollama runner --mlx-engine --model <name> --port <port>
	cmd := exec.Command(exe, "runner", "--mlx-engine", "--model", c.modelName, "--port", strconv.Itoa(port))
	cmd.Env = os.Environ()

	// Set library path environment variable for MLX libraries
	// Linux: LD_LIBRARY_PATH, Windows: PATH
	var libPathEnvVar string
	switch runtime.GOOS {
	case "linux":
		libPathEnvVar = "LD_LIBRARY_PATH"
	case "windows":
		libPathEnvVar = "PATH"
	}

	if libPathEnvVar != "" {
		libraryPaths := []string{ml.LibOllamaPath}
		if mlxDirs, err := filepath.Glob(filepath.Join(ml.LibOllamaPath, "mlx_*")); err == nil {
			libraryPaths = append(libraryPaths, mlxDirs...)
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
		slog.Debug("mlx subprocess library path", libPathEnvVar, pathEnvVal)
	}

	// Point MLX's JIT compiler at our bundled CUDA runtime headers.
	// MLX resolves headers via $CUDA_PATH/include/*.h (and checks CUDA_HOME first).
	// Always use bundled headers to avoid version mismatches with any
	// system-installed CUDA toolkit.
	if mlxDirs, err := filepath.Glob(filepath.Join(ml.LibOllamaPath, "mlx_cuda_*")); err == nil {
		for _, d := range mlxDirs {
			if _, err := os.Stat(filepath.Join(d, "include")); err == nil {
				setEnv(cmd, "CUDA_PATH", d)
				setEnv(cmd, "CUDA_HOME", d)
				slog.Debug("mlx subprocess CUDA headers", "CUDA_PATH", d)
				break
			}
		}
	}

	c.cmd = cmd

	// Forward subprocess stdout/stderr to server logs
	stdout, _ := cmd.StdoutPipe()
	stderr, _ := cmd.StderrPipe()
	status := &statusWriter{out: os.Stderr}
	c.status = status
	go func() {
		io.Copy(os.Stderr, stdout) //nolint:errcheck
	}()
	go func() {
		io.Copy(status, stderr) //nolint:errcheck
	}()

	slog.Info("starting mlx runner subprocess", "model", c.modelName, "port", c.port)
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("failed to start mlx runner: %w", err)
	}

	// Reap subprocess when it exits
	go func() {
		c.doneErr = cmd.Wait()
		close(c.done)
	}()

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

type statusResponse struct {
	Status        int
	Progress      int
	ContextLength int
	Memory        uint64
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

	var status statusResponse
	if err := json.NewDecoder(resp.Body).Decode(&status); err != nil {
		return err
	}

	c.contextLength.Store(int64(status.ContextLength))
	c.memory.Store(status.Memory)

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

func (c *Client) currentMemory() uint64 {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	c.Ping(ctx) //nolint:errcheck
	return c.memory.Load()
}

// MemorySize implements llm.LlamaServer.
func (c *Client) MemorySize() (total, vram uint64) {
	mem := c.currentMemory()
	return mem, mem
}

// VRAMByGPU implements llm.LlamaServer.
func (c *Client) VRAMByGPU(id ml.DeviceID) uint64 {
	return c.currentMemory()
}

var _ llm.LlamaServer = (*Client)(nil)

// setEnv sets or replaces an environment variable in cmd.Env.
func setEnv(cmd *exec.Cmd, key, value string) {
	entry := key + "=" + value
	prefix := strings.ToUpper(key + "=")
	for i, e := range cmd.Env {
		if strings.HasPrefix(strings.ToUpper(e), prefix) {
			cmd.Env[i] = entry
			return
		}
	}
	cmd.Env = append(cmd.Env, entry)
}
