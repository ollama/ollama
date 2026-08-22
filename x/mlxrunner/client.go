package mlxrunner

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

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/x/imagegen/manifest"
)

// Client wraps an MLX runner subprocess to implement llm.LlamaServer for LLM models.
type Client struct {
	port              int
	modelName         string
	contextLength     atomic.Int64
	softContextLength int // recommended limit to avoid poor performance
	memory            atomic.Uint64
	done              chan struct{}
	doneErr           error // valid after done is closed
	client            *http.Client
	status            *llm.StatusWriter
	loadStart         time.Time
	mu                sync.Mutex
	cmd               *exec.Cmd
}

// NewClient prepares a new MLX runner client for LLM models.
// The subprocess is not started until Load() is called.
func NewClient(modelName string, softContextLength int) (*Client, error) {
	if err := checkPlatformSupport(); err != nil {
		return nil, err
	}

	c := &Client{
		modelName:         modelName,
		softContextLength: softContextLength,
		done:              make(chan struct{}),
		client:            http.DefaultClient,
	}

	modelManifest, err := manifest.LoadManifest(modelName)
	if err != nil {
		return nil, err
	}
	c.memory.Store(uint64(modelManifest.TotalTensorSize()))

	return c, nil
}

func checkPlatformSupport() error {
	switch runtime.GOOS {
	case "darwin":
		if runtime.GOARCH != "arm64" {
			return fmt.Errorf("MLX on macOS requires Apple Silicon (arm64), got %s", runtime.GOARCH)
		}
		return nil
	case "linux", "windows":
		return nil
	default:
		return fmt.Errorf("MLX is not supported on %s", runtime.GOOS)
	}
}

// WaitUntilRunning waits for the subprocess to be ready. The load timeout is a
// stall timeout, not a deadline: it is reset every time the runner reports
// forward progress, so a slow but healthy load is not aborted.
func (c *Client) WaitUntilRunning(ctx context.Context) error {
	stallDuration := envconfig.LoadTimeout()    // If no progress happens
	stallTimer := time.Now().Add(stallDuration) // give up if we stall

	slog.Info("waiting for mlx runner to start responding")
	var lastStatus llm.ServerStatus = -1
	var lastPollErr error
	var loadProgress float32
	fullyLoaded := false

	ticker := time.NewTicker(250 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			slog.Warn("client connection closed before mlx runner finished loading, aborting load")
			return fmt.Errorf("timed out waiting for mlx runner to start: %w", ctx.Err())
		case <-c.done:
			if msg := c.status.LastError(); msg != "" {
				return fmt.Errorf("mlx runner failed: %s (exit: %v)", msg, c.doneErr)
			}
			return fmt.Errorf("mlx runner exited unexpectedly: %w", c.doneErr)
		case <-ticker.C:
		}

		if time.Now().After(stallTimer) {
			// A runner that never bound its port leaves nothing on stderr.
			detail := c.status.LastError()
			if detail == "" && lastPollErr != nil {
				detail = lastPollErr.Error()
			}
			return fmt.Errorf("timed out waiting for mlx runner to start - progress %0.2f - %s", loadProgress, detail)
		}

		pollCtx, cancel := context.WithTimeout(ctx, 200*time.Millisecond)
		status, err := c.getServerStatus(pollCtx)
		cancel()
		if err != nil {
			// Not listening yet, or briefly unresponsive. An exited runner is
			// caught by c.done and a wedged one by the stall timer.
			lastPollErr = err
			continue
		}

		if lastStatus != status.Status && status.Status != llm.ServerStatusReady {
			// Only log on status changes
			slog.Info("waiting for mlx runner to become available", "status", status.Status)
		}
		lastStatus = status.Status

		if status.Status == llm.ServerStatusReady {
			c.applyStatus(status)
			slog.Info(fmt.Sprintf("mlx runner started in %0.2f seconds", time.Since(c.loadStart).Seconds()))
			return nil
		}

		// Reset the timer as long as we're making forward progress on the load
		if progress := max(status.Progress, loadProgress); progress != loadProgress {
			loadProgress = progress
			slog.Debug(fmt.Sprintf("model load progress %0.2f", loadProgress))
			stallTimer = time.Now().Add(stallDuration)
		} else if !fullyLoaded && loadProgress >= 1.0 {
			slog.Debug("model load completed, waiting for mlx runner to become available", "status", status.Status)
			stallTimer = time.Now().Add(stallDuration)
			fullyLoaded = true
		}
	}
}

type CompletionRequest struct {
	Prompt      string
	Media       []llm.MediaData
	Options     api.Options
	Logprobs    bool
	TopLogprobs int
}

type CompletionResponse struct {
	Content    string
	Done       bool
	DoneReason int

	PromptEvalCount    int
	PromptEvalDuration time.Duration
	EvalCount          int
	EvalDuration       time.Duration

	Logprobs []llm.Logprob

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
	creq := CompletionRequest{
		Prompt:      req.Prompt,
		Media:       req.Media,
		Logprobs:    req.Logprobs,
		TopLogprobs: req.TopLogprobs,
	}
	if req.Options != nil {
		creq.Options = *req.Options
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
		if errMsg := c.status.LastError(); errMsg != "" {
			return fmt.Errorf("mlx runner failed: %s", errMsg)
		}
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return api.StatusError{StatusCode: resp.StatusCode, ErrorMessage: strings.TrimSpace(string(respBody))}
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
			Logprobs:           raw.Logprobs,
		}

		fn(cresp)
		if cresp.Done {
			return nil
		}
	}

	if err := scanner.Err(); err != nil {
		if errMsg := c.status.LastError(); errMsg != "" {
			return fmt.Errorf("mlx runner failed: %s", errMsg)
		}
		return err
	}
	return nil
}

func (c *Client) Chat(ctx context.Context, req llm.ChatRequest, fn func(llm.ChatResponse)) error {
	return errors.New("MLX runner does not support native llama-server chat")
}

func (c *Client) ApplyChatTemplate(ctx context.Context, req llm.ChatRequest) (string, error) {
	return "", errors.New("MLX runner does not support native llama-server chat templates")
}

func (c *Client) ContextLength() int {
	return int(c.contextLength.Load())
}

func (c *Client) reportedContextLength(modelContextLength int) int {
	if c.softContextLength > 0 && (modelContextLength == 0 || c.softContextLength < modelContextLength) {
		return c.softContextLength
	}
	return modelContextLength
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

	status := llm.NewStatusWriter(os.Stderr)
	c.status = status
	// os/exec serializes Write calls when shared, which keeps the status writer
	// from seeing concurrent stdout/stderr fragments.
	cmd.Stdout = status
	cmd.Stderr = status

	slog.Info("starting mlx runner subprocess", "model", c.modelName, "port", c.port)
	c.loadStart = time.Now()
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
	Status        llm.ServerStatus
	Progress      float32 // fraction of the model loaded, 0.0 to 1.0
	ContextLength int
	Memory        uint64
}

// getServerStatus fetches the runner's health. The runner serves this endpoint
// from the moment it starts listening, reporting its load status in the body,
// so a successful response does not mean the model is ready.
func (c *Client) getServerStatus(ctx context.Context) (statusResponse, error) {
	var status statusResponse

	reqURL := fmt.Sprintf("http://127.0.0.1:%d/v1/status", c.port)
	req, err := http.NewRequestWithContext(ctx, "GET", reqURL, nil)
	if err != nil {
		return status, err
	}
	resp, err := c.client.Do(req)
	if err != nil {
		return status, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return status, fmt.Errorf("health check failed: %d", resp.StatusCode)
	}

	if err := json.NewDecoder(resp.Body).Decode(&status); err != nil {
		return status, err
	}

	return status, nil
}

// Ping implements llm.LlamaServer.
func (c *Client) Ping(ctx context.Context) error {
	status, err := c.getServerStatus(ctx)
	if err != nil {
		return err
	}

	if status.Status != llm.ServerStatusReady {
		return fmt.Errorf("mlx runner not ready: %s", status.Status)
	}

	c.applyStatus(status)

	return nil
}

// applyStatus records the details the scheduler reads back from the runner.
// Only ever called with a ready runner's status: while loading, the runner
// reports no context length, and adopting that would replace what the
// scheduler already knows with zero.
func (c *Client) applyStatus(status statusResponse) {
	c.contextLength.Store(int64(c.reportedContextLength(status.ContextLength)))
	c.memory.Store(status.Memory)
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
