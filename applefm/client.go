// Package applefm runs Apple's on-device Foundation Model through the `fm`
// command line tool that ships with macOS 27 and later. The model weights are
// managed by the operating system, so there is nothing to download: the
// client starts `fm serve` on a local port and translates Ollama chat
// requests to its OpenAI-compatible chat completions endpoint.
package applefm

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
)

const (
	// ModelFormat is the model_format config value that selects this runner.
	ModelFormat = "applefm"

	// RunnerName is reported as the runner for loaded Apple Foundation Models.
	RunnerName = "applefm"

	// ContextLength is the on-device model's context window in tokens.
	ContextLength = 4096
)

// fmPath is the location of the fm tool that ships with macOS.
var fmPath = "/usr/bin/fm"

func errUnsupported(what string) error {
	return api.StatusError{
		StatusCode:   http.StatusBadRequest,
		ErrorMessage: what + " is not supported by Apple Foundation Models",
	}
}

// Available reports whether the on-device model can be used. It returns an
// error describing why not, such as an older macOS release, an unsupported
// device, or Apple Intelligence being turned off.
func Available(ctx context.Context) error {
	if runtime.GOOS != "darwin" {
		return errors.New("apple foundation models require macOS")
	}
	if _, err := os.Stat(fmPath); err != nil {
		return fmt.Errorf("apple foundation models require macOS 27 or later: %w", err)
	}
	out, err := exec.CommandContext(ctx, fmPath, "available").CombinedOutput()
	if err != nil {
		return fmt.Errorf("apple foundation model unavailable: %s", out)
	}
	return nil
}

// Client manages an `fm serve` subprocess and implements llm.LlamaServer.
type Client struct {
	modelName string
	port      int
	baseURL   string // empty until Load
	status    *llm.StatusWriter

	mu      sync.Mutex
	cmd     *exec.Cmd
	done    chan struct{}
	doneErr error // valid after done is closed
}

// NewClient prepares a client for modelName. The subprocess is not started
// until Load is called.
func NewClient(modelName string) *Client {
	return &Client{modelName: modelName, done: make(chan struct{})}
}

// Load starts `fm serve`. The model's memory is owned by the operating system,
// so there is no fit check and no GPU assignment.
func (c *Client) Load(ctx context.Context, _ ml.SystemInfo, _ []ml.DeviceInfo, _ bool) ([]ml.DeviceID, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, fmt.Errorf("failed to find a free port: %w", err)
	}
	port := l.Addr().(*net.TCPAddr).Port
	l.Close()

	cmd := exec.Command(fmPath, "serve", "--host", "127.0.0.1", "--port", strconv.Itoa(port))
	status := llm.NewStatusWriter(os.Stderr)
	cmd.Stdout = status
	cmd.Stderr = status

	c.mu.Lock()
	defer c.mu.Unlock()

	slog.Info("starting apple foundation model server", "port", port)
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("failed to start fm serve: %w", err)
	}
	c.cmd = cmd
	c.port = port
	c.baseURL = "http://127.0.0.1:" + strconv.Itoa(port)
	c.status = status

	go func() {
		c.doneErr = cmd.Wait()
		close(c.done)
	}()

	return nil, nil
}

// WaitUntilRunning waits for `fm serve` to answer health checks.
func (c *Client) WaitUntilRunning(ctx context.Context) error {
	timeout := time.After(envconfig.LoadTimeout())
	ticker := time.NewTicker(50 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-c.done:
			if msg := c.status.LastError(); msg != "" {
				return fmt.Errorf("fm serve failed: %s (exit: %v)", msg, c.doneErr)
			}
			return fmt.Errorf("fm serve exited unexpectedly (run `fm available` and `fm license` to check setup): %w", c.doneErr)
		case <-timeout:
			return errors.New("timeout waiting for fm serve to start")
		case <-ticker.C:
			// fm may accept connections before it can answer, so don't let
			// one health check outlast the load timeout.
			pingCtx, cancel := context.WithTimeout(ctx, time.Second)
			err := c.Ping(pingCtx)
			cancel()
			if err == nil {
				slog.Info("apple foundation model server is ready")
				return nil
			}
		}
	}
}

// Ping implements llm.LlamaServer.
func (c *Client) Ping(ctx context.Context) error {
	if c.baseURL == "" {
		return errors.New("fm serve is not running")
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/health", nil)
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("health check failed: %d", resp.StatusCode)
	}
	return nil
}

// Close stops the subprocess.
func (c *Client) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.cmd != nil && c.cmd.Process != nil {
		slog.Info("stopping apple foundation model server", "pid", c.cmd.Process.Pid)
		c.cmd.Process.Kill()
		<-c.done
		c.cmd = nil
	}
	return nil
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

// Pid implements llm.LlamaServer.
func (c *Client) Pid() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.cmd != nil && c.cmd.Process != nil {
		return c.cmd.Process.Pid
	}
	return -1
}

// ModelPath implements llm.LlamaServer. There are no weights on disk, so the
// scheduler identifies the model by name.
func (c *Client) ModelPath() string { return c.modelName }

func (c *Client) ContextLength() int                             { return ContextLength }
func (c *Client) MemorySize() (total, vram uint64)               { return 0, 0 }
func (c *Client) VRAMByGPU(ml.DeviceID) uint64                   { return 0 }
func (c *Client) GetPort() int                                   { return c.port }
func (c *Client) GetDeviceInfos(context.Context) []ml.DeviceInfo { return nil }

func (c *Client) ApplyChatTemplate(context.Context, llm.ChatRequest) (string, error) {
	return "", errUnsupported("chat template rendering")
}

func (c *Client) Embedding(context.Context, string, [][]byte) ([]float32, int, error) {
	return nil, 0, errUnsupported("embedding")
}

func (c *Client) Tokenize(context.Context, string) ([]int, error) {
	return nil, errUnsupported("tokenization")
}

func (c *Client) Detokenize(context.Context, []int) (string, error) {
	return "", errUnsupported("detokenization")
}

var _ llm.LlamaServer = (*Client)(nil)
