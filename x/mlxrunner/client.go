package mlxrunner

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"math"
	"net"
	"net/http"
	"net/url"
	"os/exec"
	"strconv"
	"strings"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
)

type Client struct {
	Port int
	*exec.Cmd
}

func (c *Client) JoinPath(path string) string {
	return (&url.URL{
		Scheme: "http",
		Host:   net.JoinHostPort("127.0.0.1", strconv.Itoa(c.Port)),
	}).JoinPath(path).String()
}

func (c *Client) CheckError(w *http.Response) error {
	if w.StatusCode >= 400 {
		return errors.New(w.Status)
	}
	return nil
}

// Close implements llm.LlamaServer.
func (c *Client) Close() error {
	return c.Cmd.Process.Kill()
}

// Completion implements llm.LlamaServer.
func (c *Client) Completion(ctx context.Context, req llm.CompletionRequest, fn func(llm.CompletionResponse)) error {
	var b bytes.Buffer
	if err := json.NewEncoder(&b).Encode(req); err != nil {
		return err
	}

	w, err := http.Post(c.JoinPath("/v1/completions"), "application/json", &b)
	if err != nil {
		return err
	}
	defer w.Body.Close()

	if err := c.CheckError(w); err != nil {
		return err
	}

	scanner := bufio.NewScanner(w.Body)
	for scanner.Scan() {
		bts := scanner.Bytes()

		var resp llm.CompletionResponse
		if err := json.Unmarshal(bts, &resp); err != nil {
			return err
		}

		fn(resp)
	}

	return nil
}

func (c *Client) ContextLength() int {
	return math.MaxInt
}

// Detokenize implements llm.LlamaServer.
func (c *Client) Detokenize(ctx context.Context, tokens []int) (string, error) {
	panic("unimplemented")
}

// Embedding implements llm.LlamaServer.
func (c *Client) Embedding(ctx context.Context, input string) ([]float32, int, error) {
	panic("unimplemented")
}

// GetDeviceInfos implements llm.LlamaServer.
func (c *Client) GetDeviceInfos(ctx context.Context) []ml.DeviceInfo {
	panic("unimplemented")
}

// GetPort implements llm.LlamaServer.
func (c *Client) GetPort() int {
	return c.Port
}

// HasExited implements llm.LlamaServer.
func (c *Client) HasExited() bool {
	panic("unimplemented")
}

// Load implements llm.LlamaServer.
func (c *Client) Load(ctx context.Context, _ ml.SystemInfo, _ []ml.DeviceInfo, _ bool) ([]ml.DeviceID, error) {
	w, err := http.Post(c.JoinPath("/v1/models"), "application/json", nil)
	if err != nil {
		return nil, err
	}
	defer w.Body.Close()

	return []ml.DeviceID{}, nil
}

// ModelPath implements llm.LlamaServer.
func (c *Client) ModelPath() string {
	panic("unimplemented")
}

// Pid implements llm.LlamaServer.
func (c *Client) Pid() int {
	panic("unimplemented")
}

// Ping implements llm.LlamaServer.
func (c *Client) Ping(ctx context.Context) error {
	w, err := http.Get(c.JoinPath("/v1/status"))
	if err != nil {
		return err
	}
	defer w.Body.Close()

	return nil
}

// Tokenize implements llm.LlamaServer.
func (c *Client) Tokenize(ctx context.Context, content string) ([]int, error) {
	w, err := http.Post(c.JoinPath("/v1/tokenize"), "text/plain", strings.NewReader(content))
	if err != nil {
		return nil, err
	}
	defer w.Body.Close()

	var tokens []int
	if err := json.NewDecoder(w.Body).Decode(&tokens); err != nil {
		return nil, err
	}

	return tokens, nil
}

// TotalSize implements llm.LlamaServer.
func (c *Client) TotalSize() uint64 {
	panic("unimplemented")
}

// VRAMByGPU implements llm.LlamaServer.
func (c *Client) VRAMByGPU(id ml.DeviceID) uint64 {
	panic("unimplemented")
}

// VRAMSize implements llm.LlamaServer.
func (c *Client) VRAMSize() uint64 {
	panic("unimplemented")
}

// WaitUntilRunning implements llm.LlamaServer.
func (c *Client) WaitUntilRunning(ctx context.Context) error {
	panic("unimplemented")
}

var _ llm.LlamaServer = (*Client)(nil)
