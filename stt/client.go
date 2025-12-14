package stt

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"sync"
	"time"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/whisper"
)

// WhisperClient connects to a whisperrunner subprocess
type WhisperClient struct {
	baseURL   string
	client    *http.Client
	modelPath string
	cmd       *exec.Cmd
	port      int
	mu        sync.RWMutex
	running   bool
}

// SubprocessConfig for launching whisperrunner
type SubprocessConfig struct {
	ModelPath  string
	Port       int
	UseGPU     bool
	FlashAttn  bool
	GPUDevice  int
	NumThreads int
}

// NewWhisperClient creates a client that connects to an existing subprocess
func NewWhisperClient(baseURL string, modelPath string) *WhisperClient {
	return &WhisperClient{
		baseURL:   baseURL,
		modelPath: modelPath,
		client: &http.Client{
			Timeout: 10 * time.Minute,
			Transport: &http.Transport{
				MaxIdleConns:        10,
				IdleConnTimeout:     90 * time.Second,
				DisableCompression:  true,
				MaxIdleConnsPerHost: 10,
			},
		},
		running: true,
	}
}

// LaunchSubprocess starts a new whisperrunner subprocess
func LaunchSubprocess(ctx context.Context, config SubprocessConfig) (*WhisperClient, error) {
	// Find an available port if not specified
	port := config.Port
	if port == 0 {
		listener, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			return nil, fmt.Errorf("failed to find available port: %w", err)
		}
		port = listener.Addr().(*net.TCPAddr).Port
		listener.Close()
	}

	// Find the whisperrunner executable
	exePath, err := os.Executable()
	if err != nil {
		return nil, err
	}

	// Build arguments
	args := []string{
		"whisper-runner",
		"--model", config.ModelPath,
		"--port", fmt.Sprintf("%d", port),
	}
	if config.UseGPU {
		args = append(args, "--gpu")
	} else {
		args = append(args, "--gpu=false")
	}
	if config.FlashAttn {
		args = append(args, "--flash-attn")
	}
	if config.GPUDevice > 0 {
		args = append(args, "--gpu-device", fmt.Sprintf("%d", config.GPUDevice))
	}
	if config.NumThreads > 0 {
		args = append(args, "--threads", fmt.Sprintf("%d", config.NumThreads))
	}

	cmd := exec.CommandContext(ctx, exePath, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("failed to start subprocess: %w", err)
	}

	client := &WhisperClient{
		baseURL:   fmt.Sprintf("http://127.0.0.1:%d", port),
		modelPath: config.ModelPath,
		cmd:       cmd,
		port:      port,
		client: &http.Client{
			Timeout: 10 * time.Minute,
		},
	}

	// Wait for subprocess to be ready
	deadline := time.Now().Add(60 * time.Second)
	for time.Now().Before(deadline) {
		if err := client.Ping(ctx); err == nil {
			client.running = true
			return client, nil
		}
		select {
		case <-ctx.Done():
			cmd.Process.Kill()
			return nil, ctx.Err()
		case <-time.After(100 * time.Millisecond):
		}
	}

	cmd.Process.Kill()
	return nil, errors.New("subprocess failed to start in time")
}

func (c *WhisperClient) ModelPath() string { return c.modelPath }

func (c *WhisperClient) Load(ctx context.Context, gpus []ml.DeviceInfo) error {
	body, _ := json.Marshal(map[string]any{"gpus": gpus})
	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/load", bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("load failed: %s", string(body))
	}
	return nil
}

func (c *WhisperClient) Transcribe(ctx context.Context, req TranscribeRequest) (*TranscribeResponse, error) {
	// Build internal request format
	internalReq := map[string]any{
		"samples":        req.Samples,
		"language":       req.Language,
		"translate":      req.Translate,
		"initial_prompt": req.InitialPrompt,
		"temperature":    req.Temperature,
		"no_timestamps":  req.NoTimestamps,
		"options":        req.Options,
	}

	body, err := json.Marshal(internalReq)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/transcribe", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		var errResp struct {
			Error string `json:"error"`
		}
		json.NewDecoder(resp.Body).Decode(&errResp)
		return nil, fmt.Errorf("transcription failed: %s", errResp.Error)
	}

	// Parse response - convert from API format to internal format
	var apiResp struct {
		Text     string `json:"text"`
		Language string `json:"language"`
		Segments []struct {
			Start float64 `json:"start"`
			End   float64 `json:"end"`
			Text  string  `json:"text"`
		} `json:"segments"`
		ProcessingDuration time.Duration `json:"processing_duration"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&apiResp); err != nil {
		return nil, err
	}

	// Convert to internal response format
	segments := make([]whisper.Segment, len(apiResp.Segments))
	for i, seg := range apiResp.Segments {
		segments[i] = whisper.Segment{
			Start: time.Duration(seg.Start * float64(time.Second)),
			End:   time.Duration(seg.End * float64(time.Second)),
			Text:  seg.Text,
		}
	}

	return &TranscribeResponse{
		Segments: segments,
		Language: apiResp.Language,
		Duration: apiResp.ProcessingDuration,
	}, nil
}

func (c *WhisperClient) DetectLanguage(ctx context.Context, samples []float32) (string, float32, error) {
	// Not implemented for subprocess mode - would need separate endpoint
	return "", 0, errors.New("detect language not supported in subprocess mode")
}

func (c *WhisperClient) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.running = false
	if c.cmd != nil && c.cmd.Process != nil {
		c.cmd.Process.Signal(os.Interrupt)
		done := make(chan error, 1)
		go func() {
			done <- c.cmd.Wait()
		}()

		select {
		case <-done:
		case <-time.After(5 * time.Second):
			c.cmd.Process.Kill()
		}
	}
	return nil
}

func (c *WhisperClient) Ping(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, "GET", c.baseURL+"/health", nil)
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

	var status struct {
		Status string `json:"status"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&status); err != nil {
		return err
	}

	if status.Status != "ready" {
		return fmt.Errorf("subprocess not ready: %s", status.Status)
	}
	return nil
}

func (c *WhisperClient) GetModelInfo() whisper.ModelInfo {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	req, _ := http.NewRequestWithContext(ctx, "GET", c.baseURL+"/info", nil)
	resp, err := c.client.Do(req)
	if err != nil {
		return whisper.ModelInfo{}
	}
	defer resp.Body.Close()

	var info struct {
		Multilingual bool `json:"multilingual"`
		VocabSize    int  `json:"vocab_size"`
	}
	json.NewDecoder(resp.Body).Decode(&info)

	return whisper.ModelInfo{
		Multilingual: info.Multilingual,
		VocabSize:    info.VocabSize,
	}
}

func (c *WhisperClient) IsMultilingual() bool {
	return c.GetModelInfo().Multilingual
}

// IsRunning returns true if the subprocess is still running
func (c *WhisperClient) IsRunning() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.running
}
