package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// llamaServerBackend drives a llama-server directly over its native /completion
type llamaServerBackend struct {
	addr   string
	model  string
	mode   string
	client *http.Client
	debug  bool

	cmd *exec.Cmd // non-nil when bench spawned the llama-server
}

// llamaServerReq is the subset of llama-server's /completion request bench uses.
// This mirrors llama.cpp's stable native protocol (not the ollama MLX runner
// wire types), so it is intentionally defined locally.
type llamaServerReq struct {
	Prompt      string  `json:"prompt"`
	NPredict    int     `json:"n_predict"`
	Stream      bool    `json:"stream"`
	CachePrompt bool    `json:"cache_prompt"`
	IgnoreEOS   bool    `json:"ignore_eos"`
	Temperature float64 `json:"temperature"`
	Seed        int     `json:"seed"`
}

type llamaServerTimings struct {
	CacheN      int     `json:"cache_n"`
	PromptN     int     `json:"prompt_n"`
	PromptMS    float64 `json:"prompt_ms"`
	PredictedN  int     `json:"predicted_n"`
	PredictedMS float64 `json:"predicted_ms"`
}

// llamaServerChunk is one streamed SSE event. timings is present on the final
// (stop) chunk.
type llamaServerChunk struct {
	Content string             `json:"content"`
	Stop    bool               `json:"stop"`
	Timings llamaServerTimings `json:"timings"`
}

func newLlamaServerBackend(fOpt flagOptions) *llamaServerBackend {
	return &llamaServerBackend{
		addr:   *fOpt.runner,
		model:  *fOpt.models,
		mode:   *fOpt.mode,
		client: &http.Client{},
		debug:  *fOpt.debug,
	}
}

// newLlamaServerSpawn launches a llama-server on a free port serving ggufPath,
// captures its stderr, and waits for it to become ready.
func newLlamaServerSpawn(fOpt flagOptions, ggufPath string) (*llamaServerBackend, error) {
	bin, err := resolveLlamaServer()
	if err != nil {
		return nil, err
	}
	port, err := freePort()
	if err != nil {
		return nil, fmt.Errorf("could not find a free port: %w", err)
	}

	args := []string{"-m", ggufPath, "--port", strconv.Itoa(port), "-ngl", "99"}
	if *fOpt.numCtx > 0 {
		args = append(args, "-c", strconv.Itoa(*fOpt.numCtx))
	}
	cmd := exec.Command(bin, args...)
	cmd.Env = os.Environ()
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("failed to spawn llama-server: %w", err)
	}

	b := &llamaServerBackend{
		addr:   fmt.Sprintf("127.0.0.1:%d", port),
		model:  *fOpt.models,
		mode:   *fOpt.mode,
		client: &http.Client{},
		debug:  *fOpt.debug,
		cmd:    cmd,
	}
	fmt.Fprintf(os.Stderr, "bench: spawned llama-server (pid %d) on %s for %s\n", cmd.Process.Pid, b.addr, ggufPath)

	if err := b.waitReady(time.Duration(*fOpt.timeout) * time.Second); err != nil {
		b.Cleanup(*fOpt.timeout)
		return nil, fmt.Errorf("llama-server not ready: %w", err)
	}
	return b, nil
}

// resolveLlamaServer locates the llama-server binary: the -ollama flag's sibling
// payload dir, a dev build/ tree, or PATH.
func resolveLlamaServer() (string, error) {
	var candidates []string
	if cwd, err := os.Getwd(); err == nil {
		for dir := cwd; ; {
			candidates = append(candidates, filepath.Join(dir, "build", "lib", "ollama", "llama-server"))
			candidates = append(candidates, filepath.Join(dir, "dist", "lib", "ollama", "llama-server"))
			parent := filepath.Dir(dir)
			if parent == dir {
				break
			}
			dir = parent
		}
	}
	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			return c, nil
		}
	}
	if p, err := exec.LookPath("llama-server"); err == nil {
		return p, nil
	}
	return "", fmt.Errorf("could not locate the llama-server binary; build it (cmake --build build) or put it on PATH")
}

func (b *llamaServerBackend) waitReady(timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for {
		if httpOK(b.client, "http://"+b.addr+"/health") {
			return nil
		}
		if b.cmd != nil && b.cmd.ProcessState != nil && b.cmd.ProcessState.Exited() {
			return fmt.Errorf("llama-server exited before becoming ready")
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("timed out waiting for %s", b.addr)
		}
		time.Sleep(200 * time.Millisecond)
	}
}

func (b *llamaServerBackend) Name() string { return "llama-server" }

func (b *llamaServerBackend) ModelInfo(ctx context.Context, fOpt flagOptions) ModelInfo {
	// llama-server exposes /props with model metadata; for now we report just
	// the label and let the values fall back to "unknown".
	return ModelInfo{Name: b.model}
}

func (b *llamaServerBackend) Complete(ctx context.Context, p completionParams) (completionResult, error) {
	lreq := llamaServerReq{
		Prompt:      p.prompt,
		NPredict:    p.numPredict,
		Stream:      true,
		CachePrompt: b.mode == modeDecode,
		IgnoreEOS:   p.ignoreEOS,
		Temperature: p.temperature,
		Seed:        p.seed,
	}

	body, err := json.Marshal(lreq)
	if err != nil {
		return completionResult{}, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", "http://"+b.addr+"/completion", bytes.NewReader(body))
	if err != nil {
		return completionResult{}, err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	requestStart := time.Now()
	resp, err := b.client.Do(httpReq)
	if err != nil {
		return completionResult{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return completionResult{}, fmt.Errorf("llama-server status %d", resp.StatusCode)
	}

	var res completionResult
	ttftSet := false
	gotTimings := false
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		// llama-server emits SSE: "data: {json}" lines separated by blanks.
		line, ok := strings.CutPrefix(line, "data:")
		if !ok {
			continue
		}
		line = strings.TrimSpace(line)
		if line == "" || line == "[DONE]" {
			continue
		}

		var chunk llamaServerChunk
		if err := json.Unmarshal([]byte(line), &chunk); err != nil {
			continue
		}
		if !ttftSet && chunk.Content != "" {
			res.ttft = time.Since(requestStart)
			ttftSet = true
		}
		if b.debug && chunk.Content != "" {
			fmt.Fprint(os.Stderr, chunk.Content)
		}
		if chunk.Stop {
			res.promptEvalCount = chunk.Timings.CacheN + chunk.Timings.PromptN
			res.promptEvalDuration = msToDuration(chunk.Timings.PromptMS)
			res.evalCount = chunk.Timings.PredictedN
			res.evalDuration = msToDuration(chunk.Timings.PredictedMS)
			gotTimings = true
		}
	}
	if b.debug {
		fmt.Fprintln(os.Stderr)
	}
	if err := scanner.Err(); err != nil {
		return res, err
	}
	if !gotTimings {
		return res, errNoMetrics
	}
	res.totalDuration = time.Since(requestStart)
	return res, nil
}

// Cleanup terminates the llama-server only if bench spawned it; an
// operator-managed server (reached via -runner) is left running.
func (b *llamaServerBackend) Cleanup(int) {
	if b.cmd == nil || b.cmd.Process == nil {
		return
	}
	fmt.Fprintf(os.Stderr, "bench: stopping spawned llama-server (pid %d)\n", b.cmd.Process.Pid)
	_ = b.cmd.Process.Signal(os.Interrupt)
	done := make(chan struct{})
	go func() {
		_ = b.cmd.Wait()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		_ = b.cmd.Process.Kill()
	}
	b.cmd = nil
}

func msToDuration(ms float64) time.Duration {
	return time.Duration(ms * float64(time.Millisecond))
}
