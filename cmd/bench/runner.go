package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/x/mlxrunner/wire"
)

const (
	backendMLX   = "mlx"
	backendLlama = "llama"
)

// newDirectBackend builds the right direct backend for -runner/-spawn. -spawn
// always launches an MLX runner. For -runner host:port it probes the endpoint
// and auto-detects MLX runner vs llama-server.
func newDirectBackend(fOpt flagOptions) (benchBackend, error) {
	if *fOpt.spawn {
		// The spawn target is chosen from -model: a GGUF (a .gguf file path, or
		// an ollama model name that resolves to a GGUF model) launches a
		// llama-server; anything else is treated as an MLX model name and
		// launches the MLX runner.
		if ggufPath, ok := resolveGGUF(*fOpt.models); ok {
			return newLlamaServerSpawn(fOpt, ggufPath)
		}
		return newRunnerBackend(fOpt)
	}

	client := &http.Client{}
	kind, err := detectRunnerKind(*fOpt.runner, client, time.Duration(*fOpt.timeout)*time.Second, *fOpt.debug)
	if err != nil {
		return nil, err
	}
	switch kind {
	case backendMLX:
		return newRunnerBackend(fOpt)
	case backendLlama:
		return newLlamaServerBackend(fOpt), nil
	default:
		return nil, fmt.Errorf("could not determine backend at %s", *fOpt.runner)
	}
}

// detectRunnerKind probes addr until it identifies an MLX runner (GET
// /v1/status returns 200) or a llama-server (GET /props returns 200), or the
// timeout elapses. This also serves as the readiness wait when the operator
// started the runner under a profiler just before launching bench.
func detectRunnerKind(addr string, client *http.Client, timeout time.Duration, debug bool) (string, error) {
	deadline := time.Now().Add(timeout)
	for {
		if httpOK(client, "http://"+addr+"/v1/status") {
			if debug {
				fmt.Fprintf(os.Stderr, "bench: detected MLX runner at %s\n", addr)
			}
			return backendMLX, nil
		}
		if httpOK(client, "http://"+addr+"/props") {
			if debug {
				fmt.Fprintf(os.Stderr, "bench: detected llama-server at %s\n", addr)
			}
			return backendLlama, nil
		}
		if time.Now().After(deadline) {
			return "", fmt.Errorf("no MLX runner (/v1/status) or llama-server (/props) responded at %s", addr)
		}
		time.Sleep(300 * time.Millisecond)
	}
}

// httpOK reports whether a GET to url returns 200.
func httpOK(client *http.Client, url string) bool {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return false
	}
	resp, err := client.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

// runnerBackend drives an MLX runner subprocess directly over its HTTP API,
// bypassing ollama serve. The runner is either already running (started under a
// profiler by the operator) and reached via -runner host:port, or spawned by
// bench when -spawn is set.
type runnerBackend struct {
	addr   string // host:port
	model  string
	client *http.Client
	debug  bool

	cmd *exec.Cmd // non-nil when bench spawned the runner
}

// runnerStatus mirrors the MLX runner's GET /v1/status response.
type runnerStatus struct {
	Status        int
	Progress      int
	ContextLength int
	Memory        uint64
}

func newRunnerBackend(fOpt flagOptions) (*runnerBackend, error) {
	model := *fOpt.models
	if strings.Contains(model, ",") {
		fmt.Fprintf(os.Stderr, "WARNING: direct runner mode serves a single model; using %q\n", model)
	}
	if *fOpt.imageFile != "" {
		fmt.Fprintf(os.Stderr, "WARNING: images are not supported in direct runner mode; ignoring -image\n")
	}

	b := &runnerBackend{
		model:  model,
		client: &http.Client{},
		debug:  *fOpt.debug,
	}

	if *fOpt.runner != "" {
		b.addr = *fOpt.runner
	} else {
		if err := b.spawn(fOpt); err != nil {
			return nil, err
		}
	}

	if err := b.waitReady(time.Duration(*fOpt.timeout) * time.Second); err != nil {
		b.Cleanup(*fOpt.timeout)
		return nil, fmt.Errorf("mlx runner not ready: %w", err)
	}
	return b, nil
}

// spawn launches `ollama runner --mlx-engine` on a free port and captures its
// stderr. It relies on the runner self-configuring its MLX libraries (dev
// builds resolve them relative to the executable); for non-dev installs the
// library path may need to be set in the environment.
func (b *runnerBackend) spawn(fOpt flagOptions) error {
	port, err := freePort()
	if err != nil {
		return fmt.Errorf("could not find a free port: %w", err)
	}
	exe, err := resolveOllama(fOpt)
	if err != nil {
		return err
	}

	cmd := exec.Command(exe, "runner", "--mlx-engine", "--model", b.model, "--port", strconv.Itoa(port))
	cmd.Env = os.Environ()
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to spawn mlx runner: %w", err)
	}
	b.cmd = cmd
	b.addr = fmt.Sprintf("127.0.0.1:%d", port)
	fmt.Fprintf(os.Stderr, "bench: spawned mlx runner (pid %d) on %s\n", cmd.Process.Pid, b.addr)
	return nil
}

func (b *runnerBackend) Name() string { return "mlx-runner" }

func (b *runnerBackend) status(ctx context.Context) (runnerStatus, error) {
	var st runnerStatus
	req, err := http.NewRequestWithContext(ctx, "GET", "http://"+b.addr+"/v1/status", nil)
	if err != nil {
		return st, err
	}
	resp, err := b.client.Do(req)
	if err != nil {
		return st, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return st, fmt.Errorf("status %d", resp.StatusCode)
	}
	err = json.NewDecoder(resp.Body).Decode(&st)
	return st, err
}

func (b *runnerBackend) waitReady(timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		_, err := b.status(ctx)
		cancel()
		if err == nil {
			return nil
		}
		if b.cmd != nil && b.cmd.ProcessState != nil && b.cmd.ProcessState.Exited() {
			return fmt.Errorf("runner exited before becoming ready")
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("timed out waiting for %s", b.addr)
		}
		time.Sleep(200 * time.Millisecond)
	}
}

func (b *runnerBackend) ModelInfo(ctx context.Context, fOpt flagOptions) ModelInfo {
	info := ModelInfo{Name: b.model}
	if *fOpt.numCtx > 0 {
		fmt.Fprintf(os.Stderr, "WARNING: -num-ctx is ignored in direct runner mode (context is fixed at model load)\n")
	}
	st, err := b.status(ctx)
	if err != nil {
		fmt.Fprintf(os.Stderr, "WARNING: could not fetch runner status: %v\n", err)
		return info
	}
	info.SizeBytes = int64(st.Memory)
	info.VRAMBytes = int64(st.Memory)
	info.NumCtx = int64(st.ContextLength)
	return info
}

func (b *runnerBackend) Complete(ctx context.Context, p completionParams) (completionResult, error) {
	creq := wire.CompletionRequest{
		Prompt:    p.prompt,
		IgnoreEOS: p.ignoreEOS,
		Options: api.Options{
			Runner:     api.Runner{NumCtx: p.numCtx},
			Seed:       p.seed,
			NumPredict: p.numPredict,
		},
	}
	creq.Options.Temperature = float32(p.temperature)

	body, err := json.Marshal(creq)
	if err != nil {
		return completionResult{}, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", "http://"+b.addr+"/v1/completions", strings.NewReader(string(body)))
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
		return completionResult{}, fmt.Errorf("runner returned status %d", resp.StatusCode)
	}

	var res completionResult
	ttftSet := false
	gotDone := false
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for scanner.Scan() {
		var cr wire.CompletionResponse
		if err := json.Unmarshal(scanner.Bytes(), &cr); err != nil {
			continue
		}
		if cr.Error != nil {
			return res, fmt.Errorf("runner error: %s", cr.Error.ErrorMessage)
		}
		if !ttftSet && cr.Content != "" {
			res.ttft = time.Since(requestStart)
			ttftSet = true
		}
		if b.debug && cr.Content != "" {
			fmt.Fprint(os.Stderr, cr.Content)
		}
		if cr.Done {
			gotDone = true
			res.promptEvalCount = cr.PromptEvalCount
			res.promptEvalDuration = cr.PromptEvalDuration
			res.evalCount = cr.EvalCount
			res.evalDuration = cr.EvalDuration
		}
	}
	if b.debug {
		fmt.Fprintln(os.Stderr)
	}
	if err := scanner.Err(); err != nil {
		return res, err
	}
	if !gotDone {
		return res, errNoMetrics
	}
	res.totalDuration = time.Since(requestStart)
	return res, nil
}

// Cleanup terminates the runner only if bench spawned it; an operator-managed
// runner (reached via -runner) is left running.
func (b *runnerBackend) Cleanup(timeout int) {
	if b.cmd == nil || b.cmd.Process == nil {
		return
	}
	fmt.Fprintf(os.Stderr, "bench: stopping spawned mlx runner (pid %d)\n", b.cmd.Process.Pid)
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

func freePort() (int, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port, nil
}

func resolveOllama(fOpt flagOptions) (string, error) {
	if *fOpt.ollamaBin != "" {
		return *fOpt.ollamaBin, nil
	}
	if p, err := exec.LookPath("ollama"); err == nil {
		return p, nil
	}
	if exe, err := os.Executable(); err == nil {
		cand := filepath.Join(filepath.Dir(exe), "ollama")
		if _, err := os.Stat(cand); err == nil {
			return cand, nil
		}
	}
	return "", fmt.Errorf("could not locate the ollama binary; pass -ollama <path>")
}
