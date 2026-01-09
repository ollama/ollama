// Package usage provides in-memory usage statistics collection and reporting.
package usage

import (
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/version"
)

// Stats collects usage statistics in memory and reports them periodically.
type Stats struct {
	mu sync.RWMutex

	// Atomic counters for hot path
	requestsTotal    atomic.Int64
	tokensPrompt     atomic.Int64
	tokensCompletion atomic.Int64
	errorsTotal      atomic.Int64

	// Map-based counters (require lock)
	endpoints     map[string]int64
	architectures map[string]int64
	apis          map[string]int64
	models        map[string]*ModelStats // per-model stats

	// Feature usage
	toolCalls        atomic.Int64
	structuredOutput atomic.Int64

	// Update info (set by reporter after pinging update endpoint)
	updateAvailable atomic.Value // string

	// Reporter
	stopCh   chan struct{}
	doneCh   chan struct{}
	interval time.Duration
	endpoint string
}

// ModelStats tracks per-model usage statistics.
type ModelStats struct {
	Requests     int64
	TokensInput  int64
	TokensOutput int64
}

// Request contains the data to record for a single request.
type Request struct {
	Endpoint         string // "chat", "generate", "embed"
	Model            string // model name (e.g., "llama3.2:3b")
	Architecture     string // model architecture (e.g., "llama", "qwen2")
	APIType          string // "native" or "openai_compat"
	PromptTokens     int
	CompletionTokens int
	UsedTools        bool
	StructuredOutput bool
}

// SystemInfo contains hardware information to report.
type SystemInfo struct {
	OS        string `json:"os"`
	Arch      string `json:"arch"`
	CPUCores  int    `json:"cpu_cores"`
	RAMBytes  uint64 `json:"ram_bytes"`
	GPUs      []GPU  `json:"gpus,omitempty"`
}

// GPU contains information about a GPU.
type GPU struct {
	Name         string `json:"name"`
	VRAMBytes    uint64 `json:"vram_bytes"`
	ComputeMajor int    `json:"compute_major,omitempty"`
	ComputeMinor int    `json:"compute_minor,omitempty"`
	DriverMajor  int    `json:"driver_major,omitempty"`
	DriverMinor  int    `json:"driver_minor,omitempty"`
}

// Payload is the data sent to the heartbeat endpoint.
type Payload struct {
	Version string     `json:"version"`
	Time    time.Time  `json:"time"`
	System  SystemInfo `json:"system"`

	Totals struct {
		Requests     int64 `json:"requests"`
		Errors       int64 `json:"errors"`
		InputTokens  int64 `json:"input_tokens"`
		OutputTokens int64 `json:"output_tokens"`
	} `json:"totals"`

	Endpoints     map[string]int64 `json:"endpoints"`
	Architectures map[string]int64 `json:"architectures"`
	APIs          map[string]int64 `json:"apis"`

	Features struct {
		ToolCalls        int64 `json:"tool_calls"`
		StructuredOutput int64 `json:"structured_output"`
	} `json:"features"`
}

const (
	defaultInterval = 1 * time.Hour
)

// New creates a new Stats instance.
func New(opts ...Option) *Stats {
	t := &Stats{
		endpoints:     make(map[string]int64),
		architectures: make(map[string]int64),
		apis:          make(map[string]int64),
		models:        make(map[string]*ModelStats),
		stopCh:        make(chan struct{}),
		doneCh:        make(chan struct{}),
		interval:      defaultInterval,
	}

	for _, opt := range opts {
		opt(t)
	}

	return t
}

// Option configures the Stats instance.
type Option func(*Stats)

// WithInterval sets the reporting interval.
func WithInterval(d time.Duration) Option {
	return func(t *Stats) {
		t.interval = d
	}
}

// Record records a request. This is the hot path and should be fast.
func (t *Stats) Record(r *Request) {
	t.requestsTotal.Add(1)
	t.tokensPrompt.Add(int64(r.PromptTokens))
	t.tokensCompletion.Add(int64(r.CompletionTokens))

	if r.UsedTools {
		t.toolCalls.Add(1)
	}
	if r.StructuredOutput {
		t.structuredOutput.Add(1)
	}

	t.mu.Lock()
	t.endpoints[r.Endpoint]++
	t.architectures[r.Architecture]++
	t.apis[r.APIType]++

	// Track per-model stats
	if r.Model != "" {
		if t.models[r.Model] == nil {
			t.models[r.Model] = &ModelStats{}
		}
		t.models[r.Model].Requests++
		t.models[r.Model].TokensInput += int64(r.PromptTokens)
		t.models[r.Model].TokensOutput += int64(r.CompletionTokens)
	}
	t.mu.Unlock()
}

// RecordError records a failed request.
func (t *Stats) RecordError() {
	t.errorsTotal.Add(1)
}

// GetModelStats returns a copy of per-model statistics.
func (t *Stats) GetModelStats() map[string]*ModelStats {
	t.mu.RLock()
	defer t.mu.RUnlock()

	result := make(map[string]*ModelStats, len(t.models))
	for k, v := range t.models {
		result[k] = &ModelStats{
			Requests:     v.Requests,
			TokensInput:  v.TokensInput,
			TokensOutput: v.TokensOutput,
		}
	}
	return result
}

// View returns current stats without resetting counters.
func (t *Stats) View() *Payload {
	t.mu.RLock()
	defer t.mu.RUnlock()

	now := time.Now()

	// Copy maps
	endpoints := make(map[string]int64, len(t.endpoints))
	for k, v := range t.endpoints {
		endpoints[k] = v
	}
	architectures := make(map[string]int64, len(t.architectures))
	for k, v := range t.architectures {
		architectures[k] = v
	}
	apis := make(map[string]int64, len(t.apis))
	for k, v := range t.apis {
		apis[k] = v
	}

	p := &Payload{
		Version:       version.Version,
		Time:          now,
		System:        getSystemInfo(),
		Endpoints:     endpoints,
		Architectures: architectures,
		APIs:          apis,
	}

	p.Totals.Requests = t.requestsTotal.Load()
	p.Totals.Errors = t.errorsTotal.Load()
	p.Totals.InputTokens = t.tokensPrompt.Load()
	p.Totals.OutputTokens = t.tokensCompletion.Load()
	p.Features.ToolCalls = t.toolCalls.Load()
	p.Features.StructuredOutput = t.structuredOutput.Load()

	return p
}

// Snapshot returns current stats and resets counters.
func (t *Stats) Snapshot() *Payload {
	t.mu.Lock()
	defer t.mu.Unlock()

	now := time.Now()
	p := &Payload{
		Version:       version.Version,
		Time:          now,
		System:        getSystemInfo(),
		Endpoints:     t.endpoints,
		Architectures: t.architectures,
		APIs:          t.apis,
	}

	p.Totals.Requests = t.requestsTotal.Swap(0)
	p.Totals.Errors = t.errorsTotal.Swap(0)
	p.Totals.InputTokens = t.tokensPrompt.Swap(0)
	p.Totals.OutputTokens = t.tokensCompletion.Swap(0)
	p.Features.ToolCalls = t.toolCalls.Swap(0)
	p.Features.StructuredOutput = t.structuredOutput.Swap(0)

	// Reset maps
	t.endpoints = make(map[string]int64)
	t.architectures = make(map[string]int64)
	t.apis = make(map[string]int64)

	return p
}

// getSystemInfo collects hardware information.
func getSystemInfo() SystemInfo {
	info := SystemInfo{
		OS:   runtime.GOOS,
		Arch: runtime.GOARCH,
	}

	// Get CPU and memory info
	sysInfo := discover.GetSystemInfo()
	info.CPUCores = sysInfo.ThreadCount
	info.RAMBytes = sysInfo.TotalMemory

	// Get GPU info
	gpus := getGPUInfo()
	info.GPUs = gpus

	return info
}

// GPUInfoFunc is a function that returns GPU information.
// It's set by the server package after GPU discovery.
var GPUInfoFunc func() []GPU

// getGPUInfo collects GPU information.
func getGPUInfo() []GPU {
	if GPUInfoFunc != nil {
		return GPUInfoFunc()
	}
	return nil
}

// Start begins the periodic reporting goroutine.
func (t *Stats) Start() {
	go t.reportLoop()
}

// Stop stops reporting and waits for the final report.
func (t *Stats) Stop() {
	close(t.stopCh)
	<-t.doneCh
}

// reportLoop runs the periodic reporting.
func (t *Stats) reportLoop() {
	defer close(t.doneCh)

	ticker := time.NewTicker(t.interval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			t.report()
		case <-t.stopCh:
			// Send final report before stopping
			t.report()
			return
		}
	}
}

// report sends usage stats and checks for updates.
func (t *Stats) report() {
	payload := t.Snapshot()
	t.sendHeartbeat(payload)
}
