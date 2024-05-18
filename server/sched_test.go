package server

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"log/slog"
	"os"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/app/lifecycle"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/gpu"
	"github.com/ollama/ollama/llm"
	"github.com/stretchr/testify/require"
)

func init() {
	os.Setenv("OLLAMA_DEBUG", "1")
	lifecycle.InitLogging()
}

func TestInitScheduler(t *testing.T) {
	ctx, done := context.WithCancel(context.Background())
	defer done()
	s := InitScheduler(ctx)
	s.loadedMu.Lock()
	require.NotNil(t, s.loaded)
	s.loadedMu.Unlock()
}

func TestLoad(t *testing.T) {
	ctx, done := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer done()
	s := InitScheduler(ctx)
	var ggml *llm.GGML // value not used in tests
	req := &LlmRequest{
		ctx:             ctx,
		model:           &Model{ModelPath: "foo"},
		opts:            api.DefaultOptions(),
		successCh:       make(chan *runnerRef, 1),
		errCh:           make(chan error, 1),
		sessionDuration: 2,
	}
	// Fail to load model first
	s.newServerFn = func(gpus gpu.GpuInfoList, model string, ggml *llm.GGML, adapters []string, projectors []string, opts api.Options) (llm.LlamaServer, error) {
		return nil, fmt.Errorf("something failed to load model blah")
	}
	gpus := gpu.GpuInfoList{}
	s.load(req, ggml, gpus)
	require.Empty(t, req.successCh)
	require.Len(t, req.errCh, 1)
	s.loadedMu.Lock()
	require.Empty(t, s.loaded)
	s.loadedMu.Unlock()
	err := <-req.errCh
	require.Contains(t, err.Error(), "this model may be incompatible")

	server := &mockLlm{estimatedVRAM: 10}
	s.newServerFn = func(gpus gpu.GpuInfoList, model string, ggml *llm.GGML, adapters []string, projectors []string, opts api.Options) (llm.LlamaServer, error) {
		return server, nil
	}
	s.load(req, ggml, gpus)
	select {
	case err := <-req.errCh:
		require.NoError(t, err)
	case resp := <-req.successCh:
		require.Equal(t, uint64(10), resp.estimatedVRAM)
		require.Equal(t, uint(1), resp.refCount)
		s.loadedMu.Lock()
		require.Len(t, s.loaded, 1)
		s.loadedMu.Unlock()
	}

	req.model.ModelPath = "dummy_model_path"
	server.waitResp = fmt.Errorf("wait failure")
	s.load(req, ggml, gpus)
	select {
	case err := <-req.errCh:
		require.Contains(t, err.Error(), "wait failure")
	case resp := <-req.successCh:
		t.Errorf("unexpected success %v", resp)
	}
	s.loadedMu.Lock()
	runner := s.loaded["dummy_model_path"]
	s.loadedMu.Unlock()
	require.NotNil(t, runner)
	require.Equal(t, uint(0), runner.refCount)
	time.Sleep(1 * time.Millisecond)
	require.Len(t, s.expiredCh, 1)
}

type bundle struct {
	ctx     context.Context //nolint:containedctx
	ctxDone func()
	srv     *mockLlm
	req     *LlmRequest
	ggml    *llm.GGML
}

func (scenario *bundle) newServer(gpus gpu.GpuInfoList, model string, ggml *llm.GGML, adapters []string, projectors []string, opts api.Options) (llm.LlamaServer, error) {
	return scenario.srv, nil
}

func newScenario(t *testing.T, ctx context.Context, modelName string, estimatedVRAM uint64) *bundle {
	scenario := &bundle{}
	scenario.ctx, scenario.ctxDone = context.WithCancel(ctx)
	t.Helper()

	f, err := os.CreateTemp(t.TempDir(), modelName)
	require.NoError(t, err)
	defer f.Close()

	gguf := llm.NewGGUFV3(binary.LittleEndian)
	err = gguf.Encode(f, llm.KV{
		"general.architecture":          "llama",
		"general.name":                  "name",
		"llama.context_length":          uint32(32),
		"llama.embedding_length":        uint32(4096),
		"llama.block_count":             uint32(1),
		"llama.attention.head_count":    uint32(32),
		"llama.attention.head_count_kv": uint32(32),
		"tokenizer.ggml.tokens":         []string{" "},
		"tokenizer.ggml.scores":         []float32{0},
		"tokenizer.ggml.token_type":     []int32{0},
	}, []llm.Tensor{
		{Name: "blk.0.attn.weight", Kind: uint32(0), Offset: uint64(0), Shape: []uint64{1, 1, 1, 1}, WriterTo: &bytes.Reader{}},
		{Name: "output.weight", Kind: uint32(0), Offset: uint64(0), Shape: []uint64{1, 1, 1, 1}, WriterTo: &bytes.Reader{}},
	})
	require.NoError(t, err)

	fname := f.Name()
	model := &Model{Name: modelName, ModelPath: fname}
	scenario.ggml, err = llm.LoadModel(model.ModelPath)
	require.NoError(t, err)

	scenario.req = &LlmRequest{
		ctx:             scenario.ctx,
		model:           model,
		opts:            api.DefaultOptions(),
		sessionDuration: 5 * time.Millisecond,
		successCh:       make(chan *runnerRef, 1),
		errCh:           make(chan error, 1),
	}
	scenario.srv = &mockLlm{estimatedVRAM: estimatedVRAM}
	return scenario
}

func TestRequests(t *testing.T) {
	ctx, done := context.WithTimeout(context.Background(), time.Second)
	defer done()

	// Same model, same request
	scenario1a := newScenario(t, ctx, "ollama-model-1", 10)
	scenario1a.req.sessionDuration = 0
	scenario1b := newScenario(t, ctx, "ollama-model-1", 11)
	scenario1b.req.model = scenario1a.req.model
	scenario1b.ggml = scenario1a.ggml
	scenario1b.req.sessionDuration = 0

	// simple reload of same model
	scenario2a := newScenario(t, ctx, "ollama-model-1", 20)
	tmpModel := *scenario1a.req.model
	scenario2a.req.model = &tmpModel
	scenario2a.ggml = scenario1a.ggml

	// Multiple loaded models
	scenario3a := newScenario(t, ctx, "ollama-model-3a", 1*format.GigaByte)
	scenario3b := newScenario(t, ctx, "ollama-model-3b", 24*format.GigaByte)
	scenario3c := newScenario(t, ctx, "ollama-model-4a", 30)
	scenario3c.req.opts.NumGPU = 0                           // CPU load, will be allowed
	scenario3d := newScenario(t, ctx, "ollama-model-3c", 30) // Needs prior unloaded

	s := InitScheduler(ctx)
	s.getGpuFn = func() gpu.GpuInfoList {
		g := gpu.GpuInfo{Library: "metal"}
		g.TotalMemory = 24 * format.GigaByte
		g.FreeMemory = 12 * format.GigaByte
		return []gpu.GpuInfo{g}
	}
	s.newServerFn = scenario1a.newServer
	slog.Info("scenario1a")
	s.pendingReqCh <- scenario1a.req
	require.Len(t, s.pendingReqCh, 1)
	s.Run(ctx)
	select {
	case resp := <-scenario1a.req.successCh:
		require.Equal(t, resp.llama, scenario1a.srv)
		require.Empty(t, s.pendingReqCh)
		require.Empty(t, scenario1a.req.errCh)
	case <-ctx.Done():
		t.Errorf("timeout")
	}

	// Same runner as first request due to not needing a reload
	s.newServerFn = scenario1b.newServer
	slog.Info("scenario1b")
	s.pendingReqCh <- scenario1b.req
	select {
	case resp := <-scenario1b.req.successCh:
		require.Equal(t, resp.llama, scenario1a.srv)
		require.Empty(t, s.pendingReqCh)
		require.Empty(t, scenario1b.req.errCh)
	case <-ctx.Done():
		t.Errorf("timeout")
	}

	// Trigger a reload
	s.newServerFn = scenario2a.newServer
	scenario2a.req.model.AdapterPaths = []string{"new"}
	slog.Info("scenario2a")
	s.pendingReqCh <- scenario2a.req
	// finish first two requests, so model can reload
	time.Sleep(1 * time.Millisecond)
	scenario1a.ctxDone()
	scenario1b.ctxDone()
	select {
	case resp := <-scenario2a.req.successCh:
		require.Equal(t, resp.llama, scenario2a.srv)
		require.Empty(t, s.pendingReqCh)
		require.Empty(t, scenario2a.req.errCh)
	case <-ctx.Done():
		t.Errorf("timeout")
	}

	envconfig.MaxRunners = 1
	s.newServerFn = scenario3a.newServer
	slog.Info("scenario3a")
	s.pendingReqCh <- scenario3a.req
	// finish prior request, so new model can load
	time.Sleep(1 * time.Millisecond)
	scenario2a.ctxDone()
	select {
	case resp := <-scenario3a.req.successCh:
		require.Equal(t, resp.llama, scenario3a.srv)
		require.Empty(t, s.pendingReqCh)
		require.Empty(t, scenario3a.req.errCh)
	case <-ctx.Done():
		t.Errorf("timeout")
	}
	s.loadedMu.Lock()
	require.Len(t, s.loaded, 1)
	s.loadedMu.Unlock()

	envconfig.MaxRunners = 0
	s.newServerFn = scenario3b.newServer
	slog.Info("scenario3b")
	s.pendingReqCh <- scenario3b.req
	select {
	case resp := <-scenario3b.req.successCh:
		require.Equal(t, resp.llama, scenario3b.srv)
		require.Empty(t, s.pendingReqCh)
		require.Empty(t, scenario3b.req.errCh)
	case <-ctx.Done():
		t.Errorf("timeout")
	}
	s.loadedMu.Lock()
	require.Len(t, s.loaded, 2)
	s.loadedMu.Unlock()

	// This is a CPU load with NumGPU = 0 so it should load
	s.newServerFn = scenario3c.newServer
	slog.Info("scenario3c")
	s.pendingReqCh <- scenario3c.req
	select {
	case resp := <-scenario3c.req.successCh:
		require.Equal(t, resp.llama, scenario3c.srv)
		require.Empty(t, s.pendingReqCh)
		require.Empty(t, scenario3c.req.errCh)
	case <-ctx.Done():
		t.Errorf("timeout")
	}
	s.loadedMu.Lock()
	require.Len(t, s.loaded, 3)
	s.loadedMu.Unlock()

	// Try to load a model that wont fit
	s.newServerFn = scenario3d.newServer
	slog.Info("scenario3d")
	s.loadedMu.Lock()
	require.Len(t, s.loaded, 3)
	s.loadedMu.Unlock()
	scenario3a.ctxDone() // Won't help since this one isn't big enough to make room
	time.Sleep(2 * time.Millisecond)
	s.pendingReqCh <- scenario3d.req
	// finish prior request, so new model can load
	time.Sleep(6 * time.Millisecond)
	s.loadedMu.Lock()
	require.Len(t, s.loaded, 2)
	s.loadedMu.Unlock()
	scenario3b.ctxDone()
	select {
	case resp := <-scenario3d.req.successCh:
		require.Equal(t, resp.llama, scenario3d.srv)
		require.Empty(t, s.pendingReqCh)
		require.Empty(t, scenario3d.req.errCh)
	case <-ctx.Done():
		t.Errorf("timeout")
	}
	s.loadedMu.Lock()
	require.Len(t, s.loaded, 2)
	s.loadedMu.Unlock()
}

func TestGetRunner(t *testing.T) {
	ctx, done := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer done()

	// Same model, same request
	scenario1a := newScenario(t, ctx, "ollama-model-1a", 10)
	scenario1a.req.sessionDuration = 0
	scenario1b := newScenario(t, ctx, "ollama-model-1b", 10)
	scenario1b.req.sessionDuration = 0
	scenario1c := newScenario(t, ctx, "ollama-model-1c", 10)
	scenario1c.req.sessionDuration = 0
	envconfig.MaxQueuedRequests = 1
	s := InitScheduler(ctx)
	s.getGpuFn = func() gpu.GpuInfoList {
		g := gpu.GpuInfo{Library: "metal"}
		g.TotalMemory = 24 * format.GigaByte
		g.FreeMemory = 12 * format.GigaByte
		return []gpu.GpuInfo{g}
	}
	s.newServerFn = scenario1a.newServer
	slog.Info("scenario1a")
	successCh1a, errCh1a := s.GetRunner(scenario1a.ctx, scenario1a.req.model, scenario1a.req.opts, scenario1a.req.sessionDuration)
	require.Len(t, s.pendingReqCh, 1)
	slog.Info("scenario1b")
	successCh1b, errCh1b := s.GetRunner(scenario1b.ctx, scenario1b.req.model, scenario1b.req.opts, scenario1b.req.sessionDuration)
	require.Len(t, s.pendingReqCh, 1)
	require.Empty(t, successCh1b)
	require.Len(t, errCh1b, 1)
	err := <-errCh1b
	require.Contains(t, err.Error(), "server busy")
	s.Run(ctx)
	select {
	case resp := <-successCh1a:
		require.Equal(t, resp.llama, scenario1a.srv)
		require.Empty(t, s.pendingReqCh)
		require.Empty(t, errCh1a)
	case <-ctx.Done():
		t.Errorf("timeout")
	}
	scenario1a.ctxDone()
	s.loadedMu.Lock()
	require.Len(t, s.loaded, 1)
	s.loadedMu.Unlock()

	scenario1c.req.model.ModelPath = "bad path"
	slog.Info("scenario1c")
	successCh1c, errCh1c := s.GetRunner(scenario1c.ctx, scenario1c.req.model, scenario1c.req.opts, scenario1c.req.sessionDuration)
	// Starts in pending channel, then should be quickly processsed to return an error
	time.Sleep(5 * time.Millisecond)
	require.Empty(t, successCh1c)
	s.loadedMu.Lock()
	require.Empty(t, s.loaded)
	s.loadedMu.Unlock()
	require.Len(t, errCh1c, 1)
	err = <-errCh1c
	require.Contains(t, err.Error(), "bad path")
	scenario1b.ctxDone()
}

// TODO - add one scenario that triggers the bogus finished event with positive ref count
func TestPrematureExpired(t *testing.T) {
	ctx, done := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer done()

	// Same model, same request
	scenario1a := newScenario(t, ctx, "ollama-model-1a", 10)
	s := InitScheduler(ctx)
	s.getGpuFn = func() gpu.GpuInfoList {
		g := gpu.GpuInfo{Library: "metal"}
		g.TotalMemory = 24 * format.GigaByte
		g.FreeMemory = 12 * format.GigaByte
		return []gpu.GpuInfo{g}
	}
	s.newServerFn = scenario1a.newServer
	successCh1a, errCh1a := s.GetRunner(scenario1a.ctx, scenario1a.req.model, scenario1a.req.opts, scenario1a.req.sessionDuration)
	require.Len(t, s.pendingReqCh, 1)
	s.Run(ctx)
	select {
	case resp := <-successCh1a:
		require.Equal(t, resp.llama, scenario1a.srv)
		require.Empty(t, s.pendingReqCh)
		require.Empty(t, errCh1a)
		s.loadedMu.Lock()
		require.Len(t, s.loaded, 1)
		s.loadedMu.Unlock()
		slog.Info("sending premature expired event now")
		s.expiredCh <- resp // Shouldn't happen in real life, but make sure its safe
	case <-ctx.Done():
		t.Errorf("timeout")
	}
	time.Sleep(scenario1a.req.sessionDuration)
	scenario1a.ctxDone()
	time.Sleep(20 * time.Millisecond)
	require.LessOrEqual(t, len(s.finishedReqCh), 1)
	time.Sleep(10 * time.Millisecond)
	require.Empty(t, s.finishedReqCh)
	s.loadedMu.Lock()
	require.Empty(t, s.loaded)
	s.loadedMu.Unlock()

	// also shouldn't happen in real life
	s.finishedReqCh <- scenario1a.req
	time.Sleep(5 * time.Millisecond)
}

func TestUseLoadedRunner(t *testing.T) {
	ctx, done := context.WithTimeout(context.Background(), 100*time.Millisecond)
	req := &LlmRequest{
		ctx:             ctx,
		opts:            api.DefaultOptions(),
		successCh:       make(chan *runnerRef, 1),
		sessionDuration: 2,
	}
	finished := make(chan *LlmRequest)
	llm1 := &mockLlm{}
	r1 := &runnerRef{llama: llm1, sessionDuration: 1}
	req.useLoadedRunner(r1, finished)
	require.Equal(t, uint(1), r1.refCount)
	require.Equal(t, time.Duration(2), r1.sessionDuration)
	select {
	case success := <-req.successCh:
		require.Equal(t, r1, success)
	case <-ctx.Done():
		t.Errorf("timeout")
	}
	done()
	fin := <-finished
	require.Equal(t, req, fin)
}

func TestUpdateFreeSpace(t *testing.T) {
	ctx, done := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer done()
	gpus := gpu.GpuInfoList{
		{
			Library: "a",
			ID:      "1",
		},
		{
			Library: "a",
			ID:      "2",
		},
	}
	gpus[0].TotalMemory = 1000
	gpus[0].FreeMemory = 900
	gpus[1].TotalMemory = 2000
	gpus[1].FreeMemory = 1900
	llm1 := &mockLlm{estimatedVRAM: 100}
	llm2 := &mockLlm{estimatedVRAM: 200}
	r1 := &runnerRef{llama: llm1, gpus: gpus}
	r2 := &runnerRef{llama: llm2, gpus: gpus}

	s := InitScheduler(ctx)
	s.loadedMu.Lock()
	s.loaded["a"] = r1
	s.loaded["b"] = r2
	s.loadedMu.Unlock()

	s.updateFreeSpace(gpus)
	require.Equal(t, uint64(850), gpus[0].FreeMemory)
	require.Equal(t, uint64(1850), gpus[1].FreeMemory)
}

func TestFindRunnerToUnload(t *testing.T) {
	ctx, done := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer done()

	r1 := &runnerRef{refCount: 1, sessionDuration: 1}
	r2 := &runnerRef{sessionDuration: 2}

	s := InitScheduler(ctx)
	s.loadedMu.Lock()
	s.loaded["a"] = r1
	s.loaded["b"] = r2
	s.loadedMu.Unlock()

	resp := s.findRunnerToUnload()
	require.Equal(t, r2, resp)
	r2.refCount = 1
	resp = s.findRunnerToUnload()
	require.Equal(t, r1, resp)
}

func TestNeedsReload(t *testing.T) {
	ctx, done := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer done()

	llm := &mockLlm{}
	do := api.DefaultOptions()
	runner := &runnerRef{
		model:   &Model{AdapterPaths: []string{"adapter1"}, ProjectorPaths: []string{"projector1"}},
		Options: &do,
		llama:   llm,
	}
	req := &LlmRequest{
		model: &Model{
			AdapterPaths:   []string{"adapter2"},
			ProjectorPaths: []string{"projector2"},
		},
		opts: api.DefaultOptions(),
	}
	resp := runner.needsReload(ctx, req)
	require.True(t, resp)
	req.model.AdapterPaths = runner.model.AdapterPaths
	resp = runner.needsReload(ctx, req)
	require.True(t, resp)
	req.model.ProjectorPaths = runner.model.ProjectorPaths
	runner.loading = true
	req.opts.NumBatch = 1234
	resp = runner.needsReload(ctx, req)
	require.True(t, resp)
	req.opts.NumBatch = runner.Options.NumBatch
	llm.pingResp = fmt.Errorf("foo")
	resp = runner.needsReload(ctx, req)
	require.True(t, resp)
	llm.pingResp = nil
	resp = runner.needsReload(ctx, req)
	require.False(t, resp)
	req.opts.NumGPU = 99
	resp = runner.needsReload(ctx, req)
	require.True(t, resp)
	req.opts.NumGPU = -1
	resp = runner.needsReload(ctx, req)
	require.False(t, resp)
}

func TestUnloadAllRunners(t *testing.T) {
	ctx, done := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer done()

	llm1 := &mockLlm{}
	llm2 := &mockLlm{}
	s := InitScheduler(ctx)
	s.unloadAllRunners()

	r1 := &runnerRef{llama: llm1}
	r2 := &runnerRef{llama: llm2}

	s.loadedMu.Lock()
	s.loaded["a"] = r1
	s.loaded["b"] = r2
	s.loadedMu.Unlock()
	s.unloadAllRunners()

	require.True(t, llm1.closeCalled)
	require.True(t, llm2.closeCalled)
}

func TestUnload(t *testing.T) {
	llm1 := &mockLlm{}
	r1 := &runnerRef{llama: llm1}
	r2 := &runnerRef{model: &Model{AdapterPaths: []string{"A"}}}
	r1.unload()
	require.True(t, llm1.closeCalled)
	r2.unload()
	require.Nil(t, r2.model)
}

type mockLlm struct {
	pingResp          error
	waitResp          error
	completionResp    error
	embeddingResp     []float64
	embeddingRespErr  error
	tokenizeResp      []int
	tokenizeRespErr   error
	detokenizeResp    string
	detonekizeRespErr error
	closeResp         error
	closeCalled       bool
	estimatedVRAM     uint64
	estimatedTotal    uint64
}

func (s *mockLlm) Ping(ctx context.Context) error             { return s.pingResp }
func (s *mockLlm) WaitUntilRunning(ctx context.Context) error { return s.waitResp }
func (s *mockLlm) Completion(ctx context.Context, req llm.CompletionRequest, fn func(llm.CompletionResponse)) error {
	return s.completionResp
}
func (s *mockLlm) Embedding(ctx context.Context, prompt string) ([]float64, error) {
	return s.embeddingResp, s.embeddingRespErr
}
func (s *mockLlm) Tokenize(ctx context.Context, content string) ([]int, error) {
	return s.tokenizeResp, s.tokenizeRespErr
}
func (s *mockLlm) Detokenize(ctx context.Context, tokens []int) (string, error) {
	return s.detokenizeResp, s.detonekizeRespErr
}
func (s *mockLlm) Close() error {
	s.closeCalled = true
	return s.closeResp
}
func (s *mockLlm) EstimatedVRAM() uint64  { return s.estimatedVRAM }
func (s *mockLlm) EstimatedTotal() uint64 { return s.estimatedTotal }
