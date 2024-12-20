package server

import (
	"bytes"
	"context"
	"errors"
	"log/slog"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/app/lifecycle"
	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/llm"
)

func TestMain(m *testing.M) {
	os.Setenv("OLLAMA_DEBUG", "1")
	lifecycle.InitLogging()
	os.Exit(m.Run())
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
		sessionDuration: &api.Duration{Duration: 2 * time.Second},
	}
	// Fail to load model first
	s.newServerFn = func(gpus discover.GpuInfoList, model string, ggml *llm.GGML, adapters []string, projectors []string, opts api.Options, numParallel int) (llm.LlamaServer, error) {
		return nil, errors.New("something failed to load model blah")
	}
	gpus := discover.GpuInfoList{}
	s.load(req, ggml, gpus, 0)
	require.Empty(t, req.successCh)
	require.Len(t, req.errCh, 1)
	s.loadedMu.Lock()
	require.Empty(t, s.loaded)
	s.loadedMu.Unlock()
	err := <-req.errCh
	require.Contains(t, err.Error(), "this model may be incompatible")

	server := &mockLlm{estimatedVRAM: 10, estimatedVRAMByGPU: map[string]uint64{}}
	s.newServerFn = func(gpus discover.GpuInfoList, model string, ggml *llm.GGML, adapters []string, projectors []string, opts api.Options, numParallel int) (llm.LlamaServer, error) {
		return server, nil
	}
	s.load(req, ggml, gpus, 0)
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
	server.waitResp = errors.New("wait failure")
	s.load(req, ggml, gpus, 0)
	select {
	case err := <-req.errCh:
		require.Contains(t, err.Error(), "wait failure")
	case resp := <-req.successCh:
		t.Fatalf("unexpected success %v", resp)
	}
	s.loadedMu.Lock()
	runner := s.loaded["dummy_model_path"]
	s.loadedMu.Unlock()
	require.NotNil(t, runner)
	require.Equal(t, uint(0), runner.refCount)
	time.Sleep(1 * time.Millisecond)
	require.Len(t, s.expiredCh, 1)
}

type reqBundle struct {
	ctx     context.Context //nolint:containedctx
	ctxDone func()
	srv     *mockLlm
	req     *LlmRequest
	ggml    *llm.GGML
}

func (scenario *reqBundle) newServer(gpus discover.GpuInfoList, model string, ggml *llm.GGML, adapters []string, projectors []string, opts api.Options, numParallel int) (llm.LlamaServer, error) {
	return scenario.srv, nil
}

func newScenarioRequest(t *testing.T, ctx context.Context, modelName string, estimatedVRAM uint64, duration *api.Duration) *reqBundle {
	b := &reqBundle{}
	b.ctx, b.ctxDone = context.WithCancel(ctx)
	t.Helper()

	f, err := os.CreateTemp(t.TempDir(), modelName)
	require.NoError(t, err)
	defer f.Close()

	require.NoError(t, llm.WriteGGUF(f, llm.KV{
		"general.architecture":          "llama",
		"llama.context_length":          uint32(32),
		"llama.embedding_length":        uint32(4096),
		"llama.block_count":             uint32(1),
		"llama.attention.head_count":    uint32(32),
		"llama.attention.head_count_kv": uint32(32),
		"tokenizer.ggml.tokens":         []string{" "},
		"tokenizer.ggml.scores":         []float32{0},
		"tokenizer.ggml.token_type":     []int32{0},
	}, []llm.Tensor{
		{Name: "blk.0.attn.weight", Kind: uint32(0), Offset: uint64(0), Shape: []uint64{1, 1, 1, 1}, WriterTo: bytes.NewReader(make([]byte, 32))},
		{Name: "output.weight", Kind: uint32(0), Offset: uint64(0), Shape: []uint64{1, 1, 1, 1}, WriterTo: bytes.NewReader(make([]byte, 32))},
	}))
	require.NoError(t, err)

	fname := f.Name()
	model := &Model{Name: modelName, ModelPath: fname}
	b.ggml, err = llm.LoadModel(model.ModelPath, 0)
	require.NoError(t, err)

	if duration == nil {
		duration = &api.Duration{Duration: 5 * time.Millisecond}
	}
	b.req = &LlmRequest{
		ctx:             b.ctx,
		model:           model,
		opts:            api.DefaultOptions(),
		sessionDuration: duration,
		successCh:       make(chan *runnerRef, 1),
		errCh:           make(chan error, 1),
	}
	b.srv = &mockLlm{estimatedVRAM: estimatedVRAM, estimatedVRAMByGPU: map[string]uint64{"": estimatedVRAM}}
	return b
}

func getGpuFn() discover.GpuInfoList {
	g := discover.GpuInfo{Library: "metal"}
	g.TotalMemory = 24 * format.GigaByte
	g.FreeMemory = 12 * format.GigaByte
	return []discover.GpuInfo{g}
}

func getCpuFn() discover.GpuInfoList {
	g := discover.GpuInfo{Library: "cpu"}
	g.TotalMemory = 32 * format.GigaByte
	g.FreeMemory = 26 * format.GigaByte
	return []discover.GpuInfo{g}
}

func TestRequestsSameModelSameRequest(t *testing.T) {
	ctx, done := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer done()
	s := InitScheduler(ctx)
	s.getGpuFn = getGpuFn
	s.getCpuFn = getCpuFn
	a := newScenarioRequest(t, ctx, "ollama-model-1", 10, &api.Duration{Duration: 5 * time.Millisecond})
	b := newScenarioRequest(t, ctx, "ollama-model-1", 11, &api.Duration{Duration: 0})
	b.req.model = a.req.model
	b.ggml = a.ggml

	s.newServerFn = a.newServer
	slog.Info("a")
	s.pendingReqCh <- a.req
	require.Len(t, s.pendingReqCh, 1)
	s.Run(ctx)
	select {
	case resp := <-a.req.successCh:
		require.Equal(t, resp.llama, a.srv)
		require.Empty(t, s.pendingReqCh)
		require.Empty(t, a.req.errCh)
	case err := <-a.req.errCh:
		t.Fatal(err.Error())
	case <-ctx.Done():
		t.Fatal("timeout")
	}

	// Same runner as first request due to not needing a reload
	s.newServerFn = b.newServer
	slog.Info("b")
	s.pendingReqCh <- b.req
	select {
	case resp := <-b.req.successCh:
		require.Equal(t, resp.llama, a.srv)
		require.Empty(t, s.pendingReqCh)
		require.Empty(t, b.req.errCh)
	case err := <-b.req.errCh:
		t.Fatal(err.Error())
	case <-ctx.Done():
		t.Fatal("timeout")
	}
}

func TestRequestsSimpleReloadSameModel(t *testing.T) {
	ctx, done := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer done()
	s := InitScheduler(ctx)
	s.getGpuFn = getGpuFn
	s.getCpuFn = getCpuFn
	a := newScenarioRequest(t, ctx, "ollama-model-1", 10, &api.Duration{Duration: 5 * time.Millisecond})
	b := newScenarioRequest(t, ctx, "ollama-model-1", 20, &api.Duration{Duration: 5 * time.Millisecond})
	tmpModel := *a.req.model
	b.req.model = &tmpModel
	b.ggml = a.ggml

	s.newServerFn = a.newServer
	slog.Info("a")
	s.pendingReqCh <- a.req
	require.Len(t, s.pendingReqCh, 1)
	s.Run(ctx)
	select {
	case resp := <-a.req.successCh:
		require.Equal(t, resp.llama, a.srv)
		require.Empty(t, s.pendingReqCh)
		require.Empty(t, a.req.errCh)
	case err := <-a.req.errCh:
		t.Fatal(err.Error())
	case <-ctx.Done():
		t.Fatal("timeout")
	}

	// Trigger a reload
	s.newServerFn = b.newServer
	b.req.model.AdapterPaths = []string{"new"}
	slog.Info("b")
	s.pendingReqCh <- b.req
	// finish first two requests, so model can reload
	time.Sleep(1 * time.Millisecond)
	a.ctxDone()
	select {
	case resp := <-b.req.successCh:
		require.Equal(t, resp.llama, b.srv)
		require.Empty(t, s.pendingReqCh)
		require.Empty(t, b.req.errCh)
	case err := <-b.req.errCh:
		t.Fatal(err.Error())
	case <-ctx.Done():
		t.Fatal("timeout")
	}
}

func TestRequestsMultipleLoadedModels(t *testing.T) {
	ctx, done := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer done()
	s := InitScheduler(ctx)
	s.getGpuFn = getGpuFn
	s.getCpuFn = getCpuFn

	// Multiple loaded models
	a := newScenarioRequest(t, ctx, "ollama-model-3a", 1*format.GigaByte, nil)
	b := newScenarioRequest(t, ctx, "ollama-model-3b", 24*format.GigaByte, nil)
	c := newScenarioRequest(t, ctx, "ollama-model-4a", 30, nil)
	c.req.opts.NumGPU = 0                                       // CPU load, will be allowed
	d := newScenarioRequest(t, ctx, "ollama-model-3c", 30, nil) // Needs prior unloaded

	t.Setenv("OLLAMA_MAX_LOADED_MODELS", "1")
	s.newServerFn = a.newServer
	slog.Info("a")
	s.pendingReqCh <- a.req
	s.Run(ctx)
	select {
	case resp := <-a.req.successCh:
		require.Equal(t, resp.llama, a.srv)
		require.Empty(t, s.pendingReqCh)
		require.Empty(t, a.req.errCh)
	case err := <-a.req.errCh:
		t.Fatal(err.Error())
	case <-ctx.Done():
		t.Fatal("timeout")
	}
	s.loadedMu.Lock()
	require.Len(t, s.loaded, 1)
	s.loadedMu.Unlock()

	t.Setenv("OLLAMA_MAX_LOADED_MODELS", "0")
	s.newServerFn = b.newServer
	slog.Info("b")
	s.pendingReqCh <- b.req
	select {
	case resp := <-b.req.successCh:
		require.Equal(t, resp.llama, b.srv)
		require.Empty(t, s.pendingReqCh)
		require.Empty(t, b.req.errCh)
	case err := <-b.req.errCh:
		t.Fatal(err.Error())
	case <-ctx.Done():
		t.Fatal("timeout")
	}
	s.loadedMu.Lock()
	require.Len(t, s.loaded, 2)
	s.loadedMu.Unlock()

	// This is a CPU load with NumGPU = 0 so it should load
	s.newServerFn = c.newServer
	slog.Info("c")
	s.pendingReqCh <- c.req
	select {
	case resp := <-c.req.successCh:
		require.Equal(t, resp.llama, c.srv)
		require.Empty(t, s.pendingReqCh)
		require.Empty(t, c.req.errCh)
	case err := <-c.req.errCh:
		t.Fatal(err.Error())
	case <-ctx.Done():
		t.Fatal("timeout")
	}
	s.loadedMu.Lock()
	require.Len(t, s.loaded, 3)
	s.loadedMu.Unlock()

	// Try to load a model that won't fit
	s.newServerFn = d.newServer
	slog.Info("d")
	s.loadedMu.Lock()
	require.Len(t, s.loaded, 3)
	s.loadedMu.Unlock()
	a.ctxDone() // Won't help since this one isn't big enough to make room
	time.Sleep(2 * time.Millisecond)
	s.pendingReqCh <- d.req
	// finish prior request, so new model can load
	time.Sleep(6 * time.Millisecond)
	s.loadedMu.Lock()
	require.Len(t, s.loaded, 2)
	s.loadedMu.Unlock()
	b.ctxDone()
	select {
	case resp := <-d.req.successCh:
		require.Equal(t, resp.llama, d.srv)
		require.Empty(t, s.pendingReqCh)
		require.Empty(t, d.req.errCh)
	case <-ctx.Done():
		t.Fatal("timeout")
	}
	s.loadedMu.Lock()
	require.Len(t, s.loaded, 2)
	s.loadedMu.Unlock()
}

func TestGetRunner(t *testing.T) {
	ctx, done := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer done()

	a := newScenarioRequest(t, ctx, "ollama-model-1a", 10, &api.Duration{Duration: 2 * time.Millisecond})
	b := newScenarioRequest(t, ctx, "ollama-model-1b", 10, &api.Duration{Duration: 2 * time.Millisecond})
	c := newScenarioRequest(t, ctx, "ollama-model-1c", 10, &api.Duration{Duration: 2 * time.Millisecond})
	t.Setenv("OLLAMA_MAX_QUEUE", "1")
	s := InitScheduler(ctx)
	s.getGpuFn = getGpuFn
	s.getCpuFn = getCpuFn
	s.newServerFn = a.newServer
	slog.Info("a")
	successCh1a, errCh1a := s.GetRunner(a.ctx, a.req.model, a.req.opts, a.req.sessionDuration)
	require.Len(t, s.pendingReqCh, 1)
	slog.Info("b")
	successCh1b, errCh1b := s.GetRunner(b.ctx, b.req.model, b.req.opts, b.req.sessionDuration)
	require.Len(t, s.pendingReqCh, 1)
	require.Empty(t, successCh1b)
	require.Len(t, errCh1b, 1)
	err := <-errCh1b
	require.Contains(t, err.Error(), "server busy")
	s.Run(ctx)
	select {
	case resp := <-successCh1a:
		require.Equal(t, resp.llama, a.srv)
		require.Empty(t, s.pendingReqCh)
		require.Empty(t, errCh1a)
	case err := <-errCh1a:
		t.Fatal(err.Error())
	case <-ctx.Done():
		t.Fatal("timeout")
	}
	a.ctxDone() // Set "a" model to idle so it can unload
	s.loadedMu.Lock()
	require.Len(t, s.loaded, 1)
	s.loadedMu.Unlock()

	c.req.model.ModelPath = "bad path"
	slog.Info("c")
	successCh1c, errCh1c := s.GetRunner(c.ctx, c.req.model, c.req.opts, c.req.sessionDuration)
	// Starts in pending channel, then should be quickly processed to return an error
	time.Sleep(50 * time.Millisecond) // Long enough for the "a" model to expire and unload
	require.Empty(t, successCh1c)
	s.loadedMu.Lock()
	require.Empty(t, s.loaded)
	s.loadedMu.Unlock()
	require.Len(t, errCh1c, 1)
	err = <-errCh1c
	require.Contains(t, err.Error(), "bad path")
	b.ctxDone()
}

func TestExpireRunner(t *testing.T) {
	ctx, done := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer done()
	s := InitScheduler(ctx)
	req := &LlmRequest{
		ctx:             ctx,
		model:           &Model{ModelPath: "foo"},
		opts:            api.DefaultOptions(),
		successCh:       make(chan *runnerRef, 1),
		errCh:           make(chan error, 1),
		sessionDuration: &api.Duration{Duration: 2 * time.Minute},
	}

	var ggml *llm.GGML
	gpus := discover.GpuInfoList{}
	server := &mockLlm{estimatedVRAM: 10, estimatedVRAMByGPU: map[string]uint64{}}
	s.newServerFn = func(gpus discover.GpuInfoList, model string, ggml *llm.GGML, adapters []string, projectors []string, opts api.Options, numParallel int) (llm.LlamaServer, error) {
		return server, nil
	}
	s.load(req, ggml, gpus, 0)

	select {
	case err := <-req.errCh:
		if err != nil {
			t.Fatalf("expected no errors when loading, got '%s'", err.Error())
		}
	case resp := <-req.successCh:
		s.loadedMu.Lock()
		if resp.refCount != uint(1) || len(s.loaded) != 1 {
			t.Fatalf("expected a model to be loaded")
		}
		s.loadedMu.Unlock()
	}

	s.expireRunner(&Model{ModelPath: "foo"})

	s.finishedReqCh <- req
	s.processCompleted(ctx)

	s.loadedMu.Lock()
	if len(s.loaded) != 0 {
		t.Fatalf("expected model to be unloaded")
	}
	s.loadedMu.Unlock()
}

// TODO - add one scenario that triggers the bogus finished event with positive ref count
func TestPrematureExpired(t *testing.T) {
	ctx, done := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer done()

	// Same model, same request
	scenario1a := newScenarioRequest(t, ctx, "ollama-model-1a", 10, nil)
	s := InitScheduler(ctx)
	s.getGpuFn = func() discover.GpuInfoList {
		g := discover.GpuInfo{Library: "metal"}
		g.TotalMemory = 24 * format.GigaByte
		g.FreeMemory = 12 * format.GigaByte
		return []discover.GpuInfo{g}
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
	case err := <-errCh1a:
		t.Fatal(err.Error())
	case <-ctx.Done():
		t.Fatal("timeout")
	}
	time.Sleep(scenario1a.req.sessionDuration.Duration)
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
		sessionDuration: &api.Duration{Duration: 2},
	}
	finished := make(chan *LlmRequest)
	llm1 := &mockLlm{estimatedVRAMByGPU: map[string]uint64{}}
	r1 := &runnerRef{llama: llm1, sessionDuration: 1, numParallel: 1}
	req.useLoadedRunner(r1, finished)
	require.Equal(t, uint(1), r1.refCount)
	require.Equal(t, time.Duration(2), r1.sessionDuration)
	select {
	case success := <-req.successCh:
		require.Equal(t, r1, success)
	case err := <-req.errCh:
		t.Fatal(err.Error())
	case <-ctx.Done():
		t.Fatal("timeout")
	}
	done()
	fin := <-finished
	require.Equal(t, req, fin)
}

func TestUpdateFreeSpace(t *testing.T) {
	ctx, done := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer done()
	gpus := discover.GpuInfoList{
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
	llm1 := &mockLlm{estimatedVRAMByGPU: map[string]uint64{"1": 50, "2": 50}}
	llm2 := &mockLlm{estimatedVRAMByGPU: map[string]uint64{"1": 125, "2": 75}}
	r1 := &runnerRef{llama: llm1, gpus: gpus, numParallel: 1}
	r2 := &runnerRef{llama: llm2, gpus: gpus, numParallel: 1}

	s := InitScheduler(ctx)
	s.loadedMu.Lock()
	s.loaded["a"] = r1
	s.loaded["b"] = r2
	s.loadedMu.Unlock()

	s.updateFreeSpace(gpus)
	require.Equal(t, uint64(1000-50-125), gpus[0].FreeMemory)
	require.Equal(t, uint64(2000-50-75), gpus[1].FreeMemory)
}

func TestFilterGPUsWithoutLoadingModels(t *testing.T) {
	ctx, done := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer done()
	gpus := discover.GpuInfoList{
		{
			Library: "cuda",
			ID:      "0",
		},
		{
			Library: "cuda",
			ID:      "1",
		},
	}
	r1 := &runnerRef{gpus: discover.GpuInfoList{gpus[0]}, loading: true}

	s := InitScheduler(ctx)
	s.loadedMu.Lock()
	s.loaded["a"] = r1
	s.loadedMu.Unlock()

	tmp := s.filterGPUsWithoutLoadingModels(gpus)
	require.Len(t, tmp, 1)
	require.Equal(t, "1", tmp[0].ID)

	r1.gpus = discover.GpuInfoList{gpus[1]}
	tmp = s.filterGPUsWithoutLoadingModels(gpus)
	require.Len(t, tmp, 1)
	require.Equal(t, "0", tmp[0].ID)

	r1.gpus = discover.GpuInfoList{}
	tmp = s.filterGPUsWithoutLoadingModels(gpus)
	require.Len(t, tmp, 2)
}

func TestFindRunnerToUnload(t *testing.T) {
	ctx, done := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer done()

	r1 := &runnerRef{refCount: 1, sessionDuration: 1, numParallel: 1}
	r2 := &runnerRef{sessionDuration: 2, numParallel: 1}

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

	llm := &mockLlm{estimatedVRAMByGPU: map[string]uint64{}}
	do := api.DefaultOptions()
	runner := &runnerRef{
		model: &Model{
			AdapterPaths:   []string{"adapter1"},
			ProjectorPaths: []string{"projector1"},
		},
		Options:     &do,
		llama:       llm,
		numParallel: 1,
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
	llm.pingResp = errors.New("foo")
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

	llm1 := &mockLlm{estimatedVRAMByGPU: map[string]uint64{}}
	llm2 := &mockLlm{estimatedVRAMByGPU: map[string]uint64{}}
	s := InitScheduler(ctx)
	s.unloadAllRunners()

	r1 := &runnerRef{llama: llm1, numParallel: 1}
	r2 := &runnerRef{llama: llm2, numParallel: 1}

	s.loadedMu.Lock()
	s.loaded["a"] = r1
	s.loaded["b"] = r2
	s.loadedMu.Unlock()
	s.unloadAllRunners()

	require.True(t, llm1.closeCalled)
	require.True(t, llm2.closeCalled)
}

func TestUnload(t *testing.T) {
	llm1 := &mockLlm{estimatedVRAMByGPU: map[string]uint64{}}
	r1 := &runnerRef{llama: llm1, numParallel: 1}
	r2 := &runnerRef{model: &Model{AdapterPaths: []string{"A"}}, numParallel: 1}
	r1.unload()
	require.True(t, llm1.closeCalled)
	r2.unload()
	require.Nil(t, r2.model)
}

func TestAlreadyCanceled(t *testing.T) {
	ctx, done := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer done()
	dctx, done2 := context.WithCancel(ctx)
	done2()
	scenario1a := newScenarioRequest(t, dctx, "ollama-model-1", 10, &api.Duration{Duration: 0})
	s := InitScheduler(ctx)
	slog.Info("scenario1a")
	s.pendingReqCh <- scenario1a.req
	require.Len(t, s.pendingReqCh, 1)
	s.Run(ctx)
	time.Sleep(5 * time.Millisecond)
	require.Empty(t, s.pendingReqCh)
	require.Empty(t, scenario1a.req.errCh)
	require.Empty(t, scenario1a.req.successCh)
}

func TestHomogeneousGPUs(t *testing.T) {
	ctx, done := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer done()
	s := InitScheduler(ctx)

	s.getGpuFn = func() discover.GpuInfoList {
		// Set memory values to require the model to be spread
		gpus := []discover.GpuInfo{
			{Library: "cuda"},
			{Library: "rocm"},
		}
		gpus[0].TotalMemory = 1 * format.GibiByte
		gpus[0].FreeMemory = 256 * format.MebiByte
		gpus[1].TotalMemory = 1 * format.GibiByte
		gpus[1].FreeMemory = 256 * format.MebiByte
		return gpus
	}
	s.getCpuFn = getCpuFn
	a := newScenarioRequest(t, ctx, "ollama-model-1", 10, &api.Duration{Duration: 5 * time.Millisecond})
	s.newServerFn = func(gpus discover.GpuInfoList, model string, ggml *llm.GGML, adapters []string, projectors []string, opts api.Options, numParallel int) (llm.LlamaServer, error) {
		require.Len(t, gpus, 1)
		return a.newServer(gpus, model, ggml, adapters, projectors, opts, numParallel)
	}
	slog.Info("a")
	s.pendingReqCh <- a.req
	require.Len(t, s.pendingReqCh, 1)
	s.Run(ctx)
	select {
	case resp := <-a.req.successCh:
		require.Equal(t, resp.llama, a.srv)
		require.Empty(t, s.pendingReqCh)
		require.Empty(t, a.req.errCh)
	case err := <-a.req.errCh:
		t.Fatal(err.Error())
	case <-ctx.Done():
		t.Fatal("timeout")
	}
}

type mockLlm struct {
	pingResp           error
	waitResp           error
	completionResp     error
	embeddingResp      []float32
	embeddingRespErr   error
	tokenizeResp       []int
	tokenizeRespErr    error
	detokenizeResp     string
	detonekizeRespErr  error
	closeResp          error
	closeCalled        bool
	estimatedVRAM      uint64
	estimatedTotal     uint64
	estimatedVRAMByGPU map[string]uint64
	numGPU             int
	maxGPU             int
	numCtx             int
	numParallel        int
}

func (s *mockLlm) Ping(ctx context.Context) error             { return s.pingResp }
func (s *mockLlm) WaitUntilRunning(ctx context.Context) error { return s.waitResp }
func (s *mockLlm) Completion(ctx context.Context, req llm.CompletionRequest, fn func(llm.CompletionResponse)) error {
	return s.completionResp
}

func (s *mockLlm) Embedding(ctx context.Context, input string) ([]float32, error) {
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
func (s *mockLlm) EstimatedVRAM() uint64                  { return s.estimatedVRAM }
func (s *mockLlm) EstimatedTotal() uint64                 { return s.estimatedTotal }
func (s *mockLlm) EstimatedVRAMByGPU(gpuid string) uint64 { return s.estimatedVRAMByGPU[gpuid] }
func (s *mockLlm) NumGPU() int                            { return s.numGPU }
func (s *mockLlm) MaxGPU() int                            { return s.maxGPU }
func (s *mockLlm) NumCtx() int                            { return s.numCtx }
func (s *mockLlm) NumParallel() int                       { return s.numParallel }
