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
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
)

func TestMain(m *testing.M) {
	os.Setenv("OLLAMA_DEBUG", "1")
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelDebug}))
	slog.SetDefault(logger)
	os.Exit(m.Run())
}

func TestSchedInit(t *testing.T) {
	ctx, done := context.WithCancel(t.Context())
	defer done()
	s := InitScheduler(ctx)
	s.loadedMu.Lock()
	require.NotNil(t, s.loaded)
	s.loadedMu.Unlock()
}

func TestSchedLoad(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 20*time.Millisecond)
	defer done()
	s := InitScheduler(ctx)
	s.waitForRecovery = 10 * time.Millisecond
	var f *ggml.GGML // value not used in tests
	req := &LlmRequest{
		ctx:             ctx,
		model:           &Model{ModelPath: "foo"},
		opts:            api.DefaultOptions(),
		successCh:       make(chan *runnerRef, 1),
		errCh:           make(chan error, 1),
		sessionDuration: &api.Duration{Duration: 2 * time.Second},
	}
	// Fail to load model first
	s.newServerFn = func(systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, model string, f *ggml.GGML, adapters []string, projectors []string, opts api.Options, numParallel int) (llm.LlamaServer, error) {
		return nil, errors.New("something failed to load model blah")
	}
	gpus := []ml.DeviceInfo{}
	systemInfo := ml.SystemInfo{}
	s.load(req, f, systemInfo, gpus, false)
	require.Empty(t, req.successCh)
	require.Len(t, req.errCh, 1)
	s.loadedMu.Lock()
	require.Empty(t, s.loaded)
	s.loadedMu.Unlock()
	err := <-req.errCh
	require.Contains(t, err.Error(), "this model may be incompatible")

	server := &mockLlm{vramSize: 10, vramByGPU: map[ml.DeviceID]uint64{}}
	s.newServerFn = func(systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, model string, f *ggml.GGML, adapters []string, projectors []string, opts api.Options, numParallel int) (llm.LlamaServer, error) {
		server.modelPath = model
		return server, nil
	}
	s.load(req, f, systemInfo, gpus, false)
	select {
	case err := <-req.errCh:
		require.NoError(t, err)
	case resp := <-req.successCh:
		require.Equal(t, uint64(10), resp.vramSize)
		require.Equal(t, uint(1), resp.refCount)
		s.loadedMu.Lock()
		require.Len(t, s.loaded, 1)
		s.loadedMu.Unlock()
	}

	req.model.ModelPath = "dummy_model_path"
	server.waitResp = errors.New("wait failure")
	s.load(req, f, systemInfo, gpus, false)
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
	f       *ggml.GGML
}

func (scenario *reqBundle) newServer(systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, model string, f *ggml.GGML, adapters []string, projectors []string, opts api.Options, numParallel int) (llm.LlamaServer, error) {
	scenario.srv.modelPath = model
	return scenario.srv, nil
}

func newScenarioRequest(t *testing.T, ctx context.Context, modelName string, vramSize uint64, duration *api.Duration, vramByGPU map[ml.DeviceID]uint64) *reqBundle {
	b := &reqBundle{}
	b.ctx, b.ctxDone = context.WithCancel(ctx)
	t.Helper()

	p, _ := createBinFile(t, ggml.KV{
		"general.architecture":          "llama",
		"llama.context_length":          uint32(32),
		"llama.embedding_length":        uint32(4096),
		"llama.block_count":             uint32(1),
		"llama.attention.head_count":    uint32(32),
		"llama.attention.head_count_kv": uint32(32),
		"tokenizer.ggml.tokens":         []string{" "},
		"tokenizer.ggml.scores":         []float32{0},
		"tokenizer.ggml.token_type":     []int32{0},
	}, []*ggml.Tensor{
		{Name: "blk.0.attn.weight", Kind: uint32(0), Offset: uint64(0), Shape: []uint64{1, 1, 1, 1}, WriterTo: bytes.NewReader(make([]byte, 32))},
		{Name: "output.weight", Kind: uint32(0), Offset: uint64(0), Shape: []uint64{1, 1, 1, 1}, WriterTo: bytes.NewReader(make([]byte, 32))},
	})

	model := &Model{Name: modelName, ModelPath: p}
	f, err := llm.LoadModel(model.ModelPath, 0)
	if err != nil {
		t.Fatal(err)
	}
	b.f = f
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
	b.srv = &mockLlm{vramSize: vramSize, vramByGPU: vramByGPU}
	return b
}

func getGpuFn(ctx context.Context, runners []ml.FilteredRunnerDiscovery) []ml.DeviceInfo {
	slog.Info("test getGpuFn called", "runners", runners)
	g := ml.DeviceInfo{DeviceID: ml.DeviceID{Library: "Metal"}}
	g.TotalMemory = 24 * format.GigaByte
	g.FreeMemory = 12 * format.GigaByte
	return []ml.DeviceInfo{g}
}

func getSystemInfoFn() ml.SystemInfo {
	slog.Info("test getSystemInfoFn called")
	return ml.SystemInfo{
		TotalMemory: 32 * format.GigaByte,
		FreeMemory:  26 * format.GigaByte,
	}
}

func TestSchedRequestsSameModelSameRequest(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 500*time.Millisecond)
	defer done()
	s := InitScheduler(ctx)
	s.waitForRecovery = 10 * time.Millisecond
	s.getGpuFn = getGpuFn
	s.getSystemInfoFn = getSystemInfoFn
	a := newScenarioRequest(t, ctx, "ollama-model-1", 10, &api.Duration{Duration: 5 * time.Millisecond}, nil)
	b := newScenarioRequest(t, ctx, "ollama-model-1", 11, &api.Duration{Duration: 0}, nil)
	b.req.model = a.req.model
	b.f = a.f

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

func TestSchedRequestsSimpleReloadSameModel(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 5000*time.Millisecond)
	defer done()
	s := InitScheduler(ctx)
	s.waitForRecovery = 10 * time.Millisecond
	s.getGpuFn = getGpuFn
	s.getSystemInfoFn = getSystemInfoFn
	a := newScenarioRequest(t, ctx, "ollama-model-1", 10, &api.Duration{Duration: 5 * time.Millisecond}, nil)
	b := newScenarioRequest(t, ctx, "ollama-model-1", 20, &api.Duration{Duration: 5 * time.Millisecond}, nil)
	tmpModel := *a.req.model
	b.req.model = &tmpModel
	b.f = a.f

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
	// Report recovered VRAM usage
	time.Sleep(1 * time.Millisecond)
	s.getGpuFn = func(ctx context.Context, runners []ml.FilteredRunnerDiscovery) []ml.DeviceInfo {
		slog.Info("altered getGpuFn called")
		g := ml.DeviceInfo{DeviceID: ml.DeviceID{Library: "Metal"}}
		g.TotalMemory = 24 * format.GigaByte
		g.FreeMemory = 24 * format.GigaByte
		return []ml.DeviceInfo{g}
	}
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

func TestSchedRequestsMultipleLoadedModels(t *testing.T) {
	slog.Info("TestRequestsMultipleLoadedModels")
	ctx, done := context.WithTimeout(t.Context(), 1000*time.Millisecond)
	defer done()
	s := InitScheduler(ctx)
	s.waitForRecovery = 10 * time.Millisecond
	s.getGpuFn = getGpuFn // 1 Metal GPU
	s.getSystemInfoFn = getSystemInfoFn

	// Multiple loaded models
	a := newScenarioRequest(t, ctx, "model-a-1g-gpu", 1*format.GigaByte, nil, map[ml.DeviceID]uint64{{Library: "Metal"}: 1 * format.GigaByte})
	a.req.sessionDuration = &api.Duration{Duration: 5 * time.Millisecond}
	b := newScenarioRequest(t, ctx, "model-b-10g-gpu", 10*format.GigaByte, nil, map[ml.DeviceID]uint64{{Library: "Metal"}: 10 * format.GigaByte})
	b.req.sessionDuration = &api.Duration{Duration: 5 * time.Millisecond}
	c := newScenarioRequest(t, ctx, "model-c-10g-cpu", 10*format.GigaByte, nil, nil /* No GPU load */)
	c.req.opts.NumGPU = 0                                                                                                                         // CPU load, will be allowed
	b.req.sessionDuration = &api.Duration{Duration: 10 * time.Millisecond}                                                                        // longer than b to cause the scheduler to favor unloading b over c
	d := newScenarioRequest(t, ctx, "model-d-10g-gpu", 13*format.GigaByte, nil, map[ml.DeviceID]uint64{{Library: "Metal"}: 13 * format.GigaByte}) // Needs prior unloaded

	s.newServerFn = a.newServer
	slog.Info("Loading A")
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
	slog.Info("Loading B")
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
	slog.Info("Loading C")
	s.pendingReqCh <- c.req
	select {
	case resp := <-c.req.successCh:
		require.Equal(t, resp.llama, c.srv)
		require.Empty(t, s.pendingReqCh)
		require.Empty(t, c.req.errCh)
	case err := <-c.req.errCh:
		t.Fatal(err.Error())
	case <-ctx.Done():
		slog.Info("FAIL: scheduler state", "s.loaded", s.loaded)
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
	// Mark b done so it can unload
	b.ctxDone()
	// Report recovered VRAM usage so scheduler will finish waiting and unload
	time.Sleep(1 * time.Millisecond)
	s.getGpuFn = func(ctx context.Context, runners []ml.FilteredRunnerDiscovery) []ml.DeviceInfo {
		g := ml.DeviceInfo{DeviceID: ml.DeviceID{Library: "Metal"}}
		g.TotalMemory = 24 * format.GigaByte
		g.FreeMemory = 24 * format.GigaByte
		return []ml.DeviceInfo{g}
	}
	select {
	case resp := <-d.req.successCh:
		require.Equal(t, resp.llama, d.srv)
		require.Empty(t, s.pendingReqCh)
		require.Empty(t, d.req.errCh)
	case <-ctx.Done():
		t.Fatal("timeout")
	}
	// Wait for b to close
closeWait:
	for {
		select {
		case <-ctx.Done():
			t.Fatal("timeout")
		default:
			if b.srv.closeCalled {
				break closeWait
			}
			time.Sleep(1 * time.Millisecond)
		}
	}
	s.loadedMu.Lock()
	require.Len(t, s.loaded, 2)
	s.loadedMu.Unlock()
}

func TestSchedGetRunner(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 3*time.Second)
	defer done()

	a := newScenarioRequest(t, ctx, "ollama-model-1a", 10, &api.Duration{Duration: 2 * time.Millisecond}, nil)
	b := newScenarioRequest(t, ctx, "ollama-model-1b", 10, &api.Duration{Duration: 2 * time.Millisecond}, nil)
	c := newScenarioRequest(t, ctx, "ollama-model-1c", 10, &api.Duration{Duration: 2 * time.Millisecond}, nil)
	t.Setenv("OLLAMA_MAX_QUEUE", "1")
	s := InitScheduler(ctx)
	s.waitForRecovery = 10 * time.Millisecond
	s.getGpuFn = getGpuFn
	s.getSystemInfoFn = getSystemInfoFn
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

func TestSchedExpireRunner(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 20*time.Millisecond)
	defer done()
	s := InitScheduler(ctx)
	s.waitForRecovery = 10 * time.Millisecond
	req := &LlmRequest{
		ctx:             ctx,
		model:           &Model{ModelPath: "foo"},
		opts:            api.DefaultOptions(),
		successCh:       make(chan *runnerRef, 1),
		errCh:           make(chan error, 1),
		sessionDuration: &api.Duration{Duration: 2 * time.Minute},
	}

	var f *ggml.GGML
	gpus := []ml.DeviceInfo{}
	systemInfo := ml.SystemInfo{}
	server := &mockLlm{vramSize: 10, vramByGPU: map[ml.DeviceID]uint64{}}
	s.newServerFn = func(systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, model string, f *ggml.GGML, adapters []string, projectors []string, opts api.Options, numParallel int) (llm.LlamaServer, error) {
		server.modelPath = model
		return server, nil
	}
	s.load(req, f, systemInfo, gpus, false)

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
func TestSchedPrematureExpired(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 1000*time.Millisecond)
	defer done()

	// Same model, same request
	scenario1a := newScenarioRequest(t, ctx, "ollama-model-1a", 10, &api.Duration{Duration: 100 * time.Millisecond}, nil)
	s := InitScheduler(ctx)
	s.waitForRecovery = 10 * time.Millisecond
	s.getGpuFn = getGpuFn
	s.getSystemInfoFn = getSystemInfoFn
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

func TestSchedUseLoadedRunner(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 100*time.Millisecond)
	req := &LlmRequest{
		ctx:             ctx,
		opts:            api.DefaultOptions(),
		successCh:       make(chan *runnerRef, 1),
		sessionDuration: &api.Duration{Duration: 2},
	}
	finished := make(chan *LlmRequest)
	llm1 := &mockLlm{vramByGPU: map[ml.DeviceID]uint64{}}
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

func TestSchedUpdateFreeSpace(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 100*time.Millisecond)
	defer done()
	gpus := []ml.DeviceInfo{
		{
			DeviceID: ml.DeviceID{
				ID: "1",
			},
		},
		{
			DeviceID: ml.DeviceID{
				ID: "2",
			},
		},
	}
	gpus[0].TotalMemory = 1000
	gpus[0].FreeMemory = 900
	gpus[1].TotalMemory = 2000
	gpus[1].FreeMemory = 1900
	gpuIDs := []ml.DeviceID{
		{
			ID: "1",
		},
		{
			ID: "2",
		},
	}
	llm1 := &mockLlm{vramByGPU: map[ml.DeviceID]uint64{{ID: "1"}: 50, {ID: "2"}: 50}}
	llm2 := &mockLlm{vramByGPU: map[ml.DeviceID]uint64{{ID: "1"}: 125, {ID: "2"}: 75}}
	r1 := &runnerRef{llama: llm1, gpus: gpuIDs, numParallel: 1}
	r2 := &runnerRef{llama: llm2, gpus: gpuIDs, numParallel: 1}

	s := InitScheduler(ctx)
	s.waitForRecovery = 10 * time.Millisecond
	s.loadedMu.Lock()
	s.loaded["a"] = r1
	s.loaded["b"] = r2
	s.loadedMu.Unlock()

	s.updateFreeSpace(gpus)
	require.Equal(t, uint64(1000-50-125), gpus[0].FreeMemory)
	require.Equal(t, uint64(2000-50-75), gpus[1].FreeMemory)
}

func TestSchedFindRunnerToUnload(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 100*time.Millisecond)
	defer done()

	r1 := &runnerRef{refCount: 1, sessionDuration: 1, numParallel: 1}
	r2 := &runnerRef{sessionDuration: 2, numParallel: 1}

	s := InitScheduler(ctx)
	s.waitForRecovery = 10 * time.Millisecond
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

func TestSchedNeedsReload(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 100*time.Millisecond)
	defer done()

	llm := &mockLlm{vramByGPU: map[ml.DeviceID]uint64{}}
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

func TestSchedUnloadAllRunners(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 100*time.Millisecond)
	defer done()

	llm1 := &mockLlm{vramByGPU: map[ml.DeviceID]uint64{}}
	llm2 := &mockLlm{vramByGPU: map[ml.DeviceID]uint64{}}
	s := InitScheduler(ctx)
	s.waitForRecovery = 10 * time.Millisecond
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

func TestSchedUnload(t *testing.T) {
	llm1 := &mockLlm{vramByGPU: map[ml.DeviceID]uint64{}}
	r1 := &runnerRef{llama: llm1, numParallel: 1}
	r2 := &runnerRef{model: &Model{AdapterPaths: []string{"A"}}, numParallel: 1}
	r1.unload()
	require.True(t, llm1.closeCalled)
	r2.unload()
	require.Nil(t, r2.model)
}

func TestSchedAlreadyCanceled(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 500*time.Millisecond)
	defer done()
	dctx, done2 := context.WithCancel(ctx)
	done2()
	scenario1a := newScenarioRequest(t, dctx, "ollama-model-1", 10, &api.Duration{Duration: 0}, nil)
	s := InitScheduler(ctx)
	s.waitForRecovery = 10 * time.Millisecond
	slog.Info("scenario1a")
	s.pendingReqCh <- scenario1a.req
	require.Len(t, s.pendingReqCh, 1)
	s.Run(ctx)
	time.Sleep(5 * time.Millisecond)
	require.Empty(t, s.pendingReqCh)
	require.Empty(t, scenario1a.req.errCh)
	require.Empty(t, scenario1a.req.successCh)
}

type mockLlm struct {
	modelPath         string
	pingResp          error
	waitResp          error
	completionResp    error
	embeddingResp     []float32
	embeddingRespErr  error
	tokenizeResp      []int
	tokenizeRespErr   error
	detokenizeResp    string
	detonekizeRespErr error
	closeResp         error
	closeCalled       bool
	vramSize          uint64
	totalSize         uint64
	vramByGPU         map[ml.DeviceID]uint64
}

func (s *mockLlm) ModelPath() string {
	return s.modelPath
}

func (s *mockLlm) Load(ctx context.Context, sytemInfo ml.SystemInfo, gpus []ml.DeviceInfo, requireFull bool) ([]ml.DeviceID, error) {
	if requireFull {
		if len(gpus) == 0 {
			slog.Info("mockLlm.Load CPU based load")
			return nil, nil
		}
		for _, g := range gpus {
			if g.FreeMemory >= s.vramSize {
				return []ml.DeviceID{g.DeviceID}, nil
			}
		}

		return nil, llm.ErrLoadRequiredFull
	}
	gpuIDs := make([]ml.DeviceID, len(gpus))
	for i := range gpus {
		gpuIDs[i] = gpus[i].DeviceID
	}
	return gpuIDs, nil
}
func (s *mockLlm) Ping(ctx context.Context) error             { return s.pingResp }
func (s *mockLlm) WaitUntilRunning(ctx context.Context) error { return s.waitResp }
func (s *mockLlm) Completion(ctx context.Context, req llm.CompletionRequest, fn func(llm.CompletionResponse)) error {
	return s.completionResp
}

func (s *mockLlm) Embedding(ctx context.Context, input string) ([]float32, int, error) {
	return s.embeddingResp, 0, s.embeddingRespErr
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
func (s *mockLlm) VRAMSize() uint64                                   { return s.vramSize }
func (s *mockLlm) TotalSize() uint64                                  { return s.totalSize }
func (s *mockLlm) VRAMByGPU(id ml.DeviceID) uint64                    { return s.vramByGPU[id] }
func (s *mockLlm) Pid() int                                           { return -1 }
func (s *mockLlm) GetPort() int                                       { return -1 }
func (s *mockLlm) GetDeviceInfos(ctx context.Context) []ml.DeviceInfo { return nil }
func (s *mockLlm) HasExited() bool                                    { return false }
func (s *mockLlm) GetActiveDeviceIDs() []ml.DeviceID                  { return nil }

// TestImageGenRunnerCanBeEvicted verifies that an image generation model
// loaded in the scheduler can be evicted when idle.
func TestImageGenRunnerCanBeEvicted(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 500*time.Millisecond)
	defer done()

	s := InitScheduler(ctx)
	s.getGpuFn = getGpuFn
	s.getSystemInfoFn = getSystemInfoFn

	// Simulate an image gen runner already loaded
	imageGenRunner := &runnerRef{
		model:           &Model{Name: "z-image", ModelPath: "/fake/image/model"},
		modelPath:       "/fake/image/model",
		llama:           &mockLlm{vramSize: 21 * format.GigaByte, vramByGPU: map[ml.DeviceID]uint64{}},
		sessionDuration: 5 * time.Millisecond,
		refCount:        0, // idle
	}

	s.loadedMu.Lock()
	s.loaded["/fake/image/model"] = imageGenRunner
	s.loadedMu.Unlock()

	// Verify the image gen runner is loaded
	s.loadedMu.Lock()
	require.Len(t, s.loaded, 1)
	s.loadedMu.Unlock()

	// findRunnerToUnload should find the idle image gen runner
	runner := s.findRunnerToUnload()
	require.NotNil(t, runner)
	require.Equal(t, "/fake/image/model", runner.modelPath)
}

// TestImageGenSchedulerCoexistence verifies that image generation models
// can coexist with language models in the scheduler and VRAM is tracked correctly.
func TestImageGenSchedulerCoexistence(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 500*time.Millisecond)
	defer done()

	s := InitScheduler(ctx)
	s.getGpuFn = getGpuFn
	s.getSystemInfoFn = getSystemInfoFn

	// Load both an imagegen runner and a language model runner
	imageGenRunner := &runnerRef{
		model:           &Model{Name: "flux", ModelPath: "/fake/flux/model"},
		modelPath:       "/fake/flux/model",
		llama:           &mockLlm{vramSize: 8 * format.GigaByte, vramByGPU: map[ml.DeviceID]uint64{{Library: "Metal"}: 8 * format.GigaByte}},
		sessionDuration: 10 * time.Millisecond,
		numParallel:     1,
		refCount:        0,
	}

	langModelRunner := &runnerRef{
		model:           &Model{Name: "llama3", ModelPath: "/fake/llama3/model"},
		modelPath:       "/fake/llama3/model",
		llama:           &mockLlm{vramSize: 4 * format.GigaByte, vramByGPU: map[ml.DeviceID]uint64{{Library: "Metal"}: 4 * format.GigaByte}},
		sessionDuration: 10 * time.Millisecond,
		numParallel:     1,
		refCount:        0,
	}

	s.loadedMu.Lock()
	s.loaded["/fake/flux/model"] = imageGenRunner
	s.loaded["/fake/llama3/model"] = langModelRunner
	s.loadedMu.Unlock()

	// Verify both are loaded
	s.loadedMu.Lock()
	require.Len(t, s.loaded, 2)
	require.NotNil(t, s.loaded["/fake/flux/model"])
	require.NotNil(t, s.loaded["/fake/llama3/model"])
	s.loadedMu.Unlock()

	// Verify updateFreeSpace accounts for both
	gpus := []ml.DeviceInfo{
		{
			DeviceID:    ml.DeviceID{Library: "Metal"},
			TotalMemory: 24 * format.GigaByte,
			FreeMemory:  24 * format.GigaByte,
		},
	}
	s.updateFreeSpace(gpus)

	// Free memory should be reduced by both models
	expectedFree := uint64(24*format.GigaByte) - uint64(8*format.GigaByte) - uint64(4*format.GigaByte)
	require.Equal(t, expectedFree, gpus[0].FreeMemory)
}
