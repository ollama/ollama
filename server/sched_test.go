package server

import (
	"bytes"
	"context"
	"errors"
	"log/slog"
	"os"
	"sync"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/types/model"
)

func TestMain(m *testing.M) {
	os.Setenv("OLLAMA_DEBUG", "1")
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelDebug}))
	slog.SetDefault(logger)
	os.Exit(m.Run())
}

// syncBuffer is a goroutine-safe sink for capturing slog output: scheduler
// goroutines from a previous test can still be draining — and logging through
// slog.Default — while the current test writes and reads its capture buffer.
type syncBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *syncBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

func (b *syncBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}

func withSynctestScheduler(t *testing.T, fn func(t *testing.T, ctx context.Context)) {
	t.Helper()

	// Tests inside this helper can use time.Sleep to advance synctest's fake
	// clock for scheduler timers without adding wall-clock delay or CI flake.
	synctest.Test(t, func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		fn(t, ctx)
		cancel()
		synctest.Wait()
	})
}

func requireLoadedCount(t *testing.T, s *Scheduler, want int) {
	t.Helper()

	s.loadedMu.Lock()
	defer s.loadedMu.Unlock()
	require.Len(t, s.loaded, want)
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
	withSynctestScheduler(t, func(t *testing.T, ctx context.Context) {
		s := InitScheduler(ctx)
		s.waitForRecovery = 10 * time.Millisecond

		modelPath, _ := createBinFile(t, ggml.KV{
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

		req := &LlmRequest{
			ctx:             ctx,
			model:           &Model{ModelPath: modelPath},
			opts:            api.DefaultOptions(),
			successCh:       make(chan *runnerRef, 1),
			errCh:           make(chan error, 1),
			sessionDuration: &api.Duration{Duration: 2 * time.Second},
		}
		// Fail to load model first
		s.newServerFn = func(systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, model string, f *ggml.GGML, adapters []string, projectors []string, opts api.Options, numParallel int, _ llm.LlamaServerConfig) (llm.LlamaServer, error) {
			return nil, errors.New("something failed to load model blah")
		}
		gpus := []ml.DeviceInfo{}
		systemInfo := ml.SystemInfo{}
		s.load(req, systemInfo, gpus, false)
		require.Empty(t, req.successCh)
		require.Len(t, req.errCh, 1)
		requireLoadedCount(t, s, 0)
		err := <-req.errCh
		require.Contains(t, err.Error(), "this model may be incompatible")

		server := &mockLlm{vramSize: 10, vramByGPU: map[ml.DeviceID]uint64{}}
		s.newServerFn = func(systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, model string, f *ggml.GGML, adapters []string, projectors []string, opts api.Options, numParallel int, _ llm.LlamaServerConfig) (llm.LlamaServer, error) {
			server.modelPath = model
			return server, nil
		}
		s.load(req, systemInfo, gpus, false)
		synctest.Wait()
		select {
		case err := <-req.errCh:
			require.NoError(t, err)
		case resp := <-req.successCh:
			require.Equal(t, uint64(10), resp.vramSize)
			require.Equal(t, uint(1), resp.refCount)
		}
		requireLoadedCount(t, s, 1)

		modelPath2, _ := createBinFile(t, ggml.KV{
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

		req.model.ModelPath = modelPath2
		server.waitResp = errors.New("wait failure")
		s.load(req, systemInfo, gpus, false)
		synctest.Wait()
		select {
		case err := <-req.errCh:
			require.Contains(t, err.Error(), "wait failure")
		case resp := <-req.successCh:
			t.Fatalf("unexpected success %v", resp)
		}
		s.loadedMu.Lock()
		runner := s.loaded[modelPath2]
		s.loadedMu.Unlock()
		require.NotNil(t, runner)
		require.Equal(t, uint(0), runner.refCount)
		require.Len(t, s.expiredCh, 1)
	})
}

func TestSchedLoadStoresEffectiveContextLength(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 500*time.Millisecond)
	defer done()

	s := InitScheduler(ctx)
	scenario := newScenarioRequest(t, ctx, "test", 10, nil, map[ml.DeviceID]uint64{})
	scenario.req.opts.NumCtx = 262144
	scenario.req.numCtxAuto = true
	scenario.srv.contextLength = 131072
	s.newServerFn = scenario.newServer

	s.load(scenario.req, ml.SystemInfo{}, nil, false)

	select {
	case err := <-scenario.req.errCh:
		require.NoError(t, err)
	case runner := <-scenario.req.successCh:
		require.Equal(t, 131072, runner.Options.NumCtx)
	}
}

func TestSchedLoadStoresEffectiveExplicitContextLength(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 500*time.Millisecond)
	defer done()

	s := InitScheduler(ctx)
	scenario := newScenarioRequest(t, ctx, "test", 10, nil, map[ml.DeviceID]uint64{})
	scenario.req.opts.NumCtx = 262144
	scenario.srv.contextLength = 131072
	s.newServerFn = scenario.newServer

	s.load(scenario.req, ml.SystemInfo{}, nil, false)

	select {
	case err := <-scenario.req.errCh:
		require.NoError(t, err)
	case runner := <-scenario.req.successCh:
		require.Equal(t, 131072, runner.Options.NumCtx)
	}
}

func TestSchedVisionContextFloor(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 500*time.Millisecond)
	defer done()

	visionModel := &Model{
		Name: "vision-test",
		Config: model.ConfigV2{
			Capabilities: []string{string(model.CapabilityVision)},
		},
	}

	t.Run("automatic num_ctx is floored", func(t *testing.T) {
		s := InitScheduler(ctx)
		opts := api.DefaultOptions()
		opts.NumCtx = 128

		s.getRunner(ctx, visionModel, opts, nil, true, false, nil)

		req := <-s.pendingReqCh
		require.Equal(t, 2048, req.opts.NumCtx)
		require.True(t, req.numCtxAuto)
	})

	t.Run("explicit num_ctx is floored", func(t *testing.T) {
		s := InitScheduler(ctx)
		opts := api.DefaultOptions()
		opts.NumCtx = 128

		s.getRunner(ctx, visionModel, opts, nil, false, false, nil)

		req := <-s.pendingReqCh
		require.Equal(t, 2048, req.opts.NumCtx)
		require.False(t, req.numCtxAuto)
	})
}

type reqBundle struct {
	ctx     context.Context //nolint:containedctx
	ctxDone func()
	srv     *mockLlm
	req     *LlmRequest
}

func (scenario *reqBundle) newServer(systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, model string, f *ggml.GGML, adapters []string, projectors []string, opts api.Options, numParallel int, _ llm.LlamaServerConfig) (llm.LlamaServer, error) {
	scenario.srv.modelPath = model
	return scenario.srv, nil
}

func newScenarioRequest(t *testing.T, ctx context.Context, modelName string, vramSize uint64, duration *api.Duration, vramByGPU map[ml.DeviceID]uint64) *reqBundle {
	return newScenarioRequestWithContext(t, ctx, modelName, vramSize, duration, vramByGPU, 32)
}

func newScenarioRequestWithContext(t *testing.T, ctx context.Context, modelName string, vramSize uint64, duration *api.Duration, vramByGPU map[ml.DeviceID]uint64, trainCtx uint32) *reqBundle {
	b := &reqBundle{}
	b.ctx, b.ctxDone = context.WithCancel(ctx)
	t.Helper()

	p, _ := createBinFile(t, ggml.KV{
		"general.architecture":          "llama",
		"llama.context_length":          trainCtx,
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
	withSynctestScheduler(t, func(t *testing.T, ctx context.Context) {
		s := InitScheduler(ctx)
		s.waitForRecovery = 10 * time.Millisecond
		g := ml.DeviceInfo{DeviceID: ml.DeviceID{Library: "Metal"}}
		g.TotalMemory = 24 * format.GigaByte
		g.FreeMemory = 12 * format.GigaByte
		gMu := sync.Mutex{}
		s.getGpuFn = func(ctx context.Context, runners []ml.FilteredRunnerDiscovery) []ml.DeviceInfo {
			slog.Info("test getGpuFn called", "runners", runners)
			gMu.Lock()
			defer gMu.Unlock()
			return []ml.DeviceInfo{g}
		}
		s.getSystemInfoFn = getSystemInfoFn
		a := newScenarioRequest(t, ctx, "ollama-model-1", 10, &api.Duration{Duration: 5 * time.Millisecond}, nil)
		b := newScenarioRequest(t, ctx, "ollama-model-1", 20, &api.Duration{Duration: 5 * time.Millisecond}, nil)
		tmpModel := *a.req.model
		b.req.model = &tmpModel

		s.newServerFn = a.newServer
		slog.Info("a")
		s.pendingReqCh <- a.req
		require.Len(t, s.pendingReqCh, 1)
		s.Run(ctx)
		synctest.Wait()
		select {
		case resp := <-a.req.successCh:
			require.Equal(t, resp.llama, a.srv)
			require.Empty(t, s.pendingReqCh)
			require.Empty(t, a.req.errCh)
		case err := <-a.req.errCh:
			t.Fatal(err.Error())
		}

		// Trigger a reload
		s.newServerFn = b.newServer
		b.req.model.AdapterPaths = []string{"new"}
		slog.Info("b")
		s.pendingReqCh <- b.req
		synctest.Wait()
		s.loadedMu.Lock()
		runner := s.loaded[schedulerModelKey(a.req.model)]
		s.loadedMu.Unlock()
		require.NotNil(t, runner)
		runner.refMu.Lock()
		require.True(t, runner.expireOnIdle)
		require.Equal(t, time.Duration(0), runner.sessionDuration)
		runner.refMu.Unlock()
		a.ctxDone()
		gMu.Lock()
		g.FreeMemory = 24 * format.GigaByte
		gMu.Unlock()
		synctest.Wait()
		select {
		case resp := <-b.req.successCh:
			require.Equal(t, resp.llama, b.srv)
			require.Empty(t, s.pendingReqCh)
			require.Empty(t, b.req.errCh)
		case err := <-b.req.errCh:
			t.Fatal(err.Error())
		}
	})
}

func TestSchedRequestsMultipleLoadedModels(t *testing.T) {
	withSynctestScheduler(t, func(t *testing.T, ctx context.Context) {
		slog.Info("TestRequestsMultipleLoadedModels")
		s := InitScheduler(ctx)
		s.waitForRecovery = 10 * time.Millisecond
		g := ml.DeviceInfo{DeviceID: ml.DeviceID{Library: "Metal"}}
		g.TotalMemory = 24 * format.GigaByte
		g.FreeMemory = 12 * format.GigaByte
		gMu := sync.Mutex{}
		s.getGpuFn = func(ctx context.Context, runners []ml.FilteredRunnerDiscovery) []ml.DeviceInfo {
			slog.Info("test getGpuFn called", "runners", runners)
			gMu.Lock()
			defer gMu.Unlock()
			return []ml.DeviceInfo{g}
		}
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
		synctest.Wait()
		select {
		case resp := <-a.req.successCh:
			require.Equal(t, resp.llama, a.srv)
			require.Empty(t, s.pendingReqCh)
			require.Empty(t, a.req.errCh)
		case err := <-a.req.errCh:
			t.Fatal(err.Error())
		}
		requireLoadedCount(t, s, 1)

		t.Setenv("OLLAMA_MAX_LOADED_MODELS", "0")
		s.newServerFn = b.newServer
		slog.Info("Loading B")
		s.pendingReqCh <- b.req
		synctest.Wait()
		select {
		case resp := <-b.req.successCh:
			require.Equal(t, resp.llama, b.srv)
			require.Empty(t, s.pendingReqCh)
			require.Empty(t, b.req.errCh)
		case err := <-b.req.errCh:
			t.Fatal(err.Error())
		}
		requireLoadedCount(t, s, 2)

		// This is a CPU load with NumGPU = 0 so it should load
		s.newServerFn = c.newServer
		slog.Info("Loading C")
		s.pendingReqCh <- c.req
		synctest.Wait()
		select {
		case resp := <-c.req.successCh:
			require.Equal(t, resp.llama, c.srv)
			require.Empty(t, s.pendingReqCh)
			require.Empty(t, c.req.errCh)
		case err := <-c.req.errCh:
			t.Fatal(err.Error())
		}
		requireLoadedCount(t, s, 3)

		// Try to load a model that won't fit
		s.newServerFn = d.newServer
		slog.Info("d")
		requireLoadedCount(t, s, 3)
		a.ctxDone() // Won't help since this one isn't big enough to make room
		s.pendingReqCh <- d.req
		synctest.Wait()
		requireLoadedCount(t, s, 2)

		// Mark b done so it can unload
		b.ctxDone()
		// Report recovered VRAM usage so scheduler will finish waiting and unload
		gMu.Lock()
		g.FreeMemory = 24 * format.GigaByte
		gMu.Unlock()
		synctest.Wait()
		resp := <-d.req.successCh
		require.Equal(t, resp.llama, d.srv)
		require.Empty(t, s.pendingReqCh)
		require.Empty(t, d.req.errCh)
		require.True(t, b.srv.closeCalled)
		requireLoadedCount(t, s, 2)
	})
}

func TestSchedGetRunner(t *testing.T) {
	withSynctestScheduler(t, func(t *testing.T, ctx context.Context) {
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
		successCh1a, errCh1a := s.getRunner(a.ctx, a.req.model, a.req.opts, a.req.sessionDuration, false, false, nil)
		require.Len(t, s.pendingReqCh, 1)
		slog.Info("b")
		successCh1b, errCh1b := s.getRunner(b.ctx, b.req.model, b.req.opts, b.req.sessionDuration, false, false, nil)
		require.Len(t, s.pendingReqCh, 1)
		require.Empty(t, successCh1b)
		require.Len(t, errCh1b, 1)
		err := <-errCh1b
		require.Contains(t, err.Error(), "server busy")
		s.Run(ctx)
		synctest.Wait()
		select {
		case resp := <-successCh1a:
			require.Equal(t, resp.llama, a.srv)
			require.Empty(t, s.pendingReqCh)
			require.Empty(t, errCh1a)
		case err := <-errCh1a:
			t.Fatal(err.Error())
		}
		a.ctxDone() // Set "a" model to idle so it can unload
		requireLoadedCount(t, s, 1)

		c.req.model.ModelPath = "bad path"
		slog.Info("c")
		successCh1c, errCh1c := s.getRunner(c.ctx, c.req.model, c.req.opts, c.req.sessionDuration, false, false, nil)
		synctest.Wait()
		require.Empty(t, successCh1c)
		time.Sleep(a.req.sessionDuration.Duration)
		synctest.Wait()
		requireLoadedCount(t, s, 0)
		err = <-errCh1c
		require.Contains(t, err.Error(), "bad path")
		b.ctxDone()
	})
}

func TestSchedGetRunnerUsesDigestKeyWhenModelPathEmpty(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 100*time.Millisecond)
	defer done()

	s := InitScheduler(ctx)
	opts := api.DefaultOptions()
	opts.NumCtx = 4

	loadedModel := &Model{Name: "safetensors-a", Digest: "sha-a"}
	loadedRunner := &runnerRef{
		model:       loadedModel,
		modelKey:    schedulerModelKey(loadedModel),
		llama:       &mockLlm{vramByGPU: map[ml.DeviceID]uint64{}},
		Options:     &opts,
		numParallel: 1,
	}

	s.loadedMu.Lock()
	s.loaded[loadedRunner.modelKey] = loadedRunner
	s.loadedMu.Unlock()

	reqModel := &Model{Name: "safetensors-b", Digest: "sha-b"}
	successCh, errCh := s.getRunner(ctx, reqModel, opts, nil, false, false, nil)

	require.Empty(t, successCh)
	require.Empty(t, errCh)
	require.Len(t, s.pendingReqCh, 1)
}

func TestSchedGetRunnerReusesSameDigestWhenModelPathEmpty(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 100*time.Millisecond)
	defer done()

	s := InitScheduler(ctx)
	opts := api.DefaultOptions()
	opts.NumCtx = 4

	loadedModel := &Model{Name: "safetensors-a", Digest: "sha-a"}
	loadedRunner := &runnerRef{
		model:       loadedModel,
		modelKey:    schedulerModelKey(loadedModel),
		llama:       &mockLlm{vramByGPU: map[ml.DeviceID]uint64{}},
		Options:     &opts,
		numParallel: 1,
	}

	s.loadedMu.Lock()
	s.loaded[loadedRunner.modelKey] = loadedRunner
	s.loadedMu.Unlock()

	reqCtx, cancelReq := context.WithCancel(ctx)
	successCh, errCh := s.getRunner(reqCtx, &Model{Name: "safetensors-a-copy", Digest: "sha-a"}, opts, nil, false, false, nil)
	cancelReq()

	select {
	case runner := <-successCh:
		require.Equal(t, loadedRunner, runner)
	default:
		t.Fatal("expected existing runner to be reused")
	}

	require.Empty(t, errCh)
	require.Empty(t, s.pendingReqCh)
}

func TestSchedExpireRunner(t *testing.T) {
	withSynctestScheduler(t, func(t *testing.T, ctx context.Context) {
		schedCtx, done := context.WithCancel(ctx)
		s := InitScheduler(schedCtx)
		s.waitForRecovery = 10 * time.Millisecond

		modelPath, _ := createBinFile(t, ggml.KV{
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

		reqCtx, cancelReq := context.WithCancel(schedCtx)
		defer cancelReq()

		req := &LlmRequest{
			ctx:             reqCtx,
			model:           &Model{ModelPath: modelPath},
			opts:            api.DefaultOptions(),
			successCh:       make(chan *runnerRef, 1),
			errCh:           make(chan error, 1),
			sessionDuration: &api.Duration{Duration: 2 * time.Minute},
		}

		gpus := []ml.DeviceInfo{}
		systemInfo := ml.SystemInfo{}
		server := &mockLlm{vramSize: 10, vramByGPU: map[ml.DeviceID]uint64{}}
		s.newServerFn = func(systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, model string, f *ggml.GGML, adapters []string, projectors []string, opts api.Options, numParallel int, _ llm.LlamaServerConfig) (llm.LlamaServer, error) {
			server.modelPath = model
			return server, nil
		}
		s.load(req, systemInfo, gpus, false)
		synctest.Wait()

		select {
		case err := <-req.errCh:
			if err != nil {
				t.Fatalf("expected no errors when loading, got '%s'", err.Error())
			}
		case resp := <-req.successCh:
			require.Equal(t, uint(1), resp.refCount)
			requireLoadedCount(t, s, 1)
		}

		completedDone := make(chan struct{})
		go func() {
			defer close(completedDone)
			s.processCompleted(schedCtx)
		}()

		s.expireRunner(&Model{ModelPath: modelPath})
		cancelReq()
		synctest.Wait()
		require.Len(t, s.unloadedCh, 1)
		<-s.unloadedCh
		requireLoadedCount(t, s, 0)

		done()
		synctest.Wait()
		select {
		case <-completedDone:
		default:
			t.Fatal("expected completed loop to stop")
		}
	})
}

// TODO - add one scenario that triggers the bogus finished event with positive ref count
func TestSchedPrematureExpired(t *testing.T) {
	withSynctestScheduler(t, func(t *testing.T, ctx context.Context) {
		// Same model, same request
		scenario1a := newScenarioRequest(t, ctx, "ollama-model-1a", 10, &api.Duration{Duration: 100 * time.Millisecond}, nil)
		s := InitScheduler(ctx)
		s.waitForRecovery = 10 * time.Millisecond
		s.getGpuFn = getGpuFn
		s.getSystemInfoFn = getSystemInfoFn
		s.newServerFn = scenario1a.newServer
		successCh1a, errCh1a := s.getRunner(scenario1a.ctx, scenario1a.req.model, scenario1a.req.opts, scenario1a.req.sessionDuration, false, false, nil)
		require.Len(t, s.pendingReqCh, 1)
		s.Run(ctx)
		synctest.Wait()
		var resp *runnerRef
		select {
		case resp = <-successCh1a:
			require.Equal(t, resp.llama, scenario1a.srv)
			require.Empty(t, s.pendingReqCh)
			require.Empty(t, errCh1a)
			requireLoadedCount(t, s, 1)
			slog.Info("sending premature expired event now")
			s.expiredCh <- resp // Shouldn't happen in real life, but make sure its safe
		case err := <-errCh1a:
			t.Fatal(err.Error())
		}
		synctest.Wait()
		resp.refMu.Lock()
		require.True(t, resp.expireOnIdle)
		resp.refMu.Unlock()
		scenario1a.ctxDone()
		synctest.Wait()
		requireLoadedCount(t, s, 0)

		// also shouldn't happen in real life; wait until the scheduler has safely
		// consumed the bogus event
		s.finishedReqCh <- scenario1a.req
		synctest.Wait()
		require.Empty(t, s.finishedReqCh)
	})
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
	req.tryUseLoadedRunner(r1, finished)
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

func TestSchedUseLoadedRunnerCanceledDoesNotIncrementRefCount(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	cancel()

	req := &LlmRequest{
		ctx:             ctx,
		opts:            api.DefaultOptions(),
		successCh:       make(chan *runnerRef, 1),
		errCh:           make(chan error, 1),
		sessionDuration: &api.Duration{Duration: 2},
	}
	finished := make(chan *LlmRequest)
	runner := &runnerRef{llama: &mockLlm{vramByGPU: map[ml.DeviceID]uint64{}}, sessionDuration: 1, numParallel: 1}

	require.True(t, req.tryUseLoadedRunner(runner, finished))
	require.Equal(t, uint(0), runner.refCount)
	require.Equal(t, time.Duration(1), runner.sessionDuration)
	require.Empty(t, req.successCh)
	select {
	case err := <-req.errCh:
		require.ErrorIs(t, err, context.Canceled)
	default:
		t.Fatal("expected canceled request error")
	}
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

func TestSchedUpdateFreeSpaceDoesNotHoldRunnerLockDuringVRAMRefresh(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 100*time.Millisecond)
	defer done()

	gpuID := ml.DeviceID{ID: "1", Library: "CUDA"}
	gpus := []ml.DeviceInfo{{
		DeviceID:    gpuID,
		TotalMemory: 1000,
		FreeMemory:  900,
	}}

	var runner *runnerRef
	llm := &mockLlm{
		vramByGPUFn: func(id ml.DeviceID) uint64 {
			require.True(t, runner.refMu.TryLock(), "VRAMByGPU called while runner.refMu was held")
			runner.refMu.Unlock()
			require.Equal(t, gpuID, id)
			return 125
		},
	}
	runner = &runnerRef{llama: llm, gpus: []ml.DeviceID{gpuID}, numParallel: 1}

	s := InitScheduler(ctx)
	s.loadedMu.Lock()
	s.loaded["a"] = runner
	s.loadedMu.Unlock()

	s.updateFreeSpace(gpus)
	require.Equal(t, uint64(1000-125), gpus[0].FreeMemory)
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
	req.contextShift = true
	resp = runner.needsReload(ctx, req)
	require.True(t, resp)
}

func TestResolveContextShift(t *testing.T) {
	trueValue := true
	falseValue := false

	tests := []struct {
		name  string
		shift *bool
		model *Model
		want  bool
	}{
		{name: "unset defaults to shift", want: true},
		{name: "unset deepseek2 disables shift", model: &Model{Config: model.ConfigV2{ModelFamily: "deepseek2"}}, want: false},
		{name: "unset deepseek2 family disables shift", model: &Model{Config: model.ConfigV2{ModelFamilies: []string{"llama", "deepseek2"}}}, want: false},
		{name: "explicit false disables shift", shift: &falseValue, want: false},
		{name: "explicit false disables shift for deepseek2", shift: &falseValue, model: &Model{Config: model.ConfigV2{ModelFamily: "deepseek2"}}, want: false},
		{name: "explicit true enables shift", shift: &trueValue, want: true},
		{name: "explicit true enables shift for deepseek2", shift: &trueValue, model: &Model{Config: model.ConfigV2{ModelFamily: "deepseek2"}}, want: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			require.Equal(t, tt.want, resolveContextShift(tt.shift, tt.model))
		})
	}
}

func TestSchedNeedsReloadIgnoresAutomaticNumCtxClamp(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 100*time.Millisecond)
	defer done()

	llm := &mockLlm{vramByGPU: map[ml.DeviceID]uint64{}}
	opts := api.DefaultOptions()
	opts.NumCtx = 131072
	model := &Model{}
	runner := &runnerRef{
		model:       model,
		Options:     &opts,
		llama:       llm,
		numParallel: 1,
		numCtxAuto:  true,
	}
	req := &LlmRequest{
		model:      model,
		opts:       api.DefaultOptions(),
		numCtxAuto: true,
	}
	req.opts.NumCtx = 262144

	require.False(t, runner.needsReload(ctx, req))

	req.numCtxAuto = false
	require.True(t, runner.needsReload(ctx, req))
}

func TestSchedNeedsReloadUsesEffectiveAutomaticContextShift(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 100*time.Millisecond)
	defer done()

	llm := &mockLlm{vramByGPU: map[ml.DeviceID]uint64{}}
	opts := api.DefaultOptions()
	opts.NumCtx = 128
	model := &Model{ModelPath: "model.gguf"}
	runner := &runnerRef{
		model:        model,
		Options:      &opts,
		llama:        llm,
		numParallel:  1,
		numCtxAuto:   true,
		contextShift: true,
	}
	req := &LlmRequest{
		model:      model,
		opts:       api.DefaultOptions(),
		numCtxAuto: true,
	}
	req.opts.NumCtx = 262144

	require.False(t, runner.needsReload(ctx, req))

	req.numCtxAuto = false
	require.True(t, runner.needsReload(ctx, req))
}

func TestSchedNeedsReloadUsesEffectiveExplicitContext(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 100*time.Millisecond)
	defer done()

	llm := &mockLlm{vramByGPU: map[ml.DeviceID]uint64{}}
	opts := api.DefaultOptions()
	opts.NumCtx = 2048
	model := &Model{ModelPath: "model.gguf"}
	runner := &runnerRef{
		model:        model,
		Options:      &opts,
		llama:        llm,
		numParallel:  1,
		contextShift: true,
		trainContext: 2048,
	}
	req := &LlmRequest{
		model: model,
		opts:  api.DefaultOptions(),
	}
	req.opts.NumCtx = 262144

	require.False(t, runner.needsReload(ctx, req))

	req.opts.NumCtx = 1024
	require.True(t, runner.needsReload(ctx, req))
}

func TestSchedNeedsReloadIgnoresAutomaticNumBatchDerivation(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 100*time.Millisecond)
	defer done()

	llm := &mockLlm{vramByGPU: map[ml.DeviceID]uint64{}}
	opts := api.DefaultOptions()
	opts.NumBatch = 1024
	model := &Model{}
	runner := &runnerRef{
		model:        model,
		Options:      &opts,
		llama:        llm,
		numParallel:  1,
		numBatchAuto: true,
	}
	req := &LlmRequest{
		model:        model,
		opts:         api.DefaultOptions(),
		numBatchAuto: true,
	}
	req.opts.NumBatch = 512

	require.False(t, runner.needsReload(ctx, req))

	req.numBatchAuto = false
	require.True(t, runner.needsReload(ctx, req))
}

func TestSchedNeedsReloadIgnoresAutomaticUseMMapDefault(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 100*time.Millisecond)
	defer done()

	llm := &mockLlm{vramByGPU: map[ml.DeviceID]uint64{}}
	useMmap := false
	opts := api.DefaultOptions()
	opts.UseMMap = &useMmap
	model := &Model{}
	runner := &runnerRef{
		model:       model,
		Options:     &opts,
		llama:       llm,
		numParallel: 1,
		useMMapAuto: true,
	}
	req := &LlmRequest{
		model: model,
		opts:  api.DefaultOptions(),
	}

	require.False(t, runner.needsReload(ctx, req))

	explicitUseMmap := true
	req.opts.UseMMap = &explicitUseMmap
	require.True(t, runner.needsReload(ctx, req))

	req.opts.UseMMap = &useMmap
	require.False(t, runner.needsReload(ctx, req))

	runner.useMMapAuto = false
	req.opts.UseMMap = nil
	require.True(t, runner.needsReload(ctx, req))
}

func TestAutomaticGenerationBatch(t *testing.T) {
	tests := []struct {
		name         string
		effectiveCtx int
		predicted    uint64
		available    uint64
		flash        ml.FlashAttentionType
		gpus         []ml.DeviceInfo
		want         int
	}{
		{
			name:         "small context keeps default",
			effectiveCtx: 4096,
			flash:        ml.FlashAttentionAuto,
			want:         llamaServerGenerationBatchDefault,
		},
		{
			name:         "medium context uses medium batch with unknown memory",
			effectiveCtx: 32768,
			flash:        ml.FlashAttentionAuto,
			want:         llamaServerGenerationBatchMedium,
		},
		{
			name:         "large context uses large batch with headroom",
			effectiveCtx: 131072,
			predicted:    8 * format.GibiByte,
			available:    14 * format.GibiByte,
			flash:        ml.FlashAttentionAuto,
			want:         llamaServerGenerationBatchLarge,
		},
		{
			name:         "large context steps down to medium batch without large batch headroom",
			effectiveCtx: 131072,
			predicted:    9 * format.GibiByte,
			available:    14 * format.GibiByte,
			flash:        ml.FlashAttentionAuto,
			want:         llamaServerGenerationBatchMedium,
		},
		{
			name:         "large context steps down to medium batch for headroom",
			effectiveCtx: 131072,
			predicted:    8 * format.GibiByte,
			available:    11 * format.GibiByte,
			flash:        ml.FlashAttentionAuto,
			want:         llamaServerGenerationBatchMedium,
		},
		{
			name:         "medium context steps down to default batch for headroom",
			effectiveCtx: 32768,
			predicted:    8500 * format.MebiByte,
			available:    11 * format.GibiByte,
			flash:        ml.FlashAttentionAuto,
			want:         llamaServerGenerationBatchDefault,
		},
		{
			name:         "flash attention disabled suppresses promotion",
			effectiveCtx: 131072,
			predicted:    8 * format.GibiByte,
			available:    14 * format.GibiByte,
			flash:        ml.FlashAttentionDisabled,
			gpus:         []ml.DeviceInfo{{DeviceID: ml.DeviceID{Library: "CUDA"}, FreeMemory: 14 * format.GibiByte}},
			want:         llamaServerGenerationBatchDefault,
		},
		{
			name:         "constrained CUDA without flash attention uses smaller batch",
			effectiveCtx: 131072,
			predicted:    3 * format.GibiByte,
			available:    6 * format.GibiByte,
			flash:        ml.FlashAttentionDisabled,
			gpus:         []ml.DeviceInfo{{DeviceID: ml.DeviceID{Library: "CUDA"}, FreeMemory: 6 * format.GibiByte}},
			want:         llamaServerGenerationBatchConstrained,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			require.Equal(t, tt.want, automaticGenerationBatch(tt.effectiveCtx, tt.predicted, tt.available, tt.flash, tt.gpus))
		})
	}
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

func TestRunnerRefDiscoveryAccessorsUseSnapshots(t *testing.T) {
	gpuID := ml.DeviceID{ID: "0", Library: "CUDA"}
	opts := api.DefaultOptions()
	opts.NumCtx = 2048
	runner := &runnerRef{
		llama:     &mockLlm{port: 12345, deviceInfos: []ml.DeviceInfo{{DeviceID: gpuID}}},
		model:     &Model{Name: "test"},
		modelPath: "test",
		modelKey:  "test",
		gpus:      []ml.DeviceID{gpuID},
		totalSize: 1,
		vramSize:  1,
		Options:   &opts,
	}

	require.Equal(t, 12345, runner.GetPort())
	require.False(t, runner.HasExited())
	require.Equal(t, []ml.DeviceInfo{{DeviceID: gpuID}}, runner.GetDeviceInfos(t.Context()))

	gpus := runner.GetActiveDeviceIDs()
	require.Equal(t, []ml.DeviceID{gpuID}, gpus)
	gpus[0].ID = "changed"
	runner.refMu.Lock()
	require.Equal(t, gpuID, runner.gpus[0])
	_ = runner.LogValue()
	runner.refMu.Unlock()

	runner.unload()
	require.Equal(t, -1, runner.GetPort())
	require.True(t, runner.HasExited())
	require.Nil(t, runner.GetDeviceInfos(t.Context()))
	require.Empty(t, runner.GetActiveDeviceIDs())
}

func TestSchedLoadedModelsDoesNotBlockDuringVRAMRecovery(t *testing.T) {
	withSynctestScheduler(t, func(t *testing.T, ctx context.Context) {
		s := InitScheduler(ctx)
		s.waitForRecovery = 10 * time.Millisecond

		gpuQueryEntered := make(chan struct{})
		releaseGPUQuery := make(chan struct{})
		var enterOnce sync.Once
		s.getGpuFn = func(ctx context.Context, runners []ml.FilteredRunnerDiscovery) []ml.DeviceInfo {
			enterOnce.Do(func() {
				close(gpuQueryEntered)
				select {
				case <-releaseGPUQuery:
				case <-ctx.Done():
				}
			})
			gpu := ml.DeviceInfo{DeviceID: ml.DeviceID{ID: "0", Library: "CUDA"}}
			gpu.TotalMemory = 24 * format.GigaByte
			gpu.FreeMemory = 12 * format.GigaByte
			return []ml.DeviceInfo{gpu}
		}

		model := &Model{ModelPath: "resident"}
		runner := &runnerRef{
			llama:           &mockLlm{vramByGPU: map[ml.DeviceID]uint64{}},
			model:           model,
			modelPath:       model.ModelPath,
			modelKey:        schedulerModelKey(model),
			pid:             1,
			gpus:            []ml.DeviceID{{ID: "0", Library: "CUDA"}},
			usesDiscreteGPU: true,
			vramSize:        1 * format.GigaByte,
		}
		s.loadedMu.Lock()
		s.loaded[runner.modelKey] = runner
		s.loadedMu.Unlock()

		go s.processCompleted(ctx)
		s.expiredCh <- runner
		synctest.Wait()
		select {
		case <-gpuQueryEntered:
		default:
			t.Fatal("expected VRAM recovery to start GPU discovery")
		}

		loadedModelsDone := make(chan []loadedModel, 1)
		go func() {
			loadedModelsDone <- s.loadedModels()
		}()
		synctest.Wait()
		select {
		case models := <-loadedModelsDone:
			require.Empty(t, models)
		default:
			t.Fatal("loadedModels blocked behind VRAM recovery")
		}

		close(releaseGPUQuery)
		synctest.Wait()
		time.Sleep(s.waitForRecovery)
		synctest.Wait()
		require.Len(t, s.unloadedCh, 1)
		<-s.unloadedCh
	})
}

func TestSchedLoadedModelsDoesNotHoldRunnerLockDuringMemoryRefresh(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 100*time.Millisecond)
	defer done()

	model := &Model{Name: "test", ModelPath: "/fake/model"}
	var runner *runnerRef
	llm := &mockLlm{
		contextLength: 2048,
		memorySizeFn: func() (uint64, uint64) {
			require.True(t, runner.refMu.TryLock(), "MemorySize called while runner.refMu was held")
			runner.refMu.Unlock()
			return 10 * format.GigaByte, 8 * format.GigaByte
		},
	}
	runner = &runnerRef{
		model:           model,
		modelPath:       model.ModelPath,
		modelKey:        schedulerModelKey(model),
		llama:           llm,
		sessionDuration: time.Minute,
		totalSize:       1,
		vramSize:        1,
	}

	s := InitScheduler(ctx)
	s.loadedMu.Lock()
	s.loaded[runner.modelKey] = runner
	s.loadedMu.Unlock()

	models := s.loadedModels()
	require.Len(t, models, 1)
	require.Equal(t, int64(10*format.GigaByte), models[0].size)
	require.Equal(t, int64(8*format.GigaByte), models[0].sizeVRAM)
	require.Equal(t, 2048, models[0].contextLength)
}

func TestSchedAlreadyCanceled(t *testing.T) {
	withSynctestScheduler(t, func(t *testing.T, ctx context.Context) {
		dctx, done2 := context.WithCancel(ctx)
		done2()
		scenario1a := newScenarioRequest(t, dctx, "ollama-model-1", 10, &api.Duration{Duration: 0}, nil)
		s := InitScheduler(ctx)
		s.waitForRecovery = 10 * time.Millisecond
		slog.Info("scenario1a")
		s.pendingReqCh <- scenario1a.req
		require.Len(t, s.pendingReqCh, 1)
		s.Run(ctx)
		synctest.Wait()
		require.Empty(t, s.pendingReqCh)
		select {
		case err := <-scenario1a.req.errCh:
			require.ErrorIs(t, err, context.Canceled)
		default:
			t.Fatal("expected canceled request error")
		}
		require.Empty(t, scenario1a.req.successCh)
	})
}

func TestSchedCanceledRequestDoesNotEnterPendingQueue(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	cancel()

	s := InitScheduler(t.Context())
	_, errCh := s.getRunner(ctx, &Model{Name: "canceled"}, api.DefaultOptions(), nil, false, false, nil)

	select {
	case err := <-errCh:
		require.ErrorIs(t, err, context.Canceled)
	default:
		t.Fatal("expected canceled request error")
	}
	require.Empty(t, s.pendingReqCh)
}

func TestSchedPendingCanceledDuringUnloadWaitDoesNotRetry(t *testing.T) {
	withSynctestScheduler(t, func(t *testing.T, ctx context.Context) {
		s := InitScheduler(ctx)
		s.waitForRecovery = 10 * time.Millisecond
		s.getGpuFn = getGpuFn
		s.getSystemInfoFn = getSystemInfoFn

		resident := &runnerRef{
			llama:     &mockLlm{vramByGPU: map[ml.DeviceID]uint64{}},
			modelPath: "resident",
			modelKey:  "resident",
		}
		s.loadedMu.Lock()
		s.loaded[resident.modelKey] = resident
		s.loadedMu.Unlock()

		var loadCalls atomic.Int32
		s.loadFn = func(req *LlmRequest, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, requireFull bool) bool {
			loadCalls.Add(1)
			return true
		}

		pending := newScenarioRequest(t, ctx, "pending", 1, nil, nil)
		go s.processPending(ctx)
		s.pendingReqCh <- pending.req
		synctest.Wait()

		select {
		case runner := <-s.expiredCh:
			require.Equal(t, resident, runner)
		default:
			t.Fatal("expected resident runner to be marked for expiration")
		}

		pending.ctxDone()
		s.unloadedCh <- struct{}{}
		synctest.Wait()

		select {
		case err := <-pending.req.errCh:
			require.ErrorIs(t, err, context.Canceled)
		default:
			t.Fatal("expected canceled request error")
		}
		require.Equal(t, int32(1), loadCalls.Load())
	})
}

// hasLoadedRunner is a test helper that checks if any runner is loaded.
func hasLoadedRunner(s *Scheduler) bool {
	s.loadedMu.Lock()
	defer s.loadedMu.Unlock()
	for _, r := range s.loaded {
		r.refMu.Lock()
		llama := r.llama
		r.refMu.Unlock()
		if llama != nil {
			return true
		}
	}
	return false
}

func TestSchedLlamaServerEvictsWhenVRAMInsufficient(t *testing.T) {
	// When a llama-server model is predicted to exceed available VRAM,
	// the scheduler should signal eviction before spawning
	ctx, done := context.WithTimeout(t.Context(), 500*time.Millisecond)
	defer done()
	s := InitScheduler(ctx)
	s.waitForRecovery = 10 * time.Millisecond
	// GPU with very little free memory — the model won't fit
	s.getGpuFn = func(ctx context.Context, runners []ml.FilteredRunnerDiscovery) []ml.DeviceInfo {
		g := ml.DeviceInfo{DeviceID: ml.DeviceID{Library: "Metal"}}
		g.TotalMemory = 24 * format.GigaByte
		g.FreeMemory = 0 // no free VRAM — forces eviction
		return []ml.DeviceInfo{g}
	}
	s.getSystemInfoFn = getSystemInfoFn

	// Pre-load a regular model
	s.loadedMu.Lock()
	s.loaded["existing-model"] = &runnerRef{
		llama:    &mockLlm{modelPath: "existing"},
		modelKey: "existing-model",
	}
	s.loadedMu.Unlock()

	// Create a request — the model file + KV cache will exceed 100 MiB
	scenario := newScenarioRequest(t, ctx, "llama-server-model", 1*format.GigaByte, nil, nil)

	s.newServerFn = func(systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, model string, f *ggml.GGML, adapters []string, projectors []string, opts api.Options, numParallel int, _ llm.LlamaServerConfig) (llm.LlamaServer, error) {
		return &mockLlm{modelPath: model}, nil
	}

	systemInfo := getSystemInfoFn()
	gpus := s.getGpuFn(ctx, nil)

	needEvict := s.load(scenario.req, systemInfo, gpus, true)
	require.True(t, needEvict, "expected eviction when predicted VRAM exceeds free memory")
}

func TestSchedLlamaServerExplicitPartialNumGPUSkipsFullFitEviction(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 500*time.Millisecond)
	defer done()
	s := InitScheduler(ctx)
	s.waitForRecovery = 10 * time.Millisecond
	s.getGpuFn = func(ctx context.Context, runners []ml.FilteredRunnerDiscovery) []ml.DeviceInfo {
		g := ml.DeviceInfo{DeviceID: ml.DeviceID{Library: "Metal"}}
		g.TotalMemory = 24 * format.GigaByte
		g.FreeMemory = 0
		return []ml.DeviceInfo{g}
	}
	s.getSystemInfoFn = getSystemInfoFn

	s.loadedMu.Lock()
	s.loaded["existing-model"] = &runnerRef{
		llama:    &mockLlm{modelPath: "existing"},
		modelKey: "existing-model",
	}
	s.loadedMu.Unlock()

	scenario := newScenarioRequest(t, ctx, "partial-llama-server-model", 1*format.GigaByte, nil, nil)
	scenario.req.opts.NumGPU = 1
	scenario.srv.vramSize = 0

	called := false
	s.newServerFn = func(systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, model string, f *ggml.GGML, adapters []string, projectors []string, opts api.Options, numParallel int, _ llm.LlamaServerConfig) (llm.LlamaServer, error) {
		called = true
		require.Equal(t, 1, opts.NumGPU)
		return scenario.srv, nil
	}

	systemInfo := getSystemInfoFn()
	gpus := s.getGpuFn(ctx, nil)

	needEvict := s.load(scenario.req, systemInfo, gpus, true)
	require.False(t, needEvict, "explicit partial offload should not trigger full-fit eviction")
	require.True(t, called, "scheduler should try the explicitly partial load")
}

func TestSchedLlamaServerFitsAlongside(t *testing.T) {
	// When a llama-server model is predicted to fit in remaining VRAM,
	// it should load without evicting existing models
	ctx, done := context.WithTimeout(t.Context(), 500*time.Millisecond)
	defer done()
	s := InitScheduler(ctx)
	s.waitForRecovery = 10 * time.Millisecond
	// GPU with plenty of free memory
	s.getGpuFn = func(ctx context.Context, runners []ml.FilteredRunnerDiscovery) []ml.DeviceInfo {
		g := ml.DeviceInfo{DeviceID: ml.DeviceID{Library: "Metal"}}
		g.TotalMemory = 24 * format.GigaByte
		g.FreeMemory = 20 * format.GigaByte
		return []ml.DeviceInfo{g}
	}
	s.getSystemInfoFn = getSystemInfoFn

	// Pre-load a regular model
	s.loadedMu.Lock()
	s.loaded["existing-model"] = &runnerRef{
		llama:    &mockLlm{modelPath: "existing"},
		modelKey: "existing-model",
	}
	s.loadedMu.Unlock()

	// The test model GGUF is tiny (~64 bytes) — should easily fit in 20 GiB
	scenario := newScenarioRequest(t, ctx, "small-llama-server", 1*format.GigaByte, nil, nil)

	s.newServerFn = scenario.newServer

	systemInfo := getSystemInfoFn()
	gpus := s.getGpuFn(ctx, nil)

	// Should NOT evict — model fits alongside existing
	needEvict := s.load(scenario.req, systemInfo, gpus, true)
	require.False(t, needEvict, "expected no eviction when model fits in available VRAM")
}

func TestSchedRetriesWhenLlamaServerSpillsToCPUWithLoadedRunner(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 500*time.Millisecond)
	defer done()

	var buf syncBuffer
	previous := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug})))
	defer slog.SetDefault(previous)

	s := InitScheduler(ctx)
	s.waitForRecovery = 10 * time.Millisecond
	s.getGpuFn = func(ctx context.Context, runners []ml.FilteredRunnerDiscovery) []ml.DeviceInfo {
		g := ml.DeviceInfo{DeviceID: ml.DeviceID{Library: "Metal"}}
		g.TotalMemory = 24 * format.GigaByte
		g.FreeMemory = 20 * format.GigaByte
		return []ml.DeviceInfo{g}
	}
	s.getSystemInfoFn = getSystemInfoFn

	s.loadedMu.Lock()
	s.loaded["existing-model"] = &runnerRef{
		llama:    &mockLlm{modelPath: "/fake/existing"},
		modelKey: "existing-model",
	}
	s.loadedMu.Unlock()

	scenario := newScenarioRequest(t, ctx, "cpu-spill-model", 1*format.GigaByte, nil, nil)
	scenario.srv.totalSize = 2 * format.GigaByte
	scenario.srv.vramSize = 1 * format.GigaByte
	scenario.srv.gpuLayers = 20
	scenario.srv.totalLayers = 33
	s.newServerFn = scenario.newServer

	systemInfo := getSystemInfoFn()
	gpus := s.getGpuFn(ctx, nil)

	require.True(t, s.load(scenario.req, systemInfo, gpus, true))
	require.True(t, scenario.req.loadRetryAttempted)
	require.True(t, scenario.srv.closeCalled)
	require.Nil(t, s.activeLoading)
	require.Contains(t, buf.String(), "llama-server spilled layers to CPU with other models resident; evicting residents and retrying")
	require.Contains(t, buf.String(), "gpu_layers=20")
	require.Contains(t, buf.String(), "total_layers=33")
}

func TestSchedAllowsCPUOnlyLlamaServerLoadWithLoadedRunner(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 500*time.Millisecond)
	defer done()

	var buf syncBuffer
	previous := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug})))
	defer slog.SetDefault(previous)

	s := InitScheduler(ctx)
	s.waitForRecovery = 10 * time.Millisecond
	s.getGpuFn = func(ctx context.Context, runners []ml.FilteredRunnerDiscovery) []ml.DeviceInfo {
		return nil
	}
	s.getSystemInfoFn = getSystemInfoFn

	s.loadedMu.Lock()
	s.loaded["existing-model"] = &runnerRef{
		llama:    &mockLlm{modelPath: "/fake/existing"},
		modelKey: "existing-model",
	}
	s.loadedMu.Unlock()

	scenario := newScenarioRequest(t, ctx, "cpu-only-model", 1*format.GigaByte, nil, nil)
	scenario.srv.totalSize = 2 * format.GigaByte
	scenario.srv.gpuLayers = 0
	scenario.srv.totalLayers = 33
	s.newServerFn = scenario.newServer

	systemInfo := getSystemInfoFn()
	gpus := s.getGpuFn(ctx, nil)

	require.False(t, s.load(scenario.req, systemInfo, gpus, true))
	require.False(t, scenario.req.loadRetryAttempted)
	require.False(t, scenario.srv.closeCalled)
	require.NotContains(t, buf.String(), "llama-server spilled layers to CPU with other models resident")
}

func TestSchedAllowsLlamaServerCPUOverflowWithoutLoadedRunner(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 500*time.Millisecond)
	defer done()

	var buf syncBuffer
	previous := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug})))
	defer slog.SetDefault(previous)

	s := InitScheduler(ctx)
	s.waitForRecovery = 10 * time.Millisecond
	s.getGpuFn = func(ctx context.Context, runners []ml.FilteredRunnerDiscovery) []ml.DeviceInfo {
		g := ml.DeviceInfo{DeviceID: ml.DeviceID{Library: "Metal"}}
		g.TotalMemory = 24 * format.GigaByte
		g.FreeMemory = 20 * format.GigaByte
		return []ml.DeviceInfo{g}
	}
	s.getSystemInfoFn = getSystemInfoFn

	scenario := newScenarioRequest(t, ctx, "cpu-spill-model", 1*format.GigaByte, nil, nil)
	scenario.srv.totalSize = 2 * format.GigaByte
	scenario.srv.vramSize = 1 * format.GigaByte
	scenario.srv.gpuLayers = 20
	scenario.srv.totalLayers = 33
	s.newServerFn = scenario.newServer

	systemInfo := getSystemInfoFn()
	gpus := s.getGpuFn(ctx, nil)

	require.False(t, s.load(scenario.req, systemInfo, gpus, false))
	require.False(t, scenario.req.loadRetryAttempted)
	require.False(t, scenario.srv.closeCalled)
	require.NotContains(t, buf.String(), "llama-server spilled layers to CPU with other models resident")
}

func TestSchedLlamaServerPredictionUsesTotalParallelContext(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 500*time.Millisecond)
	defer done()
	t.Setenv("OLLAMA_NUM_PARALLEL", "2")

	s := InitScheduler(ctx)
	s.waitForRecovery = 10 * time.Millisecond
	s.getGpuFn = func(ctx context.Context, runners []ml.FilteredRunnerDiscovery) []ml.DeviceInfo {
		g := ml.DeviceInfo{DeviceID: ml.DeviceID{Library: "Metal"}}
		g.TotalMemory = 24 * format.GigaByte
		g.FreeMemory = 900 * format.MebiByte
		return []ml.DeviceInfo{g}
	}
	s.getSystemInfoFn = getSystemInfoFn

	s.loadedMu.Lock()
	s.loaded["existing-model"] = &runnerRef{
		llama:    &mockLlm{modelPath: "existing"},
		modelKey: "existing-model",
	}
	s.loadedMu.Unlock()

	scenario := newScenarioRequestWithContext(t, ctx, "parallel-context-model", 1*format.GigaByte, nil, nil, 65536)
	scenario.req.opts.NumCtx = 32768

	called := false
	s.newServerFn = func(systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, model string, f *ggml.GGML, adapters []string, projectors []string, opts api.Options, numParallel int, _ llm.LlamaServerConfig) (llm.LlamaServer, error) {
		called = true
		return scenario.srv, nil
	}

	systemInfo := getSystemInfoFn()
	gpus := s.getGpuFn(ctx, nil)

	needEvict := s.load(scenario.req, systemInfo, gpus, true)
	require.True(t, needEvict, "expected eviction when total parallel context exceeds available memory")
	require.False(t, called, "preflight prediction should reject before spawning llama-server")
}

// TestSchedLoadCrashTriggersEvictAllAndRetry verifies that a post-spawn
// Load() OOM while other models are resident signals evict-all-and-retry
// on the first attempt, but fails fast on the second attempt.
func TestSchedLoadCrashTriggersEvictAllAndRetry(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 500*time.Millisecond)
	defer done()
	s := InitScheduler(ctx)
	s.waitForRecovery = 10 * time.Millisecond
	s.getGpuFn = func(ctx context.Context, runners []ml.FilteredRunnerDiscovery) []ml.DeviceInfo {
		g := ml.DeviceInfo{DeviceID: ml.DeviceID{Library: "Metal"}}
		g.TotalMemory = 24 * format.GigaByte
		g.FreeMemory = 20 * format.GigaByte
		return []ml.DeviceInfo{g}
	}
	s.getSystemInfoFn = getSystemInfoFn

	// Pre-load a different model so evict-all has something to evict.
	s.loadedMu.Lock()
	s.loaded["existing-model"] = &runnerRef{
		llama:    &mockLlm{modelPath: "/fake/existing"},
		modelKey: "existing-model",
	}
	s.loadedMu.Unlock()

	// newServerFn returns a mockLlm that crashes in Load()
	loadCrash := errors.New("cudaMalloc failed: out of memory")
	s.newServerFn = func(systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, model string, f *ggml.GGML, adapters []string, projectors []string, opts api.Options, numParallel int, _ llm.LlamaServerConfig) (llm.LlamaServer, error) {
		return &mockLlm{modelPath: model, loadErr: loadCrash}, nil
	}

	scenario := newScenarioRequest(t, ctx, "crashing-model", 1*format.GigaByte, nil, nil)
	systemInfo := getSystemInfoFn()
	gpus := s.getGpuFn(ctx, nil)

	// First attempt: should signal evict-all by returning true and NOT send
	// the error to errCh (so the caller will retry).
	needEvict := s.load(scenario.req, systemInfo, gpus, true)
	require.True(t, needEvict, "first load retry should signal eviction")
	require.True(t, scenario.req.loadRetryAttempted, "loadRetryAttempted should be set")
	select {
	case err := <-scenario.req.errCh:
		t.Fatalf("errCh should be empty on first crash, got %v", err)
	default:
	}

	// Second attempt (simulating the retry after processPending evicted all
	// other runners): same crash, but this time loadRetryAttempted is set so
	// load() should fail fast and report the error.
	needEvict = s.load(scenario.req, systemInfo, gpus, true)
	require.False(t, needEvict, "second load retry should not ask for another eviction")
	select {
	case err := <-scenario.req.errCh:
		require.ErrorIs(t, err, loadCrash)
	default:
		t.Fatal("expected error on errCh after second crash")
	}
}

func TestSchedLoadOOMReducesAutomaticContextBeforeRetry(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 500*time.Millisecond)
	defer done()
	s := InitScheduler(ctx)
	s.waitForRecovery = 10 * time.Millisecond
	s.getGpuFn = func(ctx context.Context, runners []ml.FilteredRunnerDiscovery) []ml.DeviceInfo {
		g := ml.DeviceInfo{DeviceID: ml.DeviceID{Library: "Metal"}}
		g.TotalMemory = 24 * format.GigaByte
		g.FreeMemory = 20 * format.GigaByte
		return []ml.DeviceInfo{g}
	}
	s.getSystemInfoFn = getSystemInfoFn

	s.loadedMu.Lock()
	s.loaded["existing-model"] = &runnerRef{
		llama:    &mockLlm{modelPath: "/fake/existing"},
		modelKey: "existing-model",
	}
	s.loadedMu.Unlock()

	loadCrash := errors.New("cudaMalloc failed: out of memory")
	var seenNumCtx []int
	var seenNumBatch []int
	s.newServerFn = func(systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, model string, f *ggml.GGML, adapters []string, projectors []string, opts api.Options, numParallel int, _ llm.LlamaServerConfig) (llm.LlamaServer, error) {
		seenNumCtx = append(seenNumCtx, opts.NumCtx)
		seenNumBatch = append(seenNumBatch, opts.NumBatch)
		return &mockLlm{modelPath: model, loadErr: loadCrash}, nil
	}

	scenario := newScenarioRequestWithContext(t, ctx, "crashing-model", 1*format.GigaByte, nil, nil, 131072)
	scenario.req.opts.NumCtx = 262144
	scenario.req.numCtxAuto = true
	scenario.req.numBatchAuto = true
	systemInfo := getSystemInfoFn()
	gpus := s.getGpuFn(ctx, nil)

	needEvict := s.load(scenario.req, systemInfo, gpus, true)
	require.True(t, needEvict, "first automatic-context load retry should signal eviction and retry")
	require.True(t, scenario.req.loadRetryAttempted)
	require.Equal(t, 32768, scenario.req.opts.NumCtx)
	require.Equal(t, llamaServerGenerationBatchMedium, scenario.req.opts.NumBatch)
	select {
	case err := <-scenario.req.errCh:
		t.Fatalf("errCh should be empty on first crash, got %v", err)
	default:
	}

	needEvict = s.load(scenario.req, systemInfo, gpus, true)
	require.False(t, needEvict, "second load retry should not ask for another eviction")
	require.Equal(t, []int{262144, 32768}, seenNumCtx)
	require.Equal(t, []int{llamaServerGenerationBatchLarge, llamaServerGenerationBatchMedium}, seenNumBatch)
	select {
	case err := <-scenario.req.errCh:
		require.ErrorIs(t, err, loadCrash)
	default:
		t.Fatal("expected error on errCh after second crash")
	}
}

func TestSchedLoadOOMKeepsExplicitContextBeforeRetry(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 500*time.Millisecond)
	defer done()
	s := InitScheduler(ctx)
	s.waitForRecovery = 10 * time.Millisecond
	s.getGpuFn = func(ctx context.Context, runners []ml.FilteredRunnerDiscovery) []ml.DeviceInfo {
		g := ml.DeviceInfo{DeviceID: ml.DeviceID{Library: "Metal"}}
		g.TotalMemory = 24 * format.GigaByte
		g.FreeMemory = 20 * format.GigaByte
		return []ml.DeviceInfo{g}
	}
	s.getSystemInfoFn = getSystemInfoFn

	s.loadedMu.Lock()
	s.loaded["existing-model"] = &runnerRef{
		llama:    &mockLlm{modelPath: "/fake/existing"},
		modelKey: "existing-model",
	}
	s.loadedMu.Unlock()

	loadCrash := errors.New("cudaMalloc failed: out of memory")
	s.newServerFn = func(systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, model string, f *ggml.GGML, adapters []string, projectors []string, opts api.Options, numParallel int, _ llm.LlamaServerConfig) (llm.LlamaServer, error) {
		return &mockLlm{modelPath: model, loadErr: loadCrash}, nil
	}

	scenario := newScenarioRequestWithContext(t, ctx, "crashing-model", 1*format.GigaByte, nil, nil, 131072)
	scenario.req.opts.NumCtx = 262144
	scenario.req.numCtxAuto = false
	systemInfo := getSystemInfoFn()
	gpus := s.getGpuFn(ctx, nil)

	needEvict := s.load(scenario.req, systemInfo, gpus, true)
	require.True(t, needEvict, "explicit-context load retry should still evict and retry once")
	require.True(t, scenario.req.loadRetryAttempted)
	require.Equal(t, 262144, scenario.req.opts.NumCtx)
	select {
	case err := <-scenario.req.errCh:
		t.Fatalf("errCh should be empty on first crash, got %v", err)
	default:
	}
}

func TestSchedFirstLoadOOMReducesAutomaticContextAndRetries(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), time.Second)
	defer done()
	s := InitScheduler(ctx)
	s.waitForRecovery = 10 * time.Millisecond
	s.getGpuFn = func(ctx context.Context, runners []ml.FilteredRunnerDiscovery) []ml.DeviceInfo {
		g := ml.DeviceInfo{DeviceID: ml.DeviceID{Library: "Metal"}}
		g.TotalMemory = 24 * format.GigaByte
		g.FreeMemory = 20 * format.GigaByte
		return []ml.DeviceInfo{g}
	}
	s.getSystemInfoFn = getSystemInfoFn

	loadCrash := errors.New("cudaMalloc failed: out of memory")
	var seenNumCtx []int
	s.newServerFn = func(systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, model string, f *ggml.GGML, adapters []string, projectors []string, opts api.Options, numParallel int, _ llm.LlamaServerConfig) (llm.LlamaServer, error) {
		seenNumCtx = append(seenNumCtx, opts.NumCtx)
		if len(seenNumCtx) == 1 {
			return &mockLlm{modelPath: model, loadErr: loadCrash}, nil
		}
		return &mockLlm{modelPath: model, vramSize: 1 * format.GigaByte, contextLength: opts.NumCtx}, nil
	}

	scenario := newScenarioRequestWithContext(t, ctx, "first-load-crashing-model", 1*format.GigaByte, nil, nil, 131072)
	scenario.req.opts.NumCtx = 262144
	scenario.req.numCtxAuto = true

	s.pendingReqCh <- scenario.req
	s.Run(ctx)

	select {
	case runner := <-scenario.req.successCh:
		require.Equal(t, 32768, runner.Options.NumCtx)
		require.Equal(t, []int{262144, 32768}, seenNumCtx)
	case err := <-scenario.req.errCh:
		t.Fatalf("expected retry success, got error %v", err)
	case <-ctx.Done():
		t.Fatal("timed out waiting for first-load retry")
	}
}

// TestSchedLoadCrashNoOtherModelsFailsFast verifies that a Load() crash with
// no other resident models reports the error immediately (no retry).
func TestSchedLoadCrashNoOtherModelsFailsFast(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 500*time.Millisecond)
	defer done()
	s := InitScheduler(ctx)
	s.waitForRecovery = 10 * time.Millisecond
	s.getGpuFn = func(ctx context.Context, runners []ml.FilteredRunnerDiscovery) []ml.DeviceInfo {
		g := ml.DeviceInfo{DeviceID: ml.DeviceID{Library: "Metal"}}
		g.TotalMemory = 24 * format.GigaByte
		g.FreeMemory = 20 * format.GigaByte
		return []ml.DeviceInfo{g}
	}
	s.getSystemInfoFn = getSystemInfoFn

	loadCrash := errors.New("simulated llama-server OOM crash")
	s.newServerFn = func(systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, model string, f *ggml.GGML, adapters []string, projectors []string, opts api.Options, numParallel int, _ llm.LlamaServerConfig) (llm.LlamaServer, error) {
		return &mockLlm{modelPath: model, loadErr: loadCrash}, nil
	}

	scenario := newScenarioRequest(t, ctx, "crashing-model", 1*format.GigaByte, nil, nil)
	systemInfo := getSystemInfoFn()
	gpus := s.getGpuFn(ctx, nil)

	needEvict := s.load(scenario.req, systemInfo, gpus, true)
	require.False(t, needEvict, "crash with no other runners should not ask for eviction")
	require.False(t, scenario.req.loadRetryAttempted, "loadRetryAttempted must stay false")
	select {
	case err := <-scenario.req.errCh:
		require.ErrorIs(t, err, loadCrash)
	default:
		t.Fatal("expected error on errCh immediately")
	}
}

func TestSchedLoadNonOOMWithOtherModelsFailsFast(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 500*time.Millisecond)
	defer done()
	s := InitScheduler(ctx)
	s.waitForRecovery = 10 * time.Millisecond
	s.getGpuFn = func(ctx context.Context, runners []ml.FilteredRunnerDiscovery) []ml.DeviceInfo {
		g := ml.DeviceInfo{DeviceID: ml.DeviceID{Library: "Metal"}}
		g.TotalMemory = 24 * format.GigaByte
		g.FreeMemory = 20 * format.GigaByte
		return []ml.DeviceInfo{g}
	}
	s.getSystemInfoFn = getSystemInfoFn

	s.loadedMu.Lock()
	s.loaded["existing-model"] = &runnerRef{
		llama:    &mockLlm{modelPath: "/fake/existing"},
		modelKey: "existing-model",
	}
	s.loadedMu.Unlock()

	loadCrash := errors.New("server parse failed")
	s.newServerFn = func(systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, model string, f *ggml.GGML, adapters []string, projectors []string, opts api.Options, numParallel int, _ llm.LlamaServerConfig) (llm.LlamaServer, error) {
		return &mockLlm{modelPath: model, loadErr: loadCrash}, nil
	}

	scenario := newScenarioRequest(t, ctx, "crashing-model", 1*format.GigaByte, nil, nil)
	systemInfo := getSystemInfoFn()
	gpus := s.getGpuFn(ctx, nil)

	needEvict := s.load(scenario.req, systemInfo, gpus, true)
	require.False(t, needEvict, "non-OOM load crash should not ask for eviction")
	require.False(t, scenario.req.loadRetryAttempted, "loadRetryAttempted must stay false")
	select {
	case err := <-scenario.req.errCh:
		require.ErrorIs(t, err, loadCrash)
	default:
		t.Fatal("expected error on errCh immediately")
	}
}

func TestSchedRuntimeOOMExpiresLoadedRunners(t *testing.T) {
	ctx, done := context.WithCancel(t.Context())
	defer done()
	s := InitScheduler(ctx)

	currentModel := &Model{ModelPath: "/tmp/current.gguf"}
	current := &runnerRef{
		model:           currentModel,
		modelKey:        schedulerModelKey(currentModel),
		sessionDuration: time.Hour,
		llama:           &mockLlm{modelPath: "/tmp/current.gguf"},
	}
	otherModel := &Model{ModelPath: "/tmp/other.gguf"}
	other := &runnerRef{
		model:           otherModel,
		modelKey:        schedulerModelKey(otherModel),
		sessionDuration: time.Hour,
		llama:           &mockLlm{modelPath: "/tmp/other.gguf"},
	}

	s.loadedMu.Lock()
	s.loaded[current.modelKey] = current
	s.loaded[other.modelKey] = other
	s.loadedMu.Unlock()

	s.expireRunnersForRuntimeOOM(currentModel, errors.New("cudaMalloc failed: out of memory"))

	require.Equal(t, time.Duration(0), current.sessionDuration)
	require.Equal(t, time.Duration(0), other.sessionDuration)
	require.Len(t, s.expiredCh, 2)
}

func TestSchedLlamaServerEvictsExistingOnPending(t *testing.T) {
	// When a llama-server runner is already loaded and a new model is requested,
	// the scheduler should evict the llama-server runner
	ctx, done := context.WithCancel(t.Context())
	defer done()
	s := InitScheduler(ctx)

	// Load a llama-server runner
	s.loadedMu.Lock()
	s.loaded["llama-model"] = &runnerRef{
		llama:    &mockLlm{modelPath: "/tmp/model.gguf"},
		modelKey: "llama-model",
	}
	s.loadedMu.Unlock()

	require.True(t, hasLoadedRunner(s))

	// The findRunnerToUnload should find and return the llama-server runner
	runner := s.findRunnerToUnload()
	require.NotNil(t, runner)
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
	contextLength     int
	vramByGPU         map[ml.DeviceID]uint64
	memorySizeFn      func() (uint64, uint64)
	vramByGPUFn       func(ml.DeviceID) uint64
	gpuLayers         uint64
	totalLayers       uint64
	gpuLayerOverflow  int
	port              int
	deviceInfos       []ml.DeviceInfo
	exited            bool

	// loadErr, if non-nil, is returned from Load() to simulate a post-spawn
	// load failure (e.g. llama-server crashing due to under-predicted VRAM).
	loadErr error
}

func (s *mockLlm) ModelPath() string {
	return s.modelPath
}

func (s *mockLlm) Load(ctx context.Context, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, requireFull bool) ([]ml.DeviceID, error) {
	if s.loadErr != nil {
		return nil, s.loadErr
	}
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

func (s *mockLlm) Chat(ctx context.Context, req llm.ChatRequest, fn func(llm.ChatResponse)) error {
	return errors.New("not implemented")
}

func (s *mockLlm) ApplyChatTemplate(ctx context.Context, req llm.ChatRequest) (string, error) {
	return "", errors.New("not implemented")
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

func (s *mockLlm) MemorySize() (uint64, uint64) {
	if s.memorySizeFn != nil {
		return s.memorySizeFn()
	}
	return s.totalSize, s.vramSize
}

func (s *mockLlm) VRAMByGPU(id ml.DeviceID) uint64 {
	if s.vramByGPUFn != nil {
		return s.vramByGPUFn(id)
	}
	return s.vramByGPU[id]
}
func (s *mockLlm) Pid() int { return -1 }
func (s *mockLlm) GetPort() int {
	if s.port == 0 {
		return -1
	}
	return s.port
}
func (s *mockLlm) GetDeviceInfos(ctx context.Context) []ml.DeviceInfo {
	return s.deviceInfos
}
func (s *mockLlm) HasExited() bool                   { return s.exited }
func (s *mockLlm) GetActiveDeviceIDs() []ml.DeviceID { return nil }
func (s *mockLlm) ContextLength() int                { return s.contextLength }
func (s *mockLlm) LayerOffloadStatus() (gpuLayers, totalLayers uint64, overflow int, ok bool) {
	return s.gpuLayers, s.totalLayers, s.gpuLayerOverflow, s.totalLayers > 0
}

func TestRunnerCanBeEvicted(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 500*time.Millisecond)
	defer done()

	s := InitScheduler(ctx)
	s.getGpuFn = getGpuFn
	s.getSystemInfoFn = getSystemInfoFn

	loadedRunner := &runnerRef{
		model:           &Model{Name: "test", ModelPath: "/fake/model"},
		modelPath:       "/fake/model",
		llama:           &mockLlm{vramSize: 21 * format.GigaByte, vramByGPU: map[ml.DeviceID]uint64{}},
		sessionDuration: 5 * time.Millisecond,
		refCount:        0, // idle
	}

	s.loadedMu.Lock()
	s.loaded["/fake/model"] = loadedRunner
	s.loadedMu.Unlock()

	s.loadedMu.Lock()
	require.Len(t, s.loaded, 1)
	s.loadedMu.Unlock()

	runner := s.findRunnerToUnload()
	require.NotNil(t, runner)
	require.Equal(t, "/fake/model", runner.modelPath)
}

func TestSchedulerTracksMultipleLoadedRunners(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 500*time.Millisecond)
	defer done()

	s := InitScheduler(ctx)
	s.getGpuFn = getGpuFn
	s.getSystemInfoFn = getSystemInfoFn

	firstRunner := &runnerRef{
		model:           &Model{Name: "first", ModelPath: "/fake/first/model"},
		modelPath:       "/fake/first/model",
		llama:           &mockLlm{vramSize: 8 * format.GigaByte, vramByGPU: map[ml.DeviceID]uint64{{Library: "Metal"}: 8 * format.GigaByte}},
		sessionDuration: 10 * time.Millisecond,
		numParallel:     1,
		refCount:        0,
	}

	secondRunner := &runnerRef{
		model:           &Model{Name: "second", ModelPath: "/fake/second/model"},
		modelPath:       "/fake/second/model",
		llama:           &mockLlm{vramSize: 4 * format.GigaByte, vramByGPU: map[ml.DeviceID]uint64{{Library: "Metal"}: 4 * format.GigaByte}},
		sessionDuration: 10 * time.Millisecond,
		numParallel:     1,
		refCount:        0,
	}

	s.loadedMu.Lock()
	s.loaded["/fake/first/model"] = firstRunner
	s.loaded["/fake/second/model"] = secondRunner
	s.loadedMu.Unlock()

	s.loadedMu.Lock()
	require.Len(t, s.loaded, 2)
	require.NotNil(t, s.loaded["/fake/first/model"])
	require.NotNil(t, s.loaded["/fake/second/model"])
	s.loadedMu.Unlock()

	gpus := []ml.DeviceInfo{
		{
			DeviceID:    ml.DeviceID{Library: "Metal"},
			TotalMemory: 24 * format.GigaByte,
			FreeMemory:  24 * format.GigaByte,
		},
	}
	s.updateFreeSpace(gpus)

	expectedFree := uint64(24*format.GigaByte) - uint64(8*format.GigaByte) - uint64(4*format.GigaByte)
	require.Equal(t, expectedFree, gpus[0].FreeMemory)
}
