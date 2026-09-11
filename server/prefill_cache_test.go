package server

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
)

func mustMkdir(t *testing.T, path string) {
	t.Helper()
	if err := os.Mkdir(path, 0o700); err != nil {
		t.Fatalf("mkdir %s: %v", path, err)
	}
}

func mustWriteFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

func mustDirExist(t *testing.T, path string) {
	t.Helper()
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat %s: %v", path, err)
	}
	if !info.IsDir() {
		t.Fatalf("%s is not a directory", path)
	}
}

func mustNotExist(t *testing.T, path string) {
	t.Helper()
	if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("stat %s: got %v, want %v", path, err, os.ErrNotExist)
	}
}

type prefillCacheMock struct {
	*mockLlm
	calls        []string
	restoreErr   error
	saveErr      error
	exited       bool
	saveStarted  chan struct{}
	saveContinue chan struct{}
}

func (m *prefillCacheMock) HasExited() bool {
	return m.exited
}

func (m *prefillCacheMock) RestorePrefillCache(context.Context) error {
	m.calls = append(m.calls, "restore")
	return m.restoreErr
}

func (m *prefillCacheMock) SavePrefillCache(ctx context.Context) error {
	m.calls = append(m.calls, "save")
	if m.saveStarted != nil {
		close(m.saveStarted)
		select {
		case <-m.saveContinue:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return m.saveErr
}

func (m *prefillCacheMock) Close() error {
	m.calls = append(m.calls, "close")
	return m.mockLlm.Close()
}

func TestInitSchedulerPrefillCacheGate(t *testing.T) {
	temp := t.TempDir()
	t.Setenv("TMPDIR", temp)
	t.Setenv("TMP", temp)
	t.Setenv("TEMP", temp)

	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()

	if got := InitScheduler(ctx).prefillCacheRoot; got != "" {
		t.Fatalf("prefillCacheRoot = %q, want empty: persistence must be opt-in", got)
	}

	t.Setenv("OLLAMA_PREFILL_CACHE", "1")
	root := InitScheduler(ctx).prefillCacheRoot
	if !strings.HasPrefix(root, temp) {
		t.Fatalf("prefillCacheRoot = %q, want a directory under %q", root, temp)
	}
	mustDirExist(t, root)

	// The cache does not outlive the daemon, and nothing else reclaims it.
	cancel()
	for deadline := time.Now().Add(5 * time.Second); ; {
		if _, err := os.Stat(root); errors.Is(err, os.ErrNotExist) {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("%s still exists after the scheduler context was canceled", root)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func TestPrefillCachePathIdentity(t *testing.T) {
	s := &Scheduler{prefillCacheRoot: t.TempDir()}
	base := func() *LlmRequest {
		return &LlmRequest{
			model: &Model{ModelPath: "/models/a.gguf"},
			opts:  api.Options{Runner: api.Runner{NumCtx: 8192}},
		}
	}
	llama := func(req *LlmRequest) string {
		return s.llamaPrefillCachePath(req, req.opts.Runner, 1)
	}

	want := llama(base())
	if want == "" {
		t.Fatal("llamaPrefillCachePath() = empty, want a path")
	}
	if got := llama(base()); got != want {
		t.Fatalf("llamaPrefillCachePath() = %q, want %q for the same inputs", got, want)
	}

	// Anything that changes what the runner writes into a slot has to change
	// the directory those slot files land in, or a snapshot is restored into a
	// runner it does not fit.
	for name, mutate := range map[string]func(*LlmRequest){
		"model":         func(r *LlmRequest) { r.model.ModelPath = "/models/b.gguf" },
		"draft":         func(r *LlmRequest) { r.model.DraftPath = "/models/draft.gguf" },
		"adapters":      func(r *LlmRequest) { r.model.AdapterPaths = []string{"/lora/a"} },
		"projectors":    func(r *LlmRequest) { r.model.ProjectorPaths = []string{"/mmproj/a"} },
		"options":       func(r *LlmRequest) { r.opts.NumCtx = 4096 },
		"context shift": func(r *LlmRequest) { r.contextShift = true },
	} {
		t.Run(name, func(t *testing.T) {
			req := base()
			mutate(req)
			if got := llama(req); got == want {
				t.Fatalf("llamaPrefillCachePath() is unchanged after changing %s", name)
			}
		})
	}

	t.Run("kv cache type", func(t *testing.T) {
		t.Setenv("OLLAMA_KV_CACHE_TYPE", "q8_0")
		if got := llama(base()); got == want {
			t.Fatal("llamaPrefillCachePath() is unchanged after changing the kv cache type")
		}
	})

	t.Run("num parallel", func(t *testing.T) {
		req := base()
		if got := s.llamaPrefillCachePath(req, req.opts.Runner, 2); got == want {
			t.Fatal("llamaPrefillCachePath() is unchanged after changing num parallel")
		}
	})

	t.Run("persistence off", func(t *testing.T) {
		req := base()
		if got := (&Scheduler{}).llamaPrefillCachePath(req, req.opts.Runner, 1); got != "" {
			t.Fatalf("llamaPrefillCachePath() = %q with persistence off, want empty", got)
		}
	})
}

func TestPrunePrefillCache(t *testing.T) {
	root := t.TempDir()
	inUse := filepath.Join(root, "in-use")
	old := filepath.Join(root, "old")
	recent := filepath.Join(root, "recent")
	for _, dir := range []string{inUse, old, recent} {
		mustMkdir(t, dir)
		mustWriteFile(t, filepath.Join(dir, "cache"), "1234")
	}
	past := time.Now().Add(-time.Hour)
	for _, dir := range []string{inUse, old} {
		if err := os.Chtimes(dir, past, past); err != nil {
			t.Fatalf("chtimes %s: %v", dir, err)
		}
	}

	s := &Scheduler{loaded: map[string]*runnerRef{"in-use": {prefillCacheDir: inUse}}}
	prunePrefillCache(root, 8, s.prefillCacheDirsInUse())

	// Oldest first, but never a directory a loaded runner is still writing
	// into, even when it is the oldest and the cache is over the limit.
	mustDirExist(t, inUse)
	mustNotExist(t, old)
	mustDirExist(t, recent)
}

func TestRunnerUnloadDoesNotSavePrefillCache(t *testing.T) {
	// The expired path saves explicitly before waiting for VRAM recovery.
	// unload must not save on its own: it is also reached by a failed load and
	// by a leaked duplicate runner, neither of which has state worth keeping.
	persist := &prefillCacheMock{mockLlm: &mockLlm{}}
	runner := &runnerRef{llama: persist}
	runner.refMu.Lock()
	defer runner.refMu.Unlock()

	runner.unload()
	if want := []string{"close"}; !slices.Equal(persist.calls, want) {
		t.Fatalf("runner calls = %v, want %v", persist.calls, want)
	}
}

func TestRunnerSavePrefillCacheSkipsLoadingRunner(t *testing.T) {
	persist := &prefillCacheMock{mockLlm: &mockLlm{}}
	runner := &runnerRef{llama: persist, loading: true}
	runner.refMu.Lock()
	defer runner.refMu.Unlock()

	runner.savePrefillCache(t.Context())
	if len(persist.calls) != 0 {
		t.Fatalf("runner calls = %v, want none: a loading runner has no prefix to save", persist.calls)
	}
}

func TestRunnerSavePrefillCacheSkipsExitedRunner(t *testing.T) {
	// The endpoint the save would talk to is gone, so asking only spends the
	// persist timeout on the unload path before failing.
	persist := &prefillCacheMock{mockLlm: &mockLlm{}, exited: true}
	runner := &runnerRef{llama: persist}
	runner.refMu.Lock()
	defer runner.refMu.Unlock()

	runner.savePrefillCache(t.Context())
	if len(persist.calls) != 0 {
		t.Fatalf("runner calls = %v, want none: an exited runner cannot be asked to save", persist.calls)
	}
}

func TestRunnerRestorePrefillCacheMarksEntryRecentlyUsed(t *testing.T) {
	root := t.TempDir()
	restored := filepath.Join(root, "restored")
	other := filepath.Join(root, "other")
	for dir, age := range map[string]time.Duration{restored: time.Hour, other: time.Minute} {
		mustMkdir(t, dir)
		mustWriteFile(t, filepath.Join(dir, "cache"), "1234")
		at := time.Now().Add(-age)
		if err := os.Chtimes(dir, at, at); err != nil {
			t.Fatalf("chtimes %s: %v", dir, err)
		}
	}

	runner := &runnerRef{llama: &prefillCacheMock{mockLlm: &mockLlm{}}, prefillCacheDir: restored}
	runner.refMu.Lock()
	runner.restorePrefillCache(t.Context())
	runner.refMu.Unlock()

	// Reading an entry makes it the most recently used, so the next unload
	// that prunes evicts something else even though this one is older on disk.
	prunePrefillCache(root, 4, nil)
	mustDirExist(t, restored)
	mustNotExist(t, other)
}

func TestSchedSavesPrefillCacheWithoutHoldingLoadedLock(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()

	persist := &prefillCacheMock{
		mockLlm:      &mockLlm{},
		saveStarted:  make(chan struct{}),
		saveContinue: make(chan struct{}),
	}
	runner := &runnerRef{llama: persist, modelKey: "model"}
	s := InitScheduler(ctx)
	s.loaded[runner.modelKey] = runner
	go s.processCompleted(ctx)
	s.expiredCh <- runner

	select {
	case <-persist.saveStarted:
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for prefill cache save")
	}

	// Saving talks to the runner over HTTP. Holding the loaded lock across it
	// would stall ps, stop, and every other model waiting to be scheduled.
	locked := make(chan struct{})
	go func() {
		s.loadedMu.Lock()
		_ = len(s.loaded)
		s.loadedMu.Unlock()
		close(locked)
	}()
	select {
	case <-locked:
	case <-time.After(time.Second):
		t.Fatal("loadedMu was held for the duration of the prefill cache save")
	}

	close(persist.saveContinue)
	select {
	case <-s.unloadedCh:
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for runner unload")
	}
	if want := []string{"save", "close"}; !slices.Equal(persist.calls, want) {
		t.Fatalf("runner calls = %v, want %v", persist.calls, want)
	}
}

func TestSchedPrefillCachePathSurvivesActiveLoadingRetry(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), time.Second)
	defer cancel()

	s := InitScheduler(ctx)
	s.prefillCacheRoot = t.TempDir()
	scenario := newScenarioRequest(t, ctx, "prefill-retry-test", 20, nil, map[ml.DeviceID]uint64{})
	persist := &prefillCacheMock{mockLlm: scenario.srv}
	var configuredPath string
	s.newServerFn = func(_ ml.SystemInfo, _ []ml.DeviceInfo, model string, _ *ggml.GGML, _, _ []string, _ api.Options, _ int, config llm.LlamaServerConfig) (llm.LlamaServer, error) {
		configuredPath = config.PrefillCachePath
		persist.modelPath = model
		return persist, nil
	}

	gpu := ml.DeviceInfo{DeviceID: ml.DeviceID{Library: "Metal"}, TotalMemory: 30, FreeMemory: 10}
	if needEvict := s.load(scenario.req, ml.SystemInfo{}, []ml.DeviceInfo{gpu}, true); !needEvict {
		t.Fatal("first load did not request eviction")
	}
	if configuredPath == "" {
		t.Fatal("configured prefill cache path is empty")
	}
	if got := scenario.req.prefillCachePath; got != configuredPath {
		t.Fatalf("active loading prefill cache path = %q, want %q", got, configuredPath)
	}

	gpu.FreeMemory = 30
	if needEvict := s.load(scenario.req, ml.SystemInfo{}, []ml.DeviceInfo{gpu}, true); needEvict {
		t.Fatal("second load unexpectedly requested eviction")
	}
	select {
	case err := <-scenario.req.errCh:
		t.Fatalf("load returned %v, want a runner", err)
	case runner := <-scenario.req.successCh:
		if got := runner.prefillCacheDir; got != configuredPath {
			t.Fatalf("runner prefill cache path = %q, want %q", got, configuredPath)
		}
	case <-ctx.Done():
		t.Fatal("timed out waiting for retried runner")
	}
}

func TestSchedRestoresPrefillCacheBeforeReturningRunner(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), time.Second)
	defer cancel()
	s := InitScheduler(ctx)
	scenario := newScenarioRequest(t, ctx, "prefill-test", 10, nil, map[ml.DeviceID]uint64{})
	// A restore failure still has to produce a runner: the fallback is a cold
	// prefill, not a failed load.
	persist := &prefillCacheMock{mockLlm: scenario.srv, restoreErr: errors.New("corrupt snapshot")}
	s.newServerFn = func(_ ml.SystemInfo, _ []ml.DeviceInfo, model string, _ *ggml.GGML, _, _ []string, _ api.Options, _ int, _ llm.LlamaServerConfig) (llm.LlamaServer, error) {
		persist.modelPath = model
		return persist, nil
	}

	s.load(scenario.req, ml.SystemInfo{}, nil, false)
	var runner *runnerRef
	select {
	case err := <-scenario.req.errCh:
		t.Fatalf("load returned %v, want a runner", err)
	case runner = <-scenario.req.successCh:
		if want := []string{"restore"}; !slices.Equal(persist.calls, want) {
			t.Fatalf("runner calls = %v, want %v: restore must finish before the runner is handed out", persist.calls, want)
		}
	case <-ctx.Done():
		t.Fatal("timed out waiting for loaded runner")
	}
	if runner == nil {
		t.Fatal("load returned a nil runner, want a fall back to a cold cache")
	}
}

func TestRunnerSavesPrefillCacheAfterFailedRestore(t *testing.T) {
	// A snapshot that would not load is no reason to stop producing one: the
	// runner still served requests, and its prefix is worth keeping.
	persist := &prefillCacheMock{mockLlm: &mockLlm{}, restoreErr: errors.New("corrupt snapshot")}
	runner := &runnerRef{llama: persist}
	runner.refMu.Lock()
	defer runner.refMu.Unlock()

	runner.restorePrefillCache(t.Context())
	runner.savePrefillCache(t.Context())
	if want := []string{"restore", "save"}; !slices.Equal(persist.calls, want) {
		t.Fatalf("runner calls = %v, want %v", persist.calls, want)
	}
}
