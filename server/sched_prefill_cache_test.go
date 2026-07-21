package server

import (
	"context"
	"os"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
)

type mockPrefillCacheRunner struct {
	savePath    string
	restorePath string
	saveErr     error
	restoreErr  error
}

func (m *mockPrefillCacheRunner) ModelPath() string { return "" }
func (m *mockPrefillCacheRunner) Load(context.Context, ml.SystemInfo, []ml.DeviceInfo, bool) ([]ml.DeviceID, error) {
	return nil, nil
}
func (m *mockPrefillCacheRunner) Ping(context.Context) error             { return nil }
func (m *mockPrefillCacheRunner) WaitUntilRunning(context.Context) error { return nil }
func (m *mockPrefillCacheRunner) Completion(context.Context, llm.CompletionRequest, func(llm.CompletionResponse)) error {
	return nil
}
func (m *mockPrefillCacheRunner) Chat(context.Context, llm.ChatRequest, func(llm.ChatResponse)) error {
	return nil
}
func (m *mockPrefillCacheRunner) ApplyChatTemplate(context.Context, llm.ChatRequest) (string, error) {
	return "", nil
}
func (m *mockPrefillCacheRunner) Embedding(context.Context, string) ([]float32, int, error) {
	return nil, 0, nil
}
func (m *mockPrefillCacheRunner) Tokenize(context.Context, string) ([]int, error) {
	return nil, nil
}
func (m *mockPrefillCacheRunner) Detokenize(context.Context, []int) (string, error) {
	return "", nil
}
func (m *mockPrefillCacheRunner) Close() error                 { return nil }
func (m *mockPrefillCacheRunner) MemorySize() (uint64, uint64) { return 0, 0 }
func (m *mockPrefillCacheRunner) VRAMByGPU(ml.DeviceID) uint64 { return 0 }
func (m *mockPrefillCacheRunner) Pid() int                     { return 1 }
func (m *mockPrefillCacheRunner) GetPort() int                 { return 0 }
func (m *mockPrefillCacheRunner) GetDeviceInfos(context.Context) []ml.DeviceInfo {
	return nil
}
func (m *mockPrefillCacheRunner) HasExited() bool    { return false }
func (m *mockPrefillCacheRunner) ContextLength() int { return 0 }

func (m *mockPrefillCacheRunner) SavePrefillCache(_ context.Context, path string) error {
	m.savePath = path
	return m.saveErr
}

func (m *mockPrefillCacheRunner) RestorePrefillCache(_ context.Context, path string) error {
	m.restorePath = path
	return m.restoreErr
}

func TestSchedulerPrefillCacheSaveRestore(t *testing.T) {
	dir := t.TempDir()
	store := &prefillCacheStore{dir: dir, entries: map[string]*prefillCacheEntry{}}
	sched := &Scheduler{prefillCache: store}

	mock := &mockPrefillCacheRunner{}
	runner := &runnerRef{
		modelKey:    "model-a",
		numParallel: 1,
		model:       &Model{ModelPath: "model-a"},
		Options:     &api.Options{Runner: api.Runner{NumCtx: 4096}},
		llama:       mock,
	}

	sched.savePrefillCache(runner)
	if mock.savePath == "" {
		t.Fatal("expected save to be called")
	}

	path, ok := store.pathFor(runner)
	if !ok {
		t.Fatal("expected path")
	}
	if err := os.WriteFile(path, []byte("kv"), 0o600); err != nil {
		t.Fatal(err)
	}
	store.record(runner, path)

	mock.restorePath = ""
	sched.restorePrefillCache(runner, mock)
	if mock.restorePath != path {
		t.Fatalf("restore path = %q, want %q", mock.restorePath, path)
	}
}
