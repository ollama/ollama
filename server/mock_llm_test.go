package server

import (
	"context"

	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
	"github.com/stretchr/testify/mock"
)

type mockLlm struct {
	mock.Mock
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

func (m *mockLlm) Completion(ctx context.Context, req llm.CompletionRequest, fn func(llm.CompletionResponse)) error {
	args := m.Called(ctx, req, fn)
	return args.Error(0)
}

func (m *mockLlm) Embedding(ctx context.Context, prompt string) ([]float32, error) {
	args := m.Called(ctx, prompt)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).([]float32), args.Error(1)
}

func (m *mockLlm) ImageEmbedding(ctx context.Context, image llm.ImageData) ([]float32, error) {
	args := m.Called(ctx, image)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).([]float32), args.Error(1)
}

func (m *mockLlm) Tokenize(ctx context.Context, prompt string) ([]int, error) {
	args := m.Called(ctx, prompt)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).([]int), args.Error(1)
}

func (m *mockLlm) Detokenize(ctx context.Context, tokens []int) (string, error) {
	args := m.Called(ctx, tokens)
	return args.String(0), args.Error(1)
}

func (m *mockLlm) Close() error {
	m.closeCalled = true
	return m.closeResp
}

func (m *mockLlm) ModelPath() string {
	return m.modelPath
}

func (m *mockLlm) Load(ctx context.Context, gpus discover.GpuInfoList, requireFull bool) ([]ml.DeviceID, error) {
	if requireFull {
		for _, g := range gpus {
			if g.FreeMemory >= m.vramSize {
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
func (m *mockLlm) Ping(ctx context.Context) error             { return m.pingResp }
func (m *mockLlm) WaitUntilRunning(ctx context.Context) error { return m.waitResp }
func (m *mockLlm) VRAMSize() uint64                           { return m.vramSize }
func (m *mockLlm) TotalSize() uint64                          { return m.totalSize }
func (m *mockLlm) VRAMByGPU(id ml.DeviceID) uint64            { return m.vramByGPU[id] }
func (m *mockLlm) Pid() int                                   { return -1 }
func (m *mockLlm) GetPort() int                               { return -1 }
func (m *mockLlm) GetDeviceInfos(ctx context.Context) []ml.DeviceInfo { return nil }
func (m *mockLlm) HasExited() bool                            { return false }
func (m *mockLlm) GetActiveDeviceIDs() []ml.DeviceID          { return nil }
