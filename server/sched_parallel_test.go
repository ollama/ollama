package server

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/types/model"
)

func TestSchedParallelCapacity(t *testing.T) {
	for _, family := range []string{"llama", "mllama", "qwen3vl", "qwen3vlmoe", "qwen35", "qwen35moe", "qwen3next", "lfm2", "lfm2moe", "nemotron_h", "nemotron_h_moe", "nemotron_h_omni"} {
		for _, format := range []string{"gguf", "safetensors"} {
			for _, parallel := range []int{1, 2} {
				for _, capability := range []string{"completion", "embedding", "decision", "speech"} {
					t.Run(fmt.Sprintf("%s/%s/%d/%s", family, format, parallel, capability), func(t *testing.T) {
						t.Setenv("OLLAMA_NUM_PARALLEL", fmt.Sprint(parallel))
						ctx, cancel := context.WithTimeout(t.Context(), time.Second)
						defer cancel()
						s := InitScheduler(ctx)
						// Reuse an already-created process to exercise scheduler admission
						// without launching a runner or loading model weights.
						s.activeLoading = &mockLlm{modelPath: "test"}
						req := &LlmRequest{
							ctx: ctx,
							model: &Model{ShortName: "test", Config: model.ConfigV2{
								ModelFormat: format, ModelFamily: family, Capabilities: []string{capability},
							}},
							opts:      api.DefaultOptions(),
							successCh: make(chan *runnerRef, 1),
							errCh:     make(chan error, 1),
						}
						want := parallel
						if format == "gguf" && capability != "completion" {
							want = 1
						}
						if s.load(req, ml.SystemInfo{}, nil, false) {
							t.Fatal("unexpected eviction")
						}
						select {
						case runner := <-req.successCh:
							if runner.numParallel != want {
								t.Fatalf("parallel = %d, want %d", runner.numParallel, want)
							}
						case err := <-req.errCh:
							t.Fatal(err)
						case <-ctx.Done():
							t.Fatal(ctx.Err())
						}
					})
				}
			}
		}
	}
}
