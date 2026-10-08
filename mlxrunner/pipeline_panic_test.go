package mlxrunner

import (
	"context"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	sampler "github.com/ollama/ollama/mlxrunner/sample"
)

type panicDecoder struct {
	drained bool
}

func (*panicDecoder) next(int) ([]sampler.Result, error) { panic("decode failed") }
func (d *panicDecoder) drain() ([]sampler.Result, int, error) {
	d.drained = true
	panic("second drain")
}
func (*panicDecoder) close() {}

// In #18856, a second drain replaced the original MLX panic.
func TestDecodePreservesPanic(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		d := &panicDecoder{}
		request := Request{CompletionRequest: CompletionRequest{Options: api.Options{NumPredict: 1}}}
		var failure any
		func() {
			defer func() { failure = recover() }()
			_ = (&Runner{}).decode(context.Background(), request, &cacheSession{inputs: []int32{1}}, d, 0)
		}()
		if failure != "decode failed" {
			t.Fatalf("panic = %v, want original decode failure", failure)
		}
		if d.drained {
			t.Fatal("decoder drained after a panic")
		}
	})
}

type panicForwardModel struct{ textOnlyModel }

func (panicForwardModel) Forward(_ *batch.Batch, caches []cache.Cache) (*mlx.Array, *mlx.Array) {
	caches[0].(*fakeRewindableCache).feed([]int32{1, 2, 3})
	panic("forward failed")
}

// In #18856, cache close replaced the original panic with an offset error.
func TestGeneratePreservesPanicWithoutSavingCache(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		kv := &fakeRewindableCache{tracker: &snapshotTracker{}}
		pc := newPrefixCache([]cache.Cache{kv})
		r := &Runner{Model: panicForwardModel{}, cache: pc}
		var failure any
		func() {
			defer func() { failure = recover() }()
			mlx.Scoped(func() {
				_ = r.generate(context.Background(), Request{Tokens: []int32{1, 2}})
			})
		}()
		if failure != "forward failed" {
			t.Fatalf("panic = %v, want original forward failure", failure)
		}
		if len(pc.root.children) != 0 {
			t.Fatalf("saved %d cache paths after failed forward", len(pc.root.children))
		}
	})
}

func TestGenerateStillSavesCacheOnSuccess(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		kv := &fakeRewindableCache{tracker: &snapshotTracker{}}
		pc := newPrefixCache([]cache.Cache{kv})
		tok := newTestTokenizer(t, []int32{7})
		r := &Runner{
			Model:     &fakeMTPModel{predict: map[int32]int32{1: 2, 2: 3, 3: 7}, tok: tok},
			Tokenizer: tok,
			Sampler:   sampler.New(4096),
			cache:     pc,
		}
		request := Request{
			Tokens:            []int32{1, 2},
			Responses:         make(chan CompletionResponse, 4),
			CompletionRequest: CompletionRequest{Options: api.Options{NumPredict: 1}},
		}
		mlx.Scoped(func() {
			if err := r.generate(context.Background(), request); err != nil {
				t.Fatalf("generate: %v", err)
			}
		})
		if len(pc.root.children) != 1 {
			t.Fatalf("saved cache paths = %d, want 1", len(pc.root.children))
		}
	})
}
