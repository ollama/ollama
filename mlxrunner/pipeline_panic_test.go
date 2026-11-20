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
		request := request{CompletionRequest: CompletionRequest{Options: api.Options{NumPredict: 1}}}
		var failure any
		func() {
			defer func() { failure = recover() }()
			_ = driveTestDecoder(context.Background(), &Runner{}, request, &cacheSession{inputs: []int32{1}}, d, 0)
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

func (panicForwardModel) Forward(_ *batch.Batch, caches []cache.Layer) (*mlx.Array, *mlx.Array) {
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
				s := &batchScheduler{r: r, slots: []*batchSlot{{cache: pc, targets: pc.caches}}}
				s.admit(request{Ctx: context.Background(), Tokens: []int32{1, 2}})
				s.step()
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
		request := request{
			Ctx:               context.Background(),
			Tokens:            []int32{1, 2},
			Responses:         make(chan CompletionResponse, 4),
			CompletionRequest: CompletionRequest{Options: api.Options{NumPredict: 1}},
		}
		s := &batchScheduler{r: r, slots: []*batchSlot{{cache: pc, targets: pc.caches}}}
		mlx.Scoped(func() { s.admit(request) })
		for range 10 {
			mlx.Scoped(func() { s.step() })
			if s.slots[0].request == nil {
				break
			}
		}
		if s.slots[0].request != nil {
			t.Fatal("request did not finish")
		}
		_, final := collectResponses(request.Responses)
		if !final.Done || final.EvalCount != 1 {
			t.Fatalf("final response = %+v", final)
		}
		if len(pc.root.children) != 1 {
			t.Fatalf("saved cache paths = %d, want 1", len(pc.root.children))
		}
	})
}
