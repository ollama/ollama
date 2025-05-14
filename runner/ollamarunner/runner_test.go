package ollamarunner

import (
	"context"
	"sync"
	"testing"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
	"github.com/ollama/ollama/sample"
	"golang.org/x/sync/semaphore"
)

// testBackend implements ml.Backend with minimal functionality required for tests.
type testBackend struct{}

func (b *testBackend) Config() fs.Config             { return testConfig{} }
func (b *testBackend) Get(string) ml.Tensor          { return nil }
func (b *testBackend) NewContext() ml.Context        { return &testContext{} }
func (b *testBackend) NewContextSize(int) ml.Context { return &testContext{} }

// testConfig is a stub implementation of fs.Config used by testBackend.
type testConfig struct{}

func (testConfig) Architecture() string                  { return "" }
func (testConfig) String(string, ...string) string       { return "" }
func (testConfig) Uint(string, ...uint32) uint32         { return 0 }
func (testConfig) Float(string, ...float32) float32      { return 0 }
func (testConfig) Bool(string, ...bool) bool             { return false }
func (testConfig) Strings(string, ...[]string) []string  { return nil }
func (testConfig) Ints(string, ...[]int32) []int32       { return nil }
func (testConfig) Floats(string, ...[]float32) []float32 { return nil }

type testContext struct{}

func (c *testContext) Empty(dtype ml.DType, shape ...int) ml.Tensor {
	sz := 1
	for _, s := range shape {
		sz *= s
	}
	return &testTensor{dtype: dtype, data: make([]float32, sz), shape: shape}
}
func (c *testContext) Zeros(dtype ml.DType, shape ...int) ml.Tensor { return c.Empty(dtype, shape...) }
func (c *testContext) FromFloatSlice(s []float32, shape ...int) (ml.Tensor, error) {
	t := c.Empty(ml.DTypeF32, shape...).(*testTensor)
	copy(t.data, s)
	return t, nil
}
func (c *testContext) FromIntSlice(s []int32, shape ...int) (ml.Tensor, error) {
	f := make([]float32, len(s))
	for i, v := range s {
		f[i] = float32(v)
	}
	out, _ := c.FromFloatSlice(f, shape...)
	out.(*testTensor).dtype = ml.DTypeI32
	return out, nil
}
func (c *testContext) Arange(start, stop, step float32, dtype ml.DType) ml.Tensor {
	return c.Empty(dtype, int((stop-start)/step))
}
func (c *testContext) Forward(...ml.Tensor) ml.Context { return c }
func (c *testContext) Compute(...ml.Tensor)            {}
func (c *testContext) Reserve() error                  { return nil }
func (c *testContext) MaxGraphNodes() int              { return 0 }
func (c *testContext) Close()                          {}
func (c *testContext) Input() ml.Context               { return c }
func (c *testContext) Layer(int) ml.Context            { return c }

type testTensor struct {
	ml.Tensor
	dtype ml.DType
	data  []float32
	shape []int
}

func (t *testTensor) Dim(n int) int    { return t.shape[n] }
func (t *testTensor) Stride(n int) int { return 0 }
func (t *testTensor) Shape() []int     { return t.shape }
func (t *testTensor) DType() ml.DType  { return t.dtype }
func (t *testTensor) Bytes() []byte    { return nil }
func (t *testTensor) Floats() []float32 {
	out := make([]float32, len(t.data))
	copy(out, t.data)
	return out
}
func (t *testTensor) Neg(ctx ml.Context) ml.Tensor { return nil }
func (t *testTensor) Add(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	out, _ := ctx.(*testContext).FromFloatSlice(nil, len(t.data))
	return out
}
func (t *testTensor) Mul(ctx ml.Context, t2 ml.Tensor) ml.Tensor            { return nil }
func (t *testTensor) Mulmat(ctx ml.Context, t2 ml.Tensor) ml.Tensor         { return nil }
func (t *testTensor) MulmatFullPrec(ctx ml.Context, t2 ml.Tensor) ml.Tensor { return nil }
func (t *testTensor) MulmatID(ctx ml.Context, t2, ids ml.Tensor) ml.Tensor  { return nil }
func (t *testTensor) Softmax(ctx ml.Context) ml.Tensor                      { return nil }
func (t *testTensor) LayerNorm(ctx ml.Context, w, b ml.Tensor, e float32) ml.Tensor {
	return nil
}
func (t *testTensor) View(ctx ml.Context, offset int, shape ...int) ml.Tensor {
	return ctx.(*testContext).Empty(t.dtype, shape...)
}
func (t *testTensor) Copy(ctx ml.Context, dest ml.Tensor) ml.Tensor {
	copy(dest.(*testTensor).data, t.data)
	return nil
}

// fakeModel implements model.Model and model.TextProcessor.
type fakeModel struct {
	model.Base
	decode  map[int32]string
	logits  [][]float32
	call    int
	backend ml.Backend
}

func (f *fakeModel) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	idx := f.call
	if idx >= len(f.logits) {
		idx = len(f.logits) - 1
	}
	f.call++
	return ctx.FromFloatSlice(f.logits[idx], len(f.logits[idx]))
}

func (f *fakeModel) Backend() ml.Backend {
	if f.backend == nil {
		f.backend = &testBackend{}
	}
	return f.backend
}

func (f *fakeModel) Encode(string, bool) ([]int32, error) { return nil, nil }
func (f *fakeModel) Decode(ids []int32) (string, error) {
	var s string
	for _, id := range ids {
		s += f.decode[id]
	}
	return s, nil
}
func (f *fakeModel) Is(id int32, sp model.Special) bool { return false }
func (f *fakeModel) Vocabulary() *model.Vocabulary      { return &model.Vocabulary{} }

var _ model.Model = (*fakeModel)(nil)
var _ model.TextProcessor = (*fakeModel)(nil)

func TestProcessBatchUnicode(t *testing.T) {
	tests := []struct {
		name   string
		decode map[int32]string
		logits [][]float32
		want   string
	}{
		{
			name:   "emoji",
			decode: map[int32]string{0: "A", 1: "😀", 2: "👍", 3: "!"},
			logits: [][]float32{{10, 0, 0, 0}, {0, 10, 0, 0}, {0, 0, 10, 0}, {0, 0, 0, 10}},
			want:   "A😀👍!",
		},
		{
			name:   "ascii",
			decode: map[int32]string{0: "H", 1: "e", 2: "y"},
			logits: [][]float32{{10, 0, 0}, {0, 10, 0}, {0, 0, 10}},
			want:   "Hey",
		},
		{
			name:   "multibyte",
			decode: map[int32]string{0: "世", 1: "界", 2: "😊"},
			logits: [][]float32{{10, 0, 0}, {0, 10, 0}, {0, 0, 10}},
			want:   "世界😊",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &fakeModel{decode: tt.decode, logits: tt.logits}

			s := &Server{model: m, batchSize: 1, parallel: 1}
			s.cache = &InputCache{enabled: true, slots: []InputCacheSlot{{Id: 0}}, numCtx: 10}
			s.seqs = make([]*Sequence, 1)
			s.seqsSem = semaphore.NewWeighted(1)
			if err := s.seqsSem.Acquire(context.Background(), 1); err != nil {
				t.Fatal(err)
			}
			s.cond = sync.NewCond(&s.mu)

			seq := &Sequence{
				inputs:     []input.Input{{Token: 0}},
				cache:      &s.cache.slots[0],
				responses:  make(chan string, 10),
				quit:       make(chan bool, 1),
				numPredict: len(tt.logits),
				sampler:    sample.NewSampler(0, 0, 0, 0, 0, nil),
				embedding:  make(chan []float32, 1),
			}
			s.seqs[0] = seq

			for {
				if err := s.processBatch(); err != nil {
					t.Fatal(err)
				}
				if s.seqs[0] == nil {
					break
				}
			}

			var result string
			for r := range seq.responses {
				result += r
			}

			if result != tt.want {
				t.Fatalf("got %q want %q", result, tt.want)
			}
		})
	}
}
