package kvcache

import (
	"math"
	"slices"
	"testing"

	"github.com/ollama/ollama/ml"
)

type testCase struct {
	name          string
	in            []float32
	inShape       []int
	seqs          []int
	pos           []int32
	expected      []float32
	expectedShape []int
	expectedMask  []float32
}

func TestStore(t *testing.T) {
	backend := &testBackend{}
	cache := NewCausalCache(nil)
	defer cache.Close()

	cache.Init(backend, ml.DTypeF16, 16)

	tests := []testCase{
		{
			name:          "FirstBatch",
			in:            []float32{111, 211, 121, 221, 131, 231, 112, 212, 122, 222, 132, 232, 113, 213, 123, 223, 133, 233, 114, 214, 124, 224, 134, 234},
			inShape:       []int{2, 3, 4},
			seqs:          []int{0, 0, 0, 0},
			pos:           []int32{0, 1, 2, 3},
			expected:      []float32{111, 211, 121, 221, 131, 231, 112, 212, 122, 222, 132, 232, 113, 213, 123, 223, 133, 233, 114, 214, 124, 224, 134, 234},
			expectedShape: []int{2, 3, 4},
			expectedMask:  []float32{0, float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), 0, 0, float32(math.Inf(-1)), float32(math.Inf(-1)), 0, 0, 0, float32(math.Inf(-1)), 0, 0, 0, 0},
		},
		{
			name:          "SecondBatch",
			in:            []float32{115, 215, 125, 225, 135, 235},
			inShape:       []int{2, 3, 1},
			seqs:          []int{0},
			pos:           []int32{4},
			expected:      []float32{111, 211, 121, 221, 131, 231, 112, 212, 122, 222, 132, 232, 113, 213, 123, 223, 133, 233, 114, 214, 124, 224, 134, 234, 115, 215, 125, 225, 135, 235},
			expectedShape: []int{2, 3, 5},
			expectedMask:  []float32{0, 0, 0, 0, 0},
		},
	}

	testCache(t, backend, cache, tests)
}

func TestSWA(t *testing.T) {
	backend := &testBackend{}
	cache := NewSWACache(1, nil)
	defer cache.Close()

	cache.Init(backend, ml.DTypeF32, 16)

	tests := []testCase{
		{
			name:          "SlidingWindow",
			in:            []float32{1, 2, 3, 4},
			inShape:       []int{1, 1, 4},
			seqs:          []int{0, 0, 0, 0},
			pos:           []int32{0, 1, 2, 3},
			expected:      []float32{1, 2, 3, 4},
			expectedShape: []int{1, 1, 4},
			expectedMask:  []float32{0, float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), 0, 0, float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), 0, 0, float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), 0, 0},
		},
	}

	testCache(t, backend, cache, tests)
}

func TestSequences(t *testing.T) {
	backend := &testBackend{}
	cache := NewCausalCache(nil)
	defer cache.Close()

	cache.Init(backend, ml.DTypeF16, 16)

	tests := []testCase{
		{
			name:          "FirstBatch",
			in:            []float32{1, 2, 3, 4},
			inShape:       []int{1, 1, 4},
			seqs:          []int{0, 0, 1, 1},
			pos:           []int32{0, 1, 0, 1},
			expected:      []float32{1, 2, 3, 4},
			expectedShape: []int{1, 1, 4},
			expectedMask:  []float32{0, float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), 0, 0, float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), 0, float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), 0, 0},
		},
		{
			name:          "SecondBatch",
			in:            []float32{5, 6},
			inShape:       []int{1, 1, 2},
			seqs:          []int{0, 1},
			pos:           []int32{2, 2},
			expected:      []float32{1, 2, 3, 4, 5, 6},
			expectedShape: []int{1, 1, 6},
			expectedMask:  []float32{0, 0, float32(math.Inf(-1)), float32(math.Inf(-1)), 0, float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), 0, 0, float32(math.Inf(-1)), 0},
		},
	}

	testCache(t, backend, cache, tests)
}

func TestRemove(t *testing.T) {
	backend := &testBackend{}
	cache := NewCausalCache(func(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
		return key.Add(ctx, shift), nil
	})
	defer cache.Close()

	cache.Init(backend, ml.DTypeF16, 16)

	tests := []testCase{
		{
			name:          "FirstBatch",
			in:            []float32{1, 2, 3, 4},
			inShape:       []int{1, 1, 4},
			seqs:          []int{0, 0, 1, 1},
			pos:           []int32{0, 1, 0, 1},
			expected:      []float32{1, 2, 3, 4},
			expectedShape: []int{1, 1, 4},
			expectedMask:  []float32{0, float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), 0, 0, float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), 0, float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), 0, 0},
		},
	}

	testCache(t, backend, cache, tests)

	err := cache.Remove(0, 1, math.MaxInt32)
	if err != nil {
		panic(err)
	}

	tests = []testCase{
		{
			name:          "RemoveEnd",
			in:            []float32{5, 6},
			inShape:       []int{1, 1, 2},
			seqs:          []int{0, 1},
			pos:           []int32{1, 2},
			expected:      []float32{1, 2, 3, 4, 5, 6},
			expectedShape: []int{1, 1, 6},
			expectedMask:  []float32{0, float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), 0, float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), 0, 0, float32(math.Inf(-1)), 0},
		},
	}

	testCache(t, backend, cache, tests)

	err = cache.Remove(0, 0, 1)
	if err != nil {
		panic(err)
	}

	tests = []testCase{
		{
			name:          "RemoveMiddle",
			in:            []float32{7, 8},
			inShape:       []int{1, 1, 2},
			seqs:          []int{0, 0},
			pos:           []int32{1, 2},
			expected:      []float32{7, 8, 3, 4, 4},
			expectedShape: []int{1, 1, 5},
			expectedMask:  []float32{0, float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), 0, 0, 0, float32(math.Inf(-1)), float32(math.Inf(-1)), 0},
		},
	}

	testCache(t, backend, cache, tests)
}

func TestDefrag(t *testing.T) {
	backend := &testBackend{}
	cache := NewCausalCache(func(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
		return key.Add(ctx, shift), nil
	})
	defer cache.Close()

	cache.Init(backend, ml.DTypeF16, 16)

	tests := []testCase{
		{
			name:          "FirstBatch",
			in:            []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			inShape:       []int{1, 1, 16},
			seqs:          []int{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			pos:           []int32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
			expected:      []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			expectedShape: []int{1, 1, 16},
			expectedMask:  []float32{0, float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), 0, 0, float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), 0, 0, 0, float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), 0, 0, 0, 0, float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), 0, 0, 0, 0, 0, float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), 0, 0, 0, 0, 0, 0, float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), 0, 0, 0, 0, 0, 0, 0, float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), 0, 0, 0, 0, 0, 0, 0, 0, float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), 0, 0, 0, 0, 0, 0, 0, 0, 0, float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, float32(math.Inf(-1)), float32(math.Inf(-1)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, float32(math.Inf(-1)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		},
	}

	testCache(t, backend, cache, tests)

	err := cache.Remove(0, 2, 4)
	if err != nil {
		panic(err)
	}

	err = cache.Remove(0, 13, math.MaxInt32)
	if err != nil {
		panic(err)
	}

	tests = []testCase{
		{
			name:          "Defrag",
			in:            []float32{17, 18, 19},
			inShape:       []int{1, 1, 3},
			seqs:          []int{0, 0, 0},
			pos:           []int32{16, 17, 18},
			expected:      []float32{1, 2, 12, 13, 3, 4, 5, 6, 7, 8, 9, 10, 11, 17, 18, 19},
			expectedShape: []int{1, 1, 16},
			expectedMask:  []float32{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, float32(math.Inf(-1)), float32(math.Inf(-1)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, float32(math.Inf(-1)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		},
	}

	testCache(t, backend, cache, tests)
}

func TestCopy(t *testing.T) {
	backend := &testBackend{}
	cache := NewCausalCache(func(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) { return key, nil })
	defer cache.Close()

	cache.Init(backend, ml.DTypeF16, 16)

	tests := []testCase{
		{
			name:          "FirstBatch",
			in:            []float32{1, 2, 3, 4},
			inShape:       []int{1, 1, 4},
			seqs:          []int{0, 0, 0, 0},
			pos:           []int32{0, 1, 2, 3},
			expected:      []float32{1, 2, 3, 4},
			expectedShape: []int{1, 1, 4},
			expectedMask:  []float32{0, float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), 0, 0, float32(math.Inf(-1)), float32(math.Inf(-1)), 0, 0, 0, float32(math.Inf(-1)), 0, 0, 0, 0},
		},
	}

	testCache(t, backend, cache, tests)

	cache.CopyPrefix(0, 1, 2)

	tests = []testCase{
		{
			name:          "Copy",
			in:            []float32{5, 6},
			inShape:       []int{1, 1, 2},
			seqs:          []int{1, 1},
			pos:           []int32{3, 4},
			expected:      []float32{1, 2, 3, 4, 5, 6},
			expectedShape: []int{1, 1, 6},
			expectedMask:  []float32{0, 0, float32(math.Inf(-1)), float32(math.Inf(-1)), 0, float32(math.Inf(-1)), 0, 0, float32(math.Inf(-1)), float32(math.Inf(-1)), 0, 0},
		},
	}

	testCache(t, backend, cache, tests)
}

func testCache(t *testing.T, backend ml.Backend, cache Cache, tests []testCase) {
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			context := backend.NewContext()
			defer context.Close()

			err := cache.StartForward(context, test.pos, test.seqs)
			if err != nil {
				panic(err)
			}

			cache.SetLayer(0)
			tensor, _ := context.FromFloatSlice(test.in, test.inShape...)
			cache.Put(context, tensor, tensor)

			out, _, mask := cache.Get(context)

			context.Forward(out, mask).Compute(out, mask)

			if !slices.Equal(out.Floats(), test.expected) || !slices.Equal(out.Shape(), test.expectedShape) || !slices.Equal(mask.Floats(), test.expectedMask) {
				t.Errorf("TestCache: have %v (shape %v); want %v (shape %v); mask: have %v (shape %v) want %v", out.Floats(), out.Shape(), test.expected, test.expectedShape, mask.Floats(), mask.Shape(), test.expectedMask)
			}
		})
	}
}

type testBackend struct{}

func (b *testBackend) Config() ml.Config {
	panic("not implemented")
}

func (b *testBackend) Get(name string) ml.Tensor {
	panic("not implemented")
}

func (b *testBackend) NewContext() ml.Context {
	return &testContext{}
}

func (b *testBackend) SystemInfo() string {
	return "not implemented"
}

type testContext struct{}

func (c *testContext) Empty(dtype ml.DType, shape ...int) ml.Tensor {
	total := 0

	if len(shape) > 0 {
		total = 1
		for _, s := range shape {
			total *= s
		}
	}

	return &testTensor{dtype: dtype, elementSize: 4, data: make([]float32, total), shape: shape}
}

func (c *testContext) Zeros(dtype ml.DType, shape ...int) ml.Tensor {
	return c.Empty(dtype, shape...)
}

func (c *testContext) FromFloatSlice(s []float32, shape ...int) (ml.Tensor, error) {
	t := c.Empty(ml.DTypeF32, shape...).(*testTensor)

	copy(t.data, s)

	return t, nil
}

func (c *testContext) FromIntSlice(s []int32, shape ...int) (ml.Tensor, error) {
	f := make([]float32, len(s))
	for i := range f {
		f[i] = float32(s[i])
	}

	out, _ := c.FromFloatSlice(f, shape...)
	out.(*testTensor).dtype = ml.DTypeI32

	return out, nil
}

func (c *testContext) Forward(...ml.Tensor) ml.Context { return c }

func (c *testContext) Compute(...ml.Tensor) {}

func (c *testContext) MaxTensors() int {
	return 10
}

func (c *testContext) Close() {}

type testTensor struct {
	dtype       ml.DType
	elementSize int
	data        []float32
	shape       []int
}

func (t *testTensor) Dim(n int) int {
	return t.shape[n]
}

func (t *testTensor) Stride(n int) int {
	stride := t.elementSize
	for i := range n {
		stride *= t.shape[i]
	}

	return stride
}

func (t *testTensor) Shape() []int {
	return t.shape
}

func (t *testTensor) DType() ml.DType {
	return t.dtype
}

func (t *testTensor) Bytes() []byte {
	panic("not implemented")
}

func (t *testTensor) Floats() []float32 {
	out := make([]float32, len(t.data))
	copy(out, t.data)
	return out
}

func (t *testTensor) Add(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	out := ctx.Empty(t.DType(), t.Shape()...).(*testTensor)

	for i := range out.data {
		out.data[i] = t.data[i] + t2.(*testTensor).data[i]
	}

	return out
}

func (t *testTensor) Mul(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	panic("not implemented")
}

func (t *testTensor) Mulmat(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	panic("not implemented")
}

func (t *testTensor) MulmatFullPrec(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	panic("not implemented")
}

func (t *testTensor) Softmax(ctx ml.Context) ml.Tensor {
	panic("not implemented")
}

func (t *testTensor) LayerNorm(ctx ml.Context, weight, bias ml.Tensor, eps float32) ml.Tensor {
	panic("not implemented")
}

func (t *testTensor) RMSNorm(ctx ml.Context, weight ml.Tensor, eps float32) ml.Tensor {
	panic("not implemented")
}

func (t *testTensor) Scale(ctx ml.Context, s float64) ml.Tensor {
	panic("not implemented")
}

func (t *testTensor) Conv2D(ctx ml.Context, weight ml.Tensor, s0, s1, p0, p1, d0, d1 int) ml.Tensor {
	panic("not implemented")
}

func (t *testTensor) RoPE(ctx ml.Context, positionIDs, ropeFactors ml.Tensor, dim uint32, base, scale float32) ml.Tensor {
	panic("not implemented")
}

func (t *testTensor) Tanh(ctx ml.Context) ml.Tensor {
	panic("not implemented")
}

func (t *testTensor) GELU(ctx ml.Context) ml.Tensor {
	panic("not implemented")
}

func (t *testTensor) SILU(ctx ml.Context) ml.Tensor {
	panic("not implemented")
}

func (t *testTensor) Reshape(ctx ml.Context, shape ...int) ml.Tensor {
	panic("not implemented")
}

func (t *testTensor) View(ctx ml.Context, offset int, shape ...int) ml.Tensor {
	offset /= t.elementSize

	var s []int

	switch len(shape) {
	case 1:
		s = []int{shape[0]}
	case 5:
		s = []int{shape[0], shape[2], shape[4]}
	default:
		panic("unsupported number of dimensions")
	}

	context := &testContext{}

	view := context.Empty(t.dtype, s...).(*testTensor)
	view.data = t.data[offset : offset+len(view.data)]

	return view
}

func (t *testTensor) Permute(ctx ml.Context, shape ...int) ml.Tensor {
	panic("not implemented")
}

func (t *testTensor) Contiguous(ctx ml.Context) ml.Tensor {
	panic("not implemented")
}

func (t *testTensor) Pad(ctx ml.Context, shape ...int) ml.Tensor {
	panic("not implemented")
}

func (t *testTensor) Unpad(ctx ml.Context, shape ...int) ml.Tensor {
	panic("not implemented")
}

func (t *testTensor) Stack(ctx ml.Context, dim int, s ...ml.Tensor) ml.Tensor {
	panic("not implemented")
}

func (t *testTensor) Concat(ctx ml.Context, t2 ml.Tensor, dim int) ml.Tensor {
	panic("not implemented")
}

func (t *testTensor) Rows(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	panic("not implemented")
}

func (t *testTensor) Copy(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	copy(t2.(*testTensor).data, t.data)
	return nil
}
