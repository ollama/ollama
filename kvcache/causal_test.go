package kvcache

import (
	"math"
	"slices"
	"testing"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
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

	cache.Init(backend, ml.DTypeF16, 1, 16, 16)

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

	cache.Init(backend, ml.DTypeF16, 1, 16, 16)

	tests := []testCase{
		{
			name:          "FirstBatch",
			in:            []float32{1, 2, 3, 4},
			inShape:       []int{1, 1, 4},
			seqs:          []int{0, 0, 0, 0},
			pos:           []int32{0, 1, 2, 3},
			expected:      []float32{1, 2, 3, 4},
			expectedShape: []int{1, 1, 4},
			expectedMask:  []float32{0, float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), 0, 0, float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), 0, 0, float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), 0, 0},
		},
		{
			name:          "SecondBatch",
			in:            []float32{5, 6},
			inShape:       []int{1, 1, 2},
			seqs:          []int{0, 0},
			pos:           []int32{4, 5},
			expected:      []float32{5, 6, 3, 4},
			expectedShape: []int{1, 1, 4},
			expectedMask:  []float32{0, float32(math.Inf(-1)), float32(math.Inf(-1)), 0, 0, 0, float32(math.Inf(-1)), float32(math.Inf(-1))},
		},
	}

	testCache(t, backend, cache, tests)
}

func TestChunkedAttention(t *testing.T) {
	cache := NewChunkedAttentionCache(2, nil)
	defer cache.Close()

	var b testBackend
	cache.Init(&b, ml.DTypeF16, 1, 16, 16)

	x := float32(math.Inf(-1))

	testCache(
		t, &b, cache,
		[]testCase{
			{
				name:          "FirstBatch",
				in:            []float32{1, 2, 3, 4},
				inShape:       []int{1, 1, 4},
				seqs:          []int{0, 0, 0, 0},
				pos:           []int32{0, 1, 2, 3},
				expected:      []float32{1, 2, 3, 4},
				expectedShape: []int{1, 1, 4},
				expectedMask: []float32{
					0, x, x, x,
					0, 0, x, x,
					x, x, 0, x,
					x, x, 0, 0,
				},
			},
			{
				name:          "SecondBatch",
				in:            []float32{5, 6, 7},
				inShape:       []int{1, 1, 3},
				seqs:          []int{0, 0, 0},
				pos:           []int32{4, 5, 6},
				expected:      []float32{1, 2, 3, 4, 5, 6, 7},
				expectedShape: []int{1, 1, 7},
				expectedMask: []float32{
					x, x, x, x, 0, x, x,
					x, x, x, x, 0, 0, x,
					x, x, x, x, x, x, 0,
				},
			},
			{
				name:          "ThirdBatch",
				in:            []float32{8, 9},
				inShape:       []int{1, 1, 2},
				seqs:          []int{0, 0},
				pos:           []int32{7, 8},
				expected:      []float32{1, 2, 3, 4, 5, 6, 7, 8, 9},
				expectedShape: []int{1, 1, 9},
				expectedMask: []float32{
					x, x, x, x, x, x, 0, 0, x,
					x, x, x, x, x, x, x, x, 0,
				},
			},
		},
	)
}

func TestSequences(t *testing.T) {
	backend := &testBackend{}
	cache := NewCausalCache(nil)
	defer cache.Close()

	cache.Init(backend, ml.DTypeF16, 1, 16, 16)

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

	cache.Init(backend, ml.DTypeF16, 1, 16, 16)

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

	cache.Init(backend, ml.DTypeF16, 1, 16, 16)

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

	cache.Init(backend, ml.DTypeF16, 1, 16, 16)

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

			err := cache.StartForward(context, input.Batch{Positions: test.pos, Sequences: test.seqs}, false)
			if err != nil {
				panic(err)
			}

			cache.SetLayer(0)
			tensor := context.FromFloatSlice(test.in, test.inShape...)
			cache.Put(context, tensor, tensor)

			out, _, mask := cache.Get(context)

			context.Forward(out, mask).Compute(out, mask)

			if !slices.Equal(out.Floats(), test.expected) {
				t.Errorf("TestCache: have %v; want %v", out.Floats(), test.expected)
			}

			if !slices.Equal(out.Shape(), test.expectedShape) {
				t.Errorf("TestCache: has shape %v; want %v", out.Shape(), test.expectedShape)
			}

			if !slices.Equal(mask.Floats(), test.expectedMask) {
				t.Errorf("TestCache: have mask: have %v want %v", mask.Floats(), test.expectedMask)
			}
		})
	}
}

func TestCanResume(t *testing.T) {
	backend := &testBackend{}
	windowSize := int32(4)
	cache := NewSWACache(windowSize, nil)
	defer cache.Close()

	cache.Init(backend, ml.DTypeF16, 1, 16, 16)

	context := backend.NewContext()
	defer context.Close()

	err := cache.StartForward(context, input.Batch{
		Positions: []int32{0, 1, 2, 3},
		Sequences: []int{0, 0, 0, 0},
	}, false)
	if err != nil {
		t.Fatalf("StartForward failed: %v", err)
	}

	cache.SetLayer(0)
	tensor := context.FromFloatSlice([]float32{1, 2, 3, 4}, 1, 1, 4)
	cache.Put(context, tensor, tensor)

	// with window size 4, nothing has slid out of the window yet
	if !cache.CanResume(0, 0) {
		t.Errorf("CanResume(0, 0) = false, want true (within window)")
	}
	if !cache.CanResume(0, 1) {
		t.Errorf("CanResume(0, 1) = false, want true (within window)")
	}
	if !cache.CanResume(0, 2) {
		t.Errorf("CanResume(0, 2) = false, want true (within window)")
	}
	if !cache.CanResume(0, 3) {
		t.Errorf("CanResume(0, 3) = false, want true (latest position)")
	}

	// shift window by adding position 4
	err = cache.StartForward(context, input.Batch{
		Positions: []int32{4, 5},
		Sequences: []int{0, 0},
	}, false)
	if err != nil {
		t.Fatalf("StartForward failed: %v", err)
	}

	cache.SetLayer(0)
	tensor = context.FromFloatSlice([]float32{5, 6}, 1, 1, 2)
	cache.Put(context, tensor, tensor)

	// only the latest position has overlapping windows
	if cache.CanResume(0, 0) {
		t.Errorf("after shift: CanResume(0, 0) = true, want false (outside window)")
	}
	if cache.CanResume(0, 1) {
		t.Errorf("after shift: CanResume(0, 1) = true, want false (outside window)")
	}
	if cache.CanResume(0, 2) {
		t.Errorf("after shift: CanResume(0, 2) = true, want false (outside window)")
	}
	if cache.CanResume(0, 3) {
		t.Errorf("after shift: CanResume(0, 3) = true, want false (outside window)")
	}
	if cache.CanResume(0, 4) {
		t.Errorf("after shift: CanResume(0, 4) = true, want false (outside window)")
	}
	if !cache.CanResume(0, 5) {
		t.Errorf("after shift: CanResume(0, 5) = false, want true (latest position)")
	}
}

type testBackend struct {
	ml.Backend
}

func (b *testBackend) NewContext() ml.Context {
	return &testContext{}
}

func (b *testBackend) NewContextSize(int) ml.Context {
	return &testContext{}
}

type testContext struct {
	ml.Context
}

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

func (c *testContext) FromFloatSlice(s []float32, shape ...int) ml.Tensor {
	t := c.Empty(ml.DTypeF32, shape...).(*testTensor)

	copy(t.data, s)

	return t
}

func (c *testContext) FromIntSlice(s []int32, shape ...int) ml.Tensor {
	f := make([]float32, len(s))
	for i := range f {
		f[i] = float32(s[i])
	}

	out := c.FromFloatSlice(f, shape...)
	out.(*testTensor).dtype = ml.DTypeI32

	return out
}

func (c *testContext) Arange(start, stop, step float32, dtype ml.DType) ml.Tensor {
	s := make([]float32, 0, int((stop-start)/step))
	for i := start; i < stop; i += step {
		s = append(s, i)
	}

	out := c.FromFloatSlice(s, len(s))
	out.(*testTensor).dtype = dtype
	return out
}

func (c *testContext) Input() ml.Context    { return c }
func (c *testContext) Layer(int) ml.Context { return c }

func (c *testContext) Forward(...ml.Tensor) ml.Context { return c }

func (c *testContext) Compute(...ml.Tensor) {}

func (c *testContext) Reserve() {}

func (c *testContext) MaxGraphNodes() int {
	return 10
}

func (c *testContext) Close() {}

type testTensor struct {
	ml.Tensor

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

func (t *testTensor) Floats() []float32 {
	out := make([]float32, len(t.data))
	copy(out, t.data)
	return out
}

func (t *testTensor) Neg(ctx ml.Context) ml.Tensor {
	out := ctx.Empty(t.DType(), t.Shape()...).(*testTensor)
	for i := range out.data {
		out.data[i] = -t.data[i]
	}
	return out
}

func (t *testTensor) Add(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	out := ctx.Empty(t.DType(), t.Shape()...).(*testTensor)

	for i := range out.data {
		out.data[i] = t.data[i] + t2.(*testTensor).data[i]
	}

	return out
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

func (t *testTensor) Copy(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	copy(t2.(*testTensor).data, t.data)
	return nil
}
