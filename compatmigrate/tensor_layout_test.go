package compatmigrate

import (
	"slices"
	"testing"

	"github.com/ollama/ollama/fs/gguf"
)

func sequence(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(i)
	}
	return out
}

func TestSliceTensorDim(t *testing.T) {
	data := sequence(12)
	dims := []int{2, 3, 2}

	cases := []struct {
		dim, start, end int
		want            []float32
	}{
		{dim: 0, start: 1, end: 2, want: []float32{6, 7, 8, 9, 10, 11}},
		{dim: 1, start: 1, end: 3, want: []float32{2, 3, 4, 5, 8, 9, 10, 11}},
		{dim: 2, start: 0, end: 1, want: []float32{0, 2, 4, 6, 8, 10}},
		{dim: 1, start: 0, end: 3, want: data},
		{dim: 1, start: 1, end: 1, want: []float32{}},
	}
	for _, tc := range cases {
		got, err := sliceTensorDim(data, dims, tc.dim, tc.start, tc.end)
		if err != nil {
			t.Fatalf("sliceTensorDim(dim=%d, %d:%d) error = %v", tc.dim, tc.start, tc.end, err)
		}
		if !slices.Equal(got, tc.want) {
			t.Errorf("sliceTensorDim(dim=%d, %d:%d)\n got %v\nwant %v", tc.dim, tc.start, tc.end, got, tc.want)
		}
	}

	bad := []struct {
		name            string
		data            []float32
		dim, start, end int
	}{
		{name: "dim out of range", data: data, dim: 3, start: 0, end: 1},
		{name: "negative dim", data: data, dim: -1, start: 0, end: 1},
		{name: "end before start", data: data, dim: 1, start: 2, end: 1},
		{name: "end past dim", data: data, dim: 1, start: 0, end: 4},
		{name: "negative start", data: data, dim: 1, start: -1, end: 1},
		{name: "data size mismatch", data: data[:11], dim: 0, start: 0, end: 1},
	}
	for _, tc := range bad {
		if _, err := sliceTensorDim(tc.data, dims, tc.dim, tc.start, tc.end); err == nil {
			t.Errorf("%s: expected error", tc.name)
		}
	}
}

func TestConcatTensorDim(t *testing.T) {
	a := tensorPart{data: []float32{0, 1, 2, 3}, dims: []int{2, 2}}
	b := tensorPart{data: []float32{4, 5, 6, 7, 8, 9}, dims: []int{2, 3}}
	c := tensorPart{data: []float32{4, 5}, dims: []int{1, 2}}

	got, dims, err := concatTensorDim(1, a, b)
	if err != nil {
		t.Fatalf("concatTensorDim(1) error = %v", err)
	}
	if !slices.Equal(dims, []int{2, 5}) {
		t.Errorf("concatTensorDim(1) dims = %v, want [2 5]", dims)
	}
	if want := []float32{0, 1, 4, 5, 6, 2, 3, 7, 8, 9}; !slices.Equal(got, want) {
		t.Errorf("concatTensorDim(1)\n got %v\nwant %v", got, want)
	}

	got, dims, err = concatTensorDim(0, a, c)
	if err != nil {
		t.Fatalf("concatTensorDim(0) error = %v", err)
	}
	if !slices.Equal(dims, []int{3, 2}) {
		t.Errorf("concatTensorDim(0) dims = %v, want [3 2]", dims)
	}
	if want := []float32{0, 1, 2, 3, 4, 5}; !slices.Equal(got, want) {
		t.Errorf("concatTensorDim(0)\n got %v\nwant %v", got, want)
	}

	got, dims, err = concatTensorDim(0, a)
	if err != nil {
		t.Fatalf("concatTensorDim(single) error = %v", err)
	}
	if !slices.Equal(dims, a.dims) || !slices.Equal(got, a.data) {
		t.Errorf("concatTensorDim(single) = %v %v, want %v %v", got, dims, a.data, a.dims)
	}

	bad := []struct {
		name  string
		dim   int
		parts []tensorPart
	}{
		{name: "no parts", dim: 0},
		{name: "dim out of range", dim: 2, parts: []tensorPart{a, b}},
		{name: "incompatible dims", dim: 1, parts: []tensorPart{a, c}},
		{name: "rank mismatch", dim: 0, parts: []tensorPart{a, {data: []float32{0, 1}, dims: []int{2}}}},
		{name: "data size mismatch", dim: 0, parts: []tensorPart{a, {data: []float32{0}, dims: []int{1, 2}}}},
	}
	for _, tc := range bad {
		if _, _, err := concatTensorDim(tc.dim, tc.parts...); err == nil {
			t.Errorf("%s: expected error", tc.name)
		}
	}
}

func TestPermuteTensor(t *testing.T) {
	cases := []struct {
		dims     []int
		perm     []int
		wantDims []int
		want     []float32
	}{
		{dims: []int{2, 3}, perm: []int{1, 0}, wantDims: []int{3, 2}, want: []float32{0, 3, 1, 4, 2, 5}},
		{dims: []int{2, 3}, perm: []int{0, 1}, wantDims: []int{2, 3}, want: sequence(6)},
		{dims: []int{2, 2, 3}, perm: []int{0, 2, 1}, wantDims: []int{2, 3, 2}, want: []float32{0, 3, 1, 4, 2, 5, 6, 9, 7, 10, 8, 11}},
		{dims: []int{2, 2, 3}, perm: []int{2, 0, 1}, wantDims: []int{3, 2, 2}, want: []float32{0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11}},
		{dims: []int{3}, perm: []int{0}, wantDims: []int{3}, want: sequence(3)},
	}
	for _, tc := range cases {
		got, dims, err := permuteTensor(sequence(tensorSize(tc.dims)), tc.dims, tc.perm...)
		if err != nil {
			t.Fatalf("permuteTensor(%v, %v) error = %v", tc.dims, tc.perm, err)
		}
		if !slices.Equal(dims, tc.wantDims) {
			t.Errorf("permuteTensor(%v, %v) dims = %v, want %v", tc.dims, tc.perm, dims, tc.wantDims)
		}
		if !slices.Equal(got, tc.want) {
			t.Errorf("permuteTensor(%v, %v)\n got %v\nwant %v", tc.dims, tc.perm, got, tc.want)
		}
	}

	bad := []struct {
		name string
		data []float32
		dims []int
		perm []int
	}{
		{name: "perm too short", data: sequence(6), dims: []int{2, 3}, perm: []int{0}},
		{name: "repeated axis", data: sequence(6), dims: []int{2, 3}, perm: []int{0, 0}},
		{name: "axis out of range", data: sequence(6), dims: []int{2, 3}, perm: []int{0, 2}},
		{name: "data size mismatch", data: sequence(5), dims: []int{2, 3}, perm: []int{1, 0}},
	}
	for _, tc := range bad {
		if _, _, err := permuteTensor(tc.data, tc.dims, tc.perm...); err == nil {
			t.Errorf("%s: expected error", tc.name)
		}
	}
}

// Two heads, each with qkNope=2 and vHeadDim=1 rows of kvLoraRank=3, laid out
// with heads outermost (kv last) or kv outermost (kv first).
func TestGLM47FlashRepackKVB(t *testing.T) {
	mla := glm47FlashMLA{numHeads: 2, qkNope: 2, vHeadDim: 1, kvLoraRank: 3}
	data := sequence(18)

	cases := []struct {
		name    string
		kvFirst bool
		shape   []uint64
		wantK   []float32
		wantV   []float32
	}{
		{
			name:  "kv last",
			shape: []uint64{6, 3},
			wantK: []float32{0, 3, 1, 4, 2, 5, 9, 12, 10, 13, 11, 14},
			wantV: []float32{6, 7, 8, 15, 16, 17},
		},
		{
			name:    "kv first",
			kvFirst: true,
			shape:   []uint64{3, 6},
			wantK:   []float32{0, 1, 6, 7, 12, 13, 3, 4, 9, 10, 15, 16},
			wantV:   []float32{2, 8, 14, 5, 11, 17},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			gotK, err := glm47FlashRepackKVB(mla, true, tc.kvFirst, mla.numHeads)("kv_b", data, tc.shape)
			if err != nil {
				t.Fatalf("repack K error = %v", err)
			}
			if !slices.Equal(gotK, tc.wantK) {
				t.Errorf("repack K\n got %v\nwant %v", gotK, tc.wantK)
			}

			gotV, err := glm47FlashRepackKVB(mla, false, tc.kvFirst, mla.numHeads)("kv_b", data, tc.shape)
			if err != nil {
				t.Fatalf("repack V error = %v", err)
			}
			if !slices.Equal(gotV, tc.wantV) {
				t.Errorf("repack V\n got %v\nwant %v", gotV, tc.wantV)
			}
		})
	}

	if _, err := glm47FlashRepackKVB(mla, true, false, mla.numHeads)("kv_b", sequence(12), []uint64{4, 3}); err == nil {
		t.Error("expected error when the tensor does not fill the MLA dims")
	}
	if _, err := glm47FlashRepackKVB(mla, true, true, mla.numHeads)("kv_b", data, []uint64{3, 3, 2}); err == nil {
		t.Error("expected error transposing a 3D kv-first tensor")
	}
}

func TestGLM47FlashSplitKVB(t *testing.T) {
	mla := glm47FlashMLA{numHeads: 2, qkNope: 2, vHeadDim: 1, kvLoraRank: 3, qkRope: 4}
	data := sequence(18)

	cases := []struct {
		name  string
		shape []uint64
		wantK []float32
		wantV []float32
	}{
		{
			name:  "kv last",
			shape: []uint64{6, 3},
			wantK: []float32{0, 3, 1, 4, 2, 5, 9, 12, 10, 13, 11, 14},
			wantV: []float32{6, 7, 8, 15, 16, 17},
		},
		{
			name:  "kv first",
			shape: []uint64{3, 6},
			wantK: []float32{0, 1, 6, 7, 12, 13, 3, 4, 9, 10, 15, 16},
			wantV: []float32{2, 8, 14, 5, 11, 17},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			source := sourceTensorF16("blk.0.attn_kv_b.weight", tc.shape, data)
			got, err := glm47FlashSplitKVB(source, mla)
			if err != nil {
				t.Fatalf("glm47FlashSplitKVB() error = %v", err)
			}
			if len(got) != 2 {
				t.Fatalf("expected k and v tensors, got %d", len(got))
			}

			k, v := got[0], got[1]
			if k.Name != "blk.0.attn_k_b.weight" || v.Name != "blk.0.attn_v_b.weight" {
				t.Errorf("unexpected names %q %q", k.Name, v.Name)
			}
			if !slices.Equal(k.Shape, []uint64{2, 3, 2}) {
				t.Errorf("k shape = %v, want [2 3 2]", k.Shape)
			}
			if !slices.Equal(v.Shape, []uint64{3, 1, 2}) {
				t.Errorf("v shape = %v, want [3 1 2]", v.Shape)
			}
			if k.Kind != uint32(gguf.TensorTypeF16) || v.Kind != uint32(gguf.TensorTypeF16) {
				t.Errorf("unexpected kinds %d %d", k.Kind, v.Kind)
			}
			if values := writeTensorF16(t, k); !slices.Equal(values, tc.wantK) {
				t.Errorf("k values\n got %v\nwant %v", values, tc.wantK)
			}
			if values := writeTensorF16(t, v); !slices.Equal(values, tc.wantV) {
				t.Errorf("v values\n got %v\nwant %v", values, tc.wantV)
			}
		})
	}

	if _, err := glm47FlashSplitKVB(sourceTensorF16("blk.0.attn_kv_b.weight", []uint64{9, 2}, data), mla); err == nil {
		t.Error("expected error when neither dimension matches kv_lora_rank")
	}
	if _, err := glm47FlashSplitKVB(sourceTensorF16("blk.0.attn_kv_b.weight", []uint64{3, 3, 2}, data), mla); err == nil {
		t.Error("expected error for a 3D source tensor")
	}
}
