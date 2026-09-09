package ggml

import (
	"encoding/binary"
	"maps"
	"math"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/ml"
)

func TestTensorLayers(t *testing.T) {
	tensors := make(map[string]*Tensor)
	for _, name := range []string{
		"token_embd.weight",
		"blk.0.attn_k.weight",
		"blk.0.attn_output.weight",
		"blk.0.attn_q.weight",
		"blk.0.attn_v.weight",
		"blk.0.attn_norm.weight",
		"blk.0.ffn_down.weight",
		"blk.0.ffn_gate.weight",
		"blk.0.ffn_up.weight",
		"blk.0.ffn_norm.weight",
		"output_norm.weight",
		"mm.0.bias",
		"mm.0.weight",
		"v.blk.0.attn_k.weight",
		"v.blk.0.attn_output.weight",
		"v.blk.0.attn_q.weight",
		"v.blk.0.attn_v.weight",
		"v.blk.0.attn_norm.weight",
		"v.blk.0.ffn_down.weight",
		"v.blk.0.ffn_gate.weight",
		"v.blk.0.ffn_up.weight",
		"v.blk.0.ffn_norm.weight",
		"v.patch_embd.weight",
		"v.position_embd.gate",
		"v.position_embd.weight",
	} {
		tensors[name] = &Tensor{Name: name}
	}

	cases := []struct {
		name  string
		items []*Tensor
		want  map[string]Layer
	}{
		{
			name: "text",
			items: slices.Collect(func(yield func(*Tensor) bool) {
				for k, v := range tensors {
					if !strings.HasPrefix(k, "mm.") && !strings.HasPrefix(k, "v.") {
						if !yield(v) {
							return
						}
					}
				}
			}),
			want: map[string]Layer{
				"blk.0": {
					"attn_k.weight":      tensors["blk.0.attn_k.weight"],
					"attn_q.weight":      tensors["blk.0.attn_q.weight"],
					"attn_v.weight":      tensors["blk.0.attn_v.weight"],
					"attn_output.weight": tensors["blk.0.attn_output.weight"],
					"attn_norm.weight":   tensors["blk.0.attn_norm.weight"],
					"ffn_down.weight":    tensors["blk.0.ffn_down.weight"],
					"ffn_gate.weight":    tensors["blk.0.ffn_gate.weight"],
					"ffn_up.weight":      tensors["blk.0.ffn_up.weight"],
					"ffn_norm.weight":    tensors["blk.0.ffn_norm.weight"],
				},
				"token_embd":  {"weight": tensors["token_embd.weight"]},
				"output_norm": {"weight": tensors["output_norm.weight"]},
			},
		},
		{
			name: "vision",
			items: slices.Collect(func(yield func(*Tensor) bool) {
				for k, v := range tensors {
					if strings.HasPrefix(k, "mm.") || strings.HasPrefix(k, "v.") {
						if !yield(v) {
							return
						}
					}
				}
			}),
			want: map[string]Layer{
				"mm.0": {
					"bias":   tensors["mm.0.bias"],
					"weight": tensors["mm.0.weight"],
				},
				"v.blk.0": {
					"attn_k.weight":      tensors["v.blk.0.attn_k.weight"],
					"attn_q.weight":      tensors["v.blk.0.attn_q.weight"],
					"attn_v.weight":      tensors["v.blk.0.attn_v.weight"],
					"attn_output.weight": tensors["v.blk.0.attn_output.weight"],
					"attn_norm.weight":   tensors["v.blk.0.attn_norm.weight"],
					"ffn_down.weight":    tensors["v.blk.0.ffn_down.weight"],
					"ffn_gate.weight":    tensors["v.blk.0.ffn_gate.weight"],
					"ffn_up.weight":      tensors["v.blk.0.ffn_up.weight"],
					"ffn_norm.weight":    tensors["v.blk.0.ffn_norm.weight"],
				},
				"v": {
					"patch_embd.weight":    tensors["v.patch_embd.weight"],
					"position_embd.gate":   tensors["v.position_embd.gate"],
					"position_embd.weight": tensors["v.position_embd.weight"],
				},
			},
		},
		{
			name:  "vision and text",
			items: slices.Collect(maps.Values(tensors)),
			want: map[string]Layer{
				"blk.0": {
					"attn_k.weight":      tensors["blk.0.attn_k.weight"],
					"attn_q.weight":      tensors["blk.0.attn_q.weight"],
					"attn_v.weight":      tensors["blk.0.attn_v.weight"],
					"attn_output.weight": tensors["blk.0.attn_output.weight"],
					"attn_norm.weight":   tensors["blk.0.attn_norm.weight"],
					"ffn_down.weight":    tensors["blk.0.ffn_down.weight"],
					"ffn_gate.weight":    tensors["blk.0.ffn_gate.weight"],
					"ffn_up.weight":      tensors["blk.0.ffn_up.weight"],
					"ffn_norm.weight":    tensors["blk.0.ffn_norm.weight"],
				},
				"token_embd":  {"weight": tensors["token_embd.weight"]},
				"output_norm": {"weight": tensors["output_norm.weight"]},
				"mm.0": {
					"bias":   tensors["mm.0.bias"],
					"weight": tensors["mm.0.weight"],
				},
				"v.blk.0": {
					"attn_k.weight":      tensors["v.blk.0.attn_k.weight"],
					"attn_q.weight":      tensors["v.blk.0.attn_q.weight"],
					"attn_v.weight":      tensors["v.blk.0.attn_v.weight"],
					"attn_output.weight": tensors["v.blk.0.attn_output.weight"],
					"attn_norm.weight":   tensors["v.blk.0.attn_norm.weight"],
					"ffn_down.weight":    tensors["v.blk.0.ffn_down.weight"],
					"ffn_gate.weight":    tensors["v.blk.0.ffn_gate.weight"],
					"ffn_up.weight":      tensors["v.blk.0.ffn_up.weight"],
					"ffn_norm.weight":    tensors["v.blk.0.ffn_norm.weight"],
				},
				"v": {
					"patch_embd.weight":    tensors["v.patch_embd.weight"],
					"position_embd.gate":   tensors["v.position_embd.gate"],
					"position_embd.weight": tensors["v.position_embd.weight"],
				},
			},
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			got := Tensors{items: tt.items}.GroupLayers()
			if diff := cmp.Diff(got, tt.want); diff != "" {
				t.Errorf("unexpected layers (-got +want):\n%s", diff)
			}
		})
	}
}

// ref: https://github.com/ggml-org/llama.cpp/blob/a82c9e7c23ef6db48cebfa194dc9cebbc4ac3552/ggml/src/ggml.c#L572
func TestTensorTypes(t *testing.T) {
	cases := []struct {
		kind      uint32
		blockSize uint64
		typeSize  uint64
	}{
		{0, 1, 4},
		{1, 1, 2},
		{2, 32, 18},
		{3, 32, 20},
		{6, 32, 22},
		{7, 32, 24},
		{8, 32, 34},
		{9, 32, 36},
		{10, 256, 84},
		{11, 256, 110},
		{12, 256, 144},
		{13, 256, 176},
		{14, 256, 210},
		{15, 256, 292},
		{16, 256, 66},
		{17, 256, 74},
		{18, 256, 98},
		{19, 256, 50},
		{20, 32, 18},
		{21, 256, 110},
		{22, 256, 82},
		{23, 256, 136},
		{24, 1, 1},
		{25, 1, 2},
		{26, 1, 4},
		{27, 1, 8},
		{28, 1, 8},
		{29, 256, 56},
		{30, 1, 2},
	}

	for _, tt := range cases {
		t.Run(strconv.Itoa(int(tt.kind)), func(t *testing.T) {
			tensor := Tensor{Kind: tt.kind}
			if tensor.blockSize() != tt.blockSize {
				t.Errorf("unexpected block size: got=%d want=%d", tensor.blockSize(), tt.blockSize)
			}

			if tensor.typeSize() != tt.typeSize {
				t.Errorf("unexpected type size: got=%d want=%d", tensor.typeSize(), tt.typeSize)
			}
		})
	}
}

func TestKeyValue(t *testing.T) {
	kv := KV{
		"general.architecture": "test",
		"test.strings":         &array[string]{size: 3, values: []string{"a", "b", "c"}},
		"test.float32s":        &array[float32]{size: 3, values: []float32{1.0, 2.0, 3.0}},
		"test.int32s":          &array[int32]{size: 3, values: []int32{1, 2, 3}},
		"test.uint32s":         &array[uint32]{size: 3, values: []uint32{1, 2, 3}},
	}

	if diff := cmp.Diff(kv.Strings("strings"), []string{"a", "b", "c"}); diff != "" {
		t.Errorf("unexpected strings (-got +want):\n%s", diff)
	}

	if diff := cmp.Diff(kv.Strings("nonexistent.strings"), []string(nil)); diff != "" {
		t.Errorf("unexpected strings (-got +want):\n%s", diff)
	}

	if diff := cmp.Diff(kv.Strings("default.strings", []string{"ollama"}), []string{"ollama"}); diff != "" {
		t.Errorf("unexpected strings (-got +want):\n%s", diff)
	}

	if diff := cmp.Diff(kv.Floats("float32s"), []float32{1.0, 2.0, 3.0}); diff != "" {
		t.Errorf("unexpected float32s (-got +want):\n%s", diff)
	}

	if diff := cmp.Diff(kv.Floats("nonexistent.float32s"), []float32(nil)); diff != "" {
		t.Errorf("unexpected float32s (-got +want):\n%s", diff)
	}

	if diff := cmp.Diff(kv.Floats("default.float32s", []float32{math.MaxFloat32}), []float32{math.MaxFloat32}); diff != "" {
		t.Errorf("unexpected float32s (-got +want):\n%s", diff)
	}

	if diff := cmp.Diff(kv.Ints("int32s"), []int32{1, 2, 3}); diff != "" {
		t.Errorf("unexpected int8s (-got +want):\n%s", diff)
	}

	if diff := cmp.Diff(kv.Ints("nonexistent.int32s"), []int32(nil)); diff != "" {
		t.Errorf("unexpected int8s (-got +want):\n%s", diff)
	}

	if diff := cmp.Diff(kv.Ints("default.int32s", []int32{math.MaxInt32}), []int32{math.MaxInt32}); diff != "" {
		t.Errorf("unexpected int8s (-got +want):\n%s", diff)
	}

	if diff := cmp.Diff(kv.Uints("uint32s"), []uint32{1, 2, 3}); diff != "" {
		t.Errorf("unexpected uint8s (-got +want):\n%s", diff)
	}

	if diff := cmp.Diff(kv.Uints("nonexistent.uint32s"), []uint32(nil)); diff != "" {
		t.Errorf("unexpected uint8s (-got +want):\n%s", diff)
	}

	if diff := cmp.Diff(kv.Uints("default.uint32s", []uint32{math.MaxUint32}), []uint32{math.MaxUint32}); diff != "" {
		t.Errorf("unexpected uint8s (-got +want):\n%s", diff)
	}
}

func TestHeadCount(t *testing.T) {
	valuesArray := []int32{1, 5, 3, 4}
	cases := []struct {
		kv   KV
		want uint64
	}{
		{
			kv: KV{
				"general.architecture":     "abc",
				"abc.attention.head_count": &array[int32]{values: valuesArray, size: len(valuesArray)},
			},
			want: uint64(5),
		},
		{
			kv: KV{
				"general.architecture":     "abc",
				"abc.attention.head_count": uint32(3),
			},
			want: uint64(3),
		},
	}

	for _, tt := range cases {
		got := tt.kv.HeadCountMax()
		if got != tt.want {
			t.Errorf("unexpected max value: got=%d want=%d", got, tt.want)
		}
	}
}

// makeGGML constructs a minimal GGML for unit-testing GraphSize without a file round-trip.
func makeGGML(kv KV) GGML {
	g := newGGUF(&containerGGUF{ByteOrder: binary.LittleEndian, Version: 3})
	for k, v := range kv {
		g.kv[k] = v
	}
	return GGML{container: g.containerGGUF, model: g}
}

// TestGraphSizeSlidingWindowPattern verifies that the generic SWA correction in
// GraphSize uses the per-layer bool array (attention.sliding_window_pattern) to
// reduce KV cache estimates for sliding-window layers. Without the fix, Gemma 4
// at 8k context over-estimates by ~7× (3.52 GB vs 0.48 GB), starving GPU expert
// placement.
func TestGraphSizeSlidingWindowPattern(t *testing.T) {
	// Mirrors the actual Gemma 4 26B GGUF metadata (30 layers, every 6th is full-attention).
	const (
		nLayers       = 30
		slidingWindow = uint32(1024)
		keyLenFull    = uint32(512)
		valLenFull    = uint32(512)
		keyLenSwa     = uint32(256)
		valLenSwa     = uint32(256)
	)

	// sliding_window_pattern: true = sliding layer, false = full-attention (every 6th).
	swPattern := make([]bool, nLayers)
	hkvArr := make([]uint32, nLayers)
	hArr := make([]uint32, nLayers)
	for i := range nLayers {
		swPattern[i] = (i+1)%6 != 0
		if !swPattern[i] {
			hkvArr[i] = 2 // full-attention layers have fewer KV heads in Gemma 4
		} else {
			hkvArr[i] = 8
		}
		hArr[i] = 16
	}

	kv := KV{
		"general.architecture":                    "gemma4",
		"tokenizer.ggml.tokens":                   &array[string]{size: 256000},
		"gemma4.block_count":                      uint32(nLayers),
		"gemma4.embedding_length":                 uint32(2816),
		"gemma4.attention.head_count":             &array[uint32]{values: hArr, size: nLayers},
		"gemma4.attention.head_count_kv":          &array[uint32]{values: hkvArr, size: nLayers},
		"gemma4.attention.key_length":             keyLenFull,
		"gemma4.attention.value_length":           valLenFull,
		"gemma4.attention.key_length_swa":         keyLenSwa,
		"gemma4.attention.value_length_swa":       valLenSwa,
		"gemma4.attention.sliding_window":         slidingWindow,
		"gemma4.attention.sliding_window_pattern": &array[bool]{values: swPattern, size: nLayers},
		"gemma4.context_length":                   uint32(262144),
		"gemma4.attention.layer_norm_rms_epsilon": float32(1e-6),
	}

	contexts := []struct {
		ctx     uint64
		maxKVGB float64 // generous upper bound: correct answer must be below this
		minKVGB float64 // must be above this (guards against zero/underflow)
	}{
		{4096, 0.6, 0.1},
		{8192, 0.7, 0.1},
		{32768, 1.5, 0.5},
		{131072, 4.0, 2.0},
	}

	g := makeGGML(kv)
	for _, tc := range contexts {
		kvSizes, _, _ := g.GraphSize(tc.ctx, 512, 1, "f16", ml.FlashAttentionDisabled)
		var total uint64
		for _, s := range kvSizes {
			total += s
		}
		totalGB := float64(total) / 1e9
		if totalGB > tc.maxKVGB {
			t.Errorf("ctx=%d: KV estimate %.3f GB exceeds max %.3f GB (SWA correction not applied?)",
				tc.ctx, totalGB, tc.maxKVGB)
		}
		if totalGB < tc.minKVGB {
			t.Errorf("ctx=%d: KV estimate %.3f GB below min %.3f GB (underflow?)",
				tc.ctx, totalGB, tc.minKVGB)
		}
	}

	// Regression: a model without sliding_window_pattern must be unaffected.
	kvNoSWA := KV{
		"general.architecture":                   "llama",
		"tokenizer.ggml.tokens":                  &array[string]{size: 32000},
		"llama.block_count":                      uint32(4),
		"llama.embedding_length":                 uint32(4096),
		"llama.attention.head_count":             uint32(32),
		"llama.attention.head_count_kv":          uint32(8),
		"llama.context_length":                   uint32(4096),
		"llama.attention.layer_norm_rms_epsilon": float32(1e-5),
	}
	gNoSWA := makeGGML(kvNoSWA)
	kvSizesNoSWA, _, _ := gNoSWA.GraphSize(4096, 512, 1, "f16", ml.FlashAttentionDisabled)
	for i, s := range kvSizesNoSWA {
		if s == 0 {
			t.Errorf("non-SWA model: layer %d KV size is 0, expected non-zero", i)
		}
	}
}
