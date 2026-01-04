package ggml

import (
	"maps"
	"math"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
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
