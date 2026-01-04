package server

import (
	"bytes"
	"fmt"
	"math"
	"os"
	"strings"
	"testing"

	fsggml "github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/ml/backend/ggml"
)

func TestGetTensorNewType(t *testing.T) {
	cases := []struct {
		name          string
		kv            map[string]any
		qs            quantizeState
		newType       fsggml.TensorType
		tensor_name   string
		shape         []uint64
		ftype         fsggml.FileType
		expected      fsggml.TensorType
		expectedPanic string
	}{
		{
			name:        "output_unsupported",
			kv:          map[string]any{},
			newType:     fsggml.TensorTypeQ4_0,
			tensor_name: "output.weight",
			shape:       []uint64{100, 100},
			ftype:       fsggml.FileTypeF32,
			expected:    fsggml.TensorTypeF16,
		},
		{
			name:        "output_Q8",
			kv:          map[string]any{},
			newType:     fsggml.TensorTypeQ4_0,
			tensor_name: "output.weight",
			shape:       []uint64{1024, 1024},
			ftype:       fsggml.FileTypeF32,
			expected:    fsggml.TensorTypeQ6_K,
		},
		{
			name: "attn_v.weight_q4_k_m",
			qs: quantizeState{
				iAttnV: 2,
				nAttnV: 3 * 8,
			},
			kv:          map[string]any{},
			newType:     fsggml.TensorTypeQ4_0,
			tensor_name: "blk.0.attn_v.weight",
			shape:       []uint64{256},
			ftype:       fsggml.FileTypeQ4_K_M,
			expected:    fsggml.TensorTypeQ6_K,
		},
		{
			name:        "attn_v.weight_q4_k_s",
			qs:          quantizeState{},
			kv:          map[string]any{},
			newType:     fsggml.TensorTypeQ4_0,
			tensor_name: "blk.0.attn_v.weight",
			shape:       []uint64{256},
			ftype:       fsggml.FileTypeQ4_K_S,
			expected:    fsggml.TensorTypeQ5_K,
		},
		{
			name: "attn_v.weight_8_expert",
			qs:   quantizeState{},
			kv: map[string]any{
				"general.architecture": "foo",
				"foo.expert_count":     uint32(8),
			},
			newType:     fsggml.TensorTypeQ4_0,
			tensor_name: "blk.0.attn_v.weight",
			shape:       []uint64{256},
			ftype:       fsggml.FileTypeF32,
			expected:    fsggml.TensorTypeQ8_0,
		},
		{
			name: "attn_k.weight_8_expert",
			qs:   quantizeState{},
			kv: map[string]any{
				"general.architecture": "foo",
				"foo.expert_count":     uint32(8),
			},
			newType:     fsggml.TensorTypeQ4_0,
			tensor_name: "blk.0.attn_k.weight",
			shape:       []uint64{256},
			ftype:       fsggml.FileTypeF32,
			expected:    fsggml.TensorTypeQ8_0,
		},
		{
			name: "ffn_down_q4_k_m",
			qs: quantizeState{
				iFfnDown: 1,
				nFfnDown: 8,
			},
			kv:          map[string]any{},
			newType:     fsggml.TensorTypeQ4_0,
			tensor_name: "ffn_down",
			shape:       []uint64{256},
			ftype:       fsggml.FileTypeQ4_K_M,
			expected:    fsggml.TensorTypeQ4_0,
		},
		{
			name: "ffn_down_q4_k_m_6",
			qs: quantizeState{
				iFfnDown: 2,
				nFfnDown: 3 * 8,
			},
			kv:          map[string]any{},
			newType:     fsggml.TensorTypeQ4_0,
			tensor_name: "ffn_down",
			shape:       []uint64{256},
			ftype:       fsggml.FileTypeQ4_K_M,
			expected:    fsggml.TensorTypeQ6_K,
		},
		{
			name: "ffn_down_q4_k_s",
			qs: quantizeState{
				iFfnDown: 2,
				nFfnDown: 3 * 8,
			},
			kv:          map[string]any{},
			newType:     fsggml.TensorTypeQ4_0,
			tensor_name: "ffn_down",
			shape:       []uint64{256},
			ftype:       fsggml.FileTypeQ4_K_S,
			expected:    fsggml.TensorTypeQ5_K,
		},
		{
			name:        "attn_qkv.weight_q4_k_m",
			qs:          quantizeState{},
			kv:          map[string]any{},
			newType:     fsggml.TensorTypeQ4_0,
			tensor_name: "blk.0.attn_qkv.weight",
			shape:       []uint64{256},
			ftype:       fsggml.FileTypeQ4_K_M,
			expected:    fsggml.TensorTypeQ5_K,
		},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			if tt.expectedPanic != "" {
				defer func() {
					e := recover()
					if !strings.Contains(fmt.Sprintf("%v", e), tt.expectedPanic) {
						t.Fatalf("incorrect panic\ngot: %v\nexpected: %s", e, tt.expectedPanic)
					}
				}()
			} else {
				defer func() {
					e := recover()
					if e != nil {
						t.Fatalf("hit unexpected panic %v", e)
					}
				}()
			}
			ret := getTensorNewType(tt.kv, &tt.qs, tt.newType, tt.tensor_name, tt.shape, tt.ftype)
			if ret != tt.expected {
				t.Fatalf("incorrect type returned\ngot: %d\nexpected: %d", ret, tt.expected)
			}
		})
	}
}

func TestQuantizeModel(t *testing.T) {
	cases := []struct {
		name                string
		kv                  map[string]any
		tensors             []*fsggml.Tensor
		newType             string
		expectedTensorTypes map[string]fsggml.TensorType
	}{
		{
			name: "f16_q4_k",
			kv: map[string]any{
				"general.architecture": "foo",
			},
			tensors: []*fsggml.Tensor{
				{
					Name: "blk.0.attn.weight", Kind: uint32(fsggml.TensorTypeF16),
					Offset: uint64(0), Shape: []uint64{512, 2},
					WriterTo: bytes.NewReader(
						append(append(append(quantBytes[fsggml.TensorTypeF16], quantBytes[fsggml.TensorTypeF16]...), quantBytes[fsggml.TensorTypeF16]...), quantBytes[fsggml.TensorTypeF16]...),
					),
				},
				{
					Name: "output.weight", Kind: uint32(fsggml.TensorTypeF16),
					Offset: uint64(0), Shape: []uint64{256, 4},
					WriterTo: bytes.NewReader(
						append(append(append(quantBytes[fsggml.TensorTypeF16], quantBytes[fsggml.TensorTypeF16]...), quantBytes[fsggml.TensorTypeF16]...), quantBytes[fsggml.TensorTypeF16]...),
					),
				},
			},
			newType: "Q4_K",
			expectedTensorTypes: map[string]fsggml.TensorType{
				"blk.0.attn.weight": fsggml.TensorTypeQ4_K,
				"output.weight":     fsggml.TensorTypeQ6_K,
			},
		},
		{
			name: "f32_q4_k",
			kv: map[string]any{
				"general.architecture": "foo",
			},
			tensors: []*fsggml.Tensor{
				{
					Name: "blk.0.attn_v.weight", Kind: uint32(fsggml.TensorTypeF32),
					Offset: uint64(0), Shape: []uint64{512, 2},
					WriterTo: bytes.NewReader(
						append(append(append(quantBytes[fsggml.TensorTypeF32], quantBytes[fsggml.TensorTypeF32]...), quantBytes[fsggml.TensorTypeF32]...), quantBytes[fsggml.TensorTypeF32]...),
					),
				},
				{
					Name: "output.weight", Kind: uint32(fsggml.TensorTypeF32),
					Offset: uint64(0), Shape: []uint64{512},
					WriterTo: bytes.NewReader(append(quantBytes[fsggml.TensorTypeF32], quantBytes[fsggml.TensorTypeF32]...)),
				},
			},
			newType: "Q4_K",
			expectedTensorTypes: map[string]fsggml.TensorType{
				"blk.0.attn_v.weight": fsggml.TensorTypeQ6_K,
				"output.weight":       fsggml.TensorTypeF32,
			},
		},
		{
			name: "f16_q8_0",
			kv: map[string]any{
				"general.architecture": "foo",
			},
			tensors: []*fsggml.Tensor{
				{
					Name: "blk.0.attn.weight", Kind: uint32(fsggml.TensorTypeF16),
					Offset: uint64(0), Shape: []uint64{32, 16, 2},
					WriterTo: bytes.NewReader(
						append(append(append(quantBytes[fsggml.TensorTypeF16], quantBytes[fsggml.TensorTypeF16]...), quantBytes[fsggml.TensorTypeF16]...), quantBytes[fsggml.TensorTypeF16]...),
					),
				},
				{
					Name: "output.weight", Kind: uint32(fsggml.TensorTypeF16),
					Offset: uint64(0), Shape: []uint64{256, 4},
					WriterTo: bytes.NewReader(
						append(append(append(quantBytes[fsggml.TensorTypeF16], quantBytes[fsggml.TensorTypeF16]...), quantBytes[fsggml.TensorTypeF16]...), quantBytes[fsggml.TensorTypeF16]...),
					),
				},
			},
			newType: "Q8_0",
			expectedTensorTypes: map[string]fsggml.TensorType{
				"blk.0.attn.weight": fsggml.TensorTypeQ8_0,
				"output.weight":     fsggml.TensorTypeQ8_0,
			},
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			p, _ := createBinFile(t, tt.kv, tt.tensors)
			fp, err := os.Open(p)
			if err != nil {
				t.Fatal(err.Error())
			}
			defer fp.Close()
			meta, err := fsggml.Decode(fp, -1)
			if err != nil {
				t.Fatal(err.Error())
			}
			progressCalled := false
			progress := func(n uint64) {
				// fmt.Fprintf(os.Stderr, "progress: %f\n", p)
				progressCalled = true
			}
			tmp, err := os.CreateTemp(t.TempDir(), tt.name+".out")
			if err != nil {
				t.Fatal(err.Error())
			}
			defer tmp.Close()
			ftype, err := fsggml.ParseFileType(tt.newType)
			if err != nil {
				t.Fatal(err.Error())
			}

			err = quantize(fp, tmp, meta, ftype, progress)
			if err != nil {
				t.Fatalf("error during quantize: %s", err)
			}
			if !progressCalled {
				t.Fatalf("progress was not reported")
			}
			// Now attempt to load it back and make sure types match expected
			fpNew, err := os.Open(tmp.Name())
			if err != nil {
				t.Fatalf("failed to load the quantized model %s: %s", tmp.Name(), err)
			}
			defer fpNew.Close()
			newMeta, err := fsggml.Decode(fpNew, -1)
			if err != nil {
				t.Fatalf("failed to load the quantized model %s: %s", tmp.Name(), err)
			}
			tensors := newMeta.Tensors()
			for _, l := range tensors.GroupLayers() {
				for _, tensor := range l {
					if fsggml.TensorType(tensor.Kind) != tt.expectedTensorTypes[tensor.Name] {
						t.Fatalf("incorrect output type for %s\ngot:%s\nexpected:%s", tensor.Name, fsggml.TensorType(tensor.Kind), tt.expectedTensorTypes[tensor.Name])
					}
				}
			}
		})
	}
}

func TestConvertToF32(t *testing.T) {
	expected := make([]float32, 256)
	for i := range expected {
		expected[i] = float32(i)
	}
	for dtype, data := range quantBytes {
		// Skip the no-op
		if dtype == fsggml.TensorTypeF32 {
			continue
		}
		t.Run(dtype.String(), func(t *testing.T) {
			fp32 := ggml.ConvertToF32(data, uint32(dtype), 256)
			similarity := cosineSimilarity(expected, fp32)
			if similarity < 0.999 {
				t.Fatalf("Results not similar enough: %s %f", dtype.String(), similarity)
			}
		})
	}
}

func dotProduct[V float32 | float64](v1, v2 []V) V {
	var result V = 0
	for i := range v1 {
		result += v1[i] * v2[i]
	}
	return result
}

func magnitude[V float32 | float64](v []V) V {
	var result V = 0
	for _, val := range v {
		result += val * val
	}
	return V(math.Sqrt(float64(result)))
}

func cosineSimilarity[V float32 | float64](v1, v2 []V) V {
	return dotProduct(v1, v2) / (magnitude(v1) * magnitude(v2))
}

// Precomputed quantized data - arange 256
// # For gguf-py supported types
// import gguf
// import numpy as np
// print(repr(gguf.quantize(np.arange(256, dtype=np.float16), gguf.GGMLQuantizationType.Q4_0)))
//
// For types not supported by gguf-py converted via ggml_fp32_to_fp16_row and quantize_XXX
//
//	data := make([]byte, 256*2)
//	fp32 := make([]float32, 256)
//	for i := range 256 {
//		fp32[i] = float32(i)
//	}
//	l := C.quantize_q6_K((*C.float)(&fp32[0]), unsafe.Pointer(&data[0]), 1, 256, nil)
//	for i := range data[:int(l)] {
//		fmt.Printf("%d, ", data[i])
//	}
var (
	quantBytes = map[fsggml.TensorType][]byte{
		fsggml.TensorTypeQ4_0: {
			192, 195, 72, 72, 55, 55, 55, 55, 38, 38, 38, 38, 21,
			21, 21, 21, 4, 4, 224, 199, 36, 36, 36, 36, 19, 19,
			19, 19, 19, 19, 19, 19, 2, 2, 2, 2, 240, 201, 19,
			19, 18, 18, 18, 18, 18, 18, 18, 18, 2, 2, 2, 2,
			1, 1, 240, 203, 18, 18, 18, 18, 18, 18, 18, 18, 1,
			1, 1, 1, 1, 1, 1, 1, 248, 204, 18, 18, 17, 17,
			17, 17, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 248,
			205, 17, 17, 17, 17, 1, 1, 1, 1, 1, 1, 1, 1,
			1, 1, 1, 1, 248, 206, 17, 17, 1, 1, 1, 1, 1,
			1, 1, 1, 1, 1, 1, 1, 1, 1, 248, 207, 1, 1,
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
			1,
		},
		fsggml.TensorTypeQ4_1: {
			34, 64, 0, 0, 128, 128, 145, 145, 162, 162, 179, 179, 196,
			196, 213, 213, 230, 230, 247, 247, 34, 64, 0, 80, 128, 128,
			145, 145, 162, 162, 179, 179, 196, 196, 213, 213, 230, 230, 247,
			247, 34, 64, 0, 84, 128, 128, 145, 145, 162, 162, 179, 179,
			196, 196, 213, 213, 230, 230, 247, 247, 34, 64, 0, 86, 128,
			128, 145, 145, 162, 162, 179, 179, 196, 196, 213, 213, 230, 230,
			247, 247, 34, 64, 0, 88, 128, 128, 145, 145, 162, 162, 179,
			179, 196, 196, 213, 213, 230, 230, 247, 247, 34, 64, 0, 89,
			128, 128, 145, 145, 162, 162, 179, 179, 196, 196, 213, 213, 230,
			230, 247, 247, 34, 64, 0, 90, 128, 128, 145, 145, 162, 162,
			179, 179, 196, 196, 213, 213, 230, 230, 247, 247, 34, 64, 0,
			91, 128, 128, 145, 145, 162, 162, 179, 179, 196, 196, 213, 213,
			230, 230, 247, 247,
		},
		fsggml.TensorTypeQ5_0: {
			192, 191, 1, 0, 0, 0, 128, 127, 127, 110, 110, 93, 93,
			76, 76, 59, 59, 42, 42, 25, 25, 8, 224, 195, 0, 0,
			0, 0, 72, 72, 55, 55, 55, 55, 38, 38, 38, 38, 21,
			21, 21, 21, 4, 4, 240, 197, 0, 0, 0, 0, 53, 37,
			37, 37, 37, 36, 36, 20, 20, 20, 20, 19, 19, 3, 3,
			3, 240, 199, 0, 0, 0, 0, 36, 36, 36, 36, 19, 19,
			19, 19, 19, 19, 19, 19, 2, 2, 2, 2, 248, 200, 0,
			0, 0, 0, 35, 19, 19, 19, 19, 19, 19, 18, 18, 18,
			18, 2, 2, 2, 2, 2, 248, 201, 0, 0, 0, 0, 19,
			19, 18, 18, 18, 18, 18, 18, 18, 18, 2, 2, 2, 2,
			1, 1, 248, 202, 0, 0, 0, 0, 18, 18, 18, 18, 18,
			18, 18, 18, 18, 2, 2, 1, 1, 1, 1, 1, 248, 203,
			0, 0, 0, 0, 18, 18, 18, 18, 18, 18, 18, 18, 1,
			1, 1, 1, 1, 1, 1, 1,
		},
		fsggml.TensorTypeQ5_1: {
			0, 60, 0, 0, 0, 0, 255, 255, 0, 17, 34, 51, 68,
			85, 102, 119, 136, 153, 170, 187, 204, 221, 238, 255, 0, 60,
			0, 80, 0, 0, 255, 255, 0, 17, 34, 51, 68, 85, 102,
			119, 136, 153, 170, 187, 204, 221, 238, 255, 0, 60, 0, 84,
			0, 0, 255, 255, 0, 17, 34, 51, 68, 85, 102, 119, 136,
			153, 170, 187, 204, 221, 238, 255, 0, 60, 0, 86, 0, 0,
			255, 255, 0, 17, 34, 51, 68, 85, 102, 119, 136, 153, 170,
			187, 204, 221, 238, 255, 0, 60, 0, 88, 0, 0, 255, 255,
			0, 17, 34, 51, 68, 85, 102, 119, 136, 153, 170, 187, 204,
			221, 238, 255, 0, 60, 0, 89, 0, 0, 255, 255, 0, 17,
			34, 51, 68, 85, 102, 119, 136, 153, 170, 187, 204, 221, 238,
			255, 0, 60, 0, 90, 0, 0, 255, 255, 0, 17, 34, 51,
			68, 85, 102, 119, 136, 153, 170, 187, 204, 221, 238, 255, 0,
			60, 0, 91, 0, 0, 255, 255, 0, 17, 34, 51, 68, 85,
			102, 119, 136, 153, 170, 187, 204, 221, 238, 255,
		},
		fsggml.TensorTypeQ8_0: {
			208, 51, 0, 4, 8, 12, 16, 20, 25, 29, 33, 37, 41,
			45, 49, 53, 57, 61, 66, 70, 74, 78, 82, 86, 90, 94,
			98, 102, 107, 111, 115, 119, 123, 127, 240, 55, 65, 67, 69,
			71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95,
			97, 99, 101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 121,
			123, 125, 127, 252, 57, 86, 87, 88, 90, 91, 92, 94, 95,
			96, 98, 99, 100, 102, 103, 104, 106, 107, 108, 110, 111, 112,
			114, 115, 116, 118, 119, 120, 122, 123, 124, 126, 127, 0, 60,
			96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108,
			109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121,
			122, 123, 124, 125, 126, 127, 2, 61, 102, 103, 104, 105, 105,
			106, 107, 108, 109, 109, 110, 111, 112, 113, 113, 114, 115, 116,
			117, 117, 118, 119, 120, 121, 121, 122, 123, 124, 125, 125, 126,
			127, 4, 62, 106, 107, 108, 108, 109, 110, 110, 111, 112, 112,
			113, 114, 114, 115, 116, 116, 117, 118, 118, 119, 120, 120, 121,
			122, 122, 123, 124, 124, 125, 126, 126, 127, 6, 63, 109, 110,
			110, 111, 112, 112, 113, 113, 114, 114, 115, 116, 116, 117, 117,
			118, 118, 119, 120, 120, 121, 121, 122, 122, 123, 124, 124, 125,
			125, 126, 126, 127, 4, 64, 112, 112, 113, 113, 114, 114, 115,
			115, 116, 116, 117, 117, 118, 118, 119, 119, 120, 120, 121, 121,
			122, 122, 123, 123, 124, 124, 125, 125, 126, 126, 127, 127,
		},
		fsggml.TensorTypeBF16: {
			0, 0, 128, 63, 0, 64, 64, 64, 128, 64, 160, 64, 192,
			64, 224, 64, 0, 65, 16, 65, 32, 65, 48, 65, 64, 65,
			80, 65, 96, 65, 112, 65, 128, 65, 136, 65, 144, 65, 152,
			65, 160, 65, 168, 65, 176, 65, 184, 65, 192, 65, 200, 65,
			208, 65, 216, 65, 224, 65, 232, 65, 240, 65, 248, 65, 0,
			66, 4, 66, 8, 66, 12, 66, 16, 66, 20, 66, 24, 66,
			28, 66, 32, 66, 36, 66, 40, 66, 44, 66, 48, 66, 52,
			66, 56, 66, 60, 66, 64, 66, 68, 66, 72, 66, 76, 66,
			80, 66, 84, 66, 88, 66, 92, 66, 96, 66, 100, 66, 104,
			66, 108, 66, 112, 66, 116, 66, 120, 66, 124, 66, 128, 66,
			130, 66, 132, 66, 134, 66, 136, 66, 138, 66, 140, 66, 142,
			66, 144, 66, 146, 66, 148, 66, 150, 66, 152, 66, 154, 66,
			156, 66, 158, 66, 160, 66, 162, 66, 164, 66, 166, 66, 168,
			66, 170, 66, 172, 66, 174, 66, 176, 66, 178, 66, 180, 66,
			182, 66, 184, 66, 186, 66, 188, 66, 190, 66, 192, 66, 194,
			66, 196, 66, 198, 66, 200, 66, 202, 66, 204, 66, 206, 66,
			208, 66, 210, 66, 212, 66, 214, 66, 216, 66, 218, 66, 220,
			66, 222, 66, 224, 66, 226, 66, 228, 66, 230, 66, 232, 66,
			234, 66, 236, 66, 238, 66, 240, 66, 242, 66, 244, 66, 246,
			66, 248, 66, 250, 66, 252, 66, 254, 66, 0, 67, 1, 67,
			2, 67, 3, 67, 4, 67, 5, 67, 6, 67, 7, 67, 8,
			67, 9, 67, 10, 67, 11, 67, 12, 67, 13, 67, 14, 67,
			15, 67, 16, 67, 17, 67, 18, 67, 19, 67, 20, 67, 21,
			67, 22, 67, 23, 67, 24, 67, 25, 67, 26, 67, 27, 67,
			28, 67, 29, 67, 30, 67, 31, 67, 32, 67, 33, 67, 34,
			67, 35, 67, 36, 67, 37, 67, 38, 67, 39, 67, 40, 67,
			41, 67, 42, 67, 43, 67, 44, 67, 45, 67, 46, 67, 47,
			67, 48, 67, 49, 67, 50, 67, 51, 67, 52, 67, 53, 67,
			54, 67, 55, 67, 56, 67, 57, 67, 58, 67, 59, 67, 60,
			67, 61, 67, 62, 67, 63, 67, 64, 67, 65, 67, 66, 67,
			67, 67, 68, 67, 69, 67, 70, 67, 71, 67, 72, 67, 73,
			67, 74, 67, 75, 67, 76, 67, 77, 67, 78, 67, 79, 67,
			80, 67, 81, 67, 82, 67, 83, 67, 84, 67, 85, 67, 86,
			67, 87, 67, 88, 67, 89, 67, 90, 67, 91, 67, 92, 67,
			93, 67, 94, 67, 95, 67, 96, 67, 97, 67, 98, 67, 99,
			67, 100, 67, 101, 67, 102, 67, 103, 67, 104, 67, 105, 67,
			106, 67, 107, 67, 108, 67, 109, 67, 110, 67, 111, 67, 112,
			67, 113, 67, 114, 67, 115, 67, 116, 67, 117, 67, 118, 67,
			119, 67, 120, 67, 121, 67, 122, 67, 123, 67, 124, 67, 125,
			67, 126, 67, 127, 67,
		},
		fsggml.TensorTypeF16: {
			0, 0, 0, 60, 0, 64, 0, 66, 0, 68, 0, 69, 0, 70, 0, 71, 0,
			72, 128, 72, 0, 73, 128, 73, 0, 74, 128, 74, 0, 75, 128, 75,
			0, 76, 64, 76, 128, 76, 192, 76, 0, 77, 64, 77, 128, 77, 192,
			77, 0, 78, 64, 78, 128, 78, 192, 78, 0, 79, 64, 79, 128, 79,
			192, 79, 0, 80, 32, 80, 64, 80, 96, 80, 128, 80, 160, 80,
			192, 80, 224, 80, 0, 81, 32, 81, 64, 81, 96, 81, 128, 81,
			160, 81, 192, 81, 224, 81, 0, 82, 32, 82, 64, 82, 96, 82,
			128, 82, 160, 82, 192, 82, 224, 82, 0, 83, 32, 83, 64, 83,
			96, 83, 128, 83, 160, 83, 192, 83, 224, 83, 0, 84, 16, 84,
			32, 84, 48, 84, 64, 84, 80, 84, 96, 84, 112, 84, 128, 84,
			144, 84, 160, 84, 176, 84, 192, 84, 208, 84, 224, 84, 240,
			84, 0, 85, 16, 85, 32, 85, 48, 85, 64, 85, 80, 85, 96, 85,
			112, 85, 128, 85, 144, 85, 160, 85, 176, 85, 192, 85, 208,
			85, 224, 85, 240, 85, 0, 86, 16, 86, 32, 86, 48, 86, 64,
			86, 80, 86, 96, 86, 112, 86, 128, 86, 144, 86, 160, 86,
			176, 86, 192, 86, 208, 86, 224, 86, 240, 86, 0, 87, 16,
			87, 32, 87, 48, 87, 64, 87, 80, 87, 96, 87, 112, 87, 128,
			87, 144, 87, 160, 87, 176, 87, 192, 87, 208, 87, 224, 87,
			240, 87, 0, 88, 8, 88, 16, 88, 24, 88, 32, 88, 40, 88,
			48, 88, 56, 88, 64, 88, 72, 88, 80, 88, 88, 88, 96, 88,
			104, 88, 112, 88, 120, 88, 128, 88, 136, 88, 144, 88, 152,
			88, 160, 88, 168, 88, 176, 88, 184, 88, 192, 88, 200, 88,
			208, 88, 216, 88, 224, 88, 232, 88, 240, 88, 248, 88, 0,
			89, 8, 89, 16, 89, 24, 89, 32, 89, 40, 89, 48, 89, 56, 89,
			64, 89, 72, 89, 80, 89, 88, 89, 96, 89, 104, 89, 112, 89,
			120, 89, 128, 89, 136, 89, 144, 89, 152, 89, 160, 89, 168,
			89, 176, 89, 184, 89, 192, 89, 200, 89, 208, 89, 216, 89,
			224, 89, 232, 89, 240, 89, 248, 89, 0, 90, 8, 90, 16, 90,
			24, 90, 32, 90, 40, 90, 48, 90, 56, 90, 64, 90, 72, 90, 80,
			90, 88, 90, 96, 90, 104, 90, 112, 90, 120, 90, 128, 90,
			136, 90, 144, 90, 152, 90, 160, 90, 168, 90, 176, 90, 184,
			90, 192, 90, 200, 90, 208, 90, 216, 90, 224, 90, 232, 90,
			240, 90, 248, 90, 0, 91, 8, 91, 16, 91, 24, 91, 32, 91, 40,
			91, 48, 91, 56, 91, 64, 91, 72, 91, 80, 91, 88, 91, 96, 91,
			104, 91, 112, 91, 120, 91, 128, 91, 136, 91, 144, 91, 152,
			91, 160, 91, 168, 91, 176, 91, 184, 91, 192, 91, 200, 91,
			208, 91, 216, 91, 224, 91, 232, 91, 240, 91, 248, 91,
		},
		fsggml.TensorTypeF32: {
			0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128,
			64, 0, 0, 160, 64, 0, 0, 192, 64, 0, 0, 224, 64, 0, 0, 0, 65, 0,
			0, 16, 65, 0, 0, 32, 65, 0, 0, 48, 65, 0, 0, 64, 65, 0, 0, 80, 65,
			0, 0, 96, 65, 0, 0, 112, 65, 0, 0, 128, 65, 0, 0, 136, 65, 0, 0,
			144, 65, 0, 0, 152, 65, 0, 0, 160, 65, 0, 0, 168, 65, 0, 0, 176,
			65, 0, 0, 184, 65, 0, 0, 192, 65, 0, 0, 200, 65, 0, 0, 208, 65, 0,
			0, 216, 65, 0, 0, 224, 65, 0, 0, 232, 65, 0, 0, 240, 65, 0, 0, 248,
			65, 0, 0, 0, 66, 0, 0, 4, 66, 0, 0, 8, 66, 0, 0, 12, 66, 0, 0, 16,
			66, 0, 0, 20, 66, 0, 0, 24, 66, 0, 0, 28, 66, 0, 0, 32, 66, 0, 0,
			36, 66, 0, 0, 40, 66, 0, 0, 44, 66, 0, 0, 48, 66, 0, 0, 52, 66, 0,
			0, 56, 66, 0, 0, 60, 66, 0, 0, 64, 66, 0, 0, 68, 66, 0, 0, 72, 66,
			0, 0, 76, 66, 0, 0, 80, 66, 0, 0, 84, 66, 0, 0, 88, 66, 0, 0, 92, 66,
			0, 0, 96, 66, 0, 0, 100, 66, 0, 0, 104, 66, 0, 0, 108, 66, 0, 0, 112,
			66, 0, 0, 116, 66, 0, 0, 120, 66, 0, 0, 124, 66, 0, 0, 128, 66, 0, 0,
			130, 66, 0, 0, 132, 66, 0, 0, 134, 66, 0, 0, 136, 66, 0, 0, 138, 66,
			0, 0, 140, 66, 0, 0, 142, 66, 0, 0, 144, 66, 0, 0, 146, 66, 0, 0, 148,
			66, 0, 0, 150, 66, 0, 0, 152, 66, 0, 0, 154, 66, 0, 0, 156, 66, 0, 0,
			158, 66, 0, 0, 160, 66, 0, 0, 162, 66, 0, 0, 164, 66, 0, 0, 166, 66,
			0, 0, 168, 66, 0, 0, 170, 66, 0, 0, 172, 66, 0, 0, 174, 66, 0, 0, 176,
			66, 0, 0, 178, 66, 0, 0, 180, 66, 0, 0, 182, 66, 0, 0, 184, 66, 0, 0,
			186, 66, 0, 0, 188, 66, 0, 0, 190, 66, 0, 0, 192, 66, 0, 0, 194, 66, 0,
			0, 196, 66, 0, 0, 198, 66, 0, 0, 200, 66, 0, 0, 202, 66, 0, 0, 204, 66,
			0, 0, 206, 66, 0, 0, 208, 66, 0, 0, 210, 66, 0, 0, 212, 66, 0, 0, 214, 66,
			0, 0, 216, 66, 0, 0, 218, 66, 0, 0, 220, 66, 0, 0, 222, 66, 0, 0, 224, 66,
			0, 0, 226, 66, 0, 0, 228, 66, 0, 0, 230, 66, 0, 0, 232, 66, 0, 0, 234, 66,
			0, 0, 236, 66, 0, 0, 238, 66, 0, 0, 240, 66, 0, 0, 242, 66, 0, 0, 244, 66,
			0, 0, 246, 66, 0, 0, 248, 66, 0, 0, 250, 66, 0, 0, 252, 66, 0, 0, 254, 66,
			0, 0, 0, 67, 0, 0, 1, 67, 0, 0, 2, 67, 0, 0, 3, 67, 0, 0, 4, 67, 0, 0, 5, 67,
			0, 0, 6, 67, 0, 0, 7, 67, 0, 0, 8, 67, 0, 0, 9, 67, 0, 0, 10, 67, 0, 0, 11,
			67, 0, 0, 12, 67, 0, 0, 13, 67, 0, 0, 14, 67, 0, 0, 15, 67, 0, 0, 16, 67,
			0, 0, 17, 67, 0, 0, 18, 67, 0, 0, 19, 67, 0, 0, 20, 67, 0, 0, 21, 67, 0, 0,
			22, 67, 0, 0, 23, 67, 0, 0, 24, 67, 0, 0, 25, 67, 0, 0, 26, 67, 0, 0, 27,
			67, 0, 0, 28, 67, 0, 0, 29, 67, 0, 0, 30, 67, 0, 0, 31, 67, 0, 0, 32, 67,
			0, 0, 33, 67, 0, 0, 34, 67, 0, 0, 35, 67, 0, 0, 36, 67, 0, 0, 37, 67, 0, 0,
			38, 67, 0, 0, 39, 67, 0, 0, 40, 67, 0, 0, 41, 67, 0, 0, 42, 67, 0, 0, 43, 67,
			0, 0, 44, 67, 0, 0, 45, 67, 0, 0, 46, 67, 0, 0, 47, 67, 0, 0, 48, 67, 0, 0,
			49, 67, 0, 0, 50, 67, 0, 0, 51, 67, 0, 0, 52, 67, 0, 0, 53, 67, 0, 0, 54, 67,
			0, 0, 55, 67, 0, 0, 56, 67, 0, 0, 57, 67, 0, 0, 58, 67, 0, 0, 59, 67, 0, 0,
			60, 67, 0, 0, 61, 67, 0, 0, 62, 67, 0, 0, 63, 67, 0, 0, 64, 67, 0, 0, 65, 67,
			0, 0, 66, 67, 0, 0, 67, 67, 0, 0, 68, 67, 0, 0, 69, 67, 0, 0, 70, 67, 0, 0, 71,
			67, 0, 0, 72, 67, 0, 0, 73, 67, 0, 0, 74, 67, 0, 0, 75, 67, 0, 0, 76, 67, 0,
			0, 77, 67, 0, 0, 78, 67, 0, 0, 79, 67, 0, 0, 80, 67, 0, 0, 81, 67, 0, 0, 82,
			67, 0, 0, 83, 67, 0, 0, 84, 67, 0, 0, 85, 67, 0, 0, 86, 67, 0, 0, 87, 67, 0,
			0, 88, 67, 0, 0, 89, 67, 0, 0, 90, 67, 0, 0, 91, 67, 0, 0, 92, 67, 0, 0, 93,
			67, 0, 0, 94, 67, 0, 0, 95, 67, 0, 0, 96, 67, 0, 0, 97, 67, 0, 0, 98, 67, 0,
			0, 99, 67, 0, 0, 100, 67, 0, 0, 101, 67, 0, 0, 102, 67, 0, 0, 103, 67, 0, 0,
			104, 67, 0, 0, 105, 67, 0, 0, 106, 67, 0, 0, 107, 67, 0, 0, 108, 67, 0, 0, 109,
			67, 0, 0, 110, 67, 0, 0, 111, 67, 0, 0, 112, 67, 0, 0, 113, 67, 0, 0, 114, 67,
			0, 0, 115, 67, 0, 0, 116, 67, 0, 0, 117, 67, 0, 0, 118, 67, 0, 0, 119, 67, 0,
			0, 120, 67, 0, 0, 121, 67, 0, 0, 122, 67, 0, 0, 123, 67, 0, 0, 124, 67, 0, 0,
			125, 67, 0, 0, 126, 67, 0, 0, 127, 67,
		},
		fsggml.TensorTypeQ4_K: {
			52, 52, 0, 0, 136, 208, 216, 223, 0, 0, 0, 0, 8, 0, 8, 15, 128,
			128, 129, 129, 146, 146, 147, 147, 164, 164, 165, 165, 166, 182,
			183, 183, 184, 200, 201, 201, 202, 218, 218, 219, 219, 236, 236,
			237, 237, 254, 254, 255, 202, 202, 202, 203, 203, 203, 219, 219,
			219, 220, 220, 220, 220, 220, 236, 237, 237, 237, 237, 237,
			237, 237, 238, 254, 254, 254, 254, 254, 255, 255, 255, 255, 220,
			220, 220, 220, 221, 221, 221, 221, 221, 221, 221, 237, 237, 237,
			238, 238, 238, 238, 238, 238, 238, 238, 238, 254, 254, 255, 255,
			255, 255, 255, 255, 255, 237, 237, 237, 237, 237, 237, 237, 238,
			238, 238, 238, 238, 238, 238, 238, 238, 254, 254, 254, 254, 254,
			254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
		},
		fsggml.TensorTypeQ2_K: {
			1, 2, 3, 3, 4, 5, 7, 7, 8, 9, 10, 11, 12, 13, 14, 15, 184, 184,
			184, 185, 249, 249, 249, 249, 249, 250, 250, 254, 254, 254, 254,
			255, 253, 253, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254,
			254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
			255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
			255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 171, 69, 0, 0,
		},
		fsggml.TensorTypeQ5_K: {
			32, 48, 0, 0, 136, 208, 216, 223, 0, 0, 0, 0, 8, 0, 7, 15, 254,
			254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254,
			254, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
			255, 255, 255, 255, 255, 0, 1, 2, 19, 20, 37, 38, 55, 56, 73, 74,
			91, 92, 109, 110, 127, 112, 128, 129, 146, 147, 164, 165, 182, 183,
			200, 201, 218, 219, 236, 237, 254, 133, 133, 149, 150, 150, 150,
			167, 167, 167, 168, 184, 184, 185, 185, 201, 202, 202, 202, 219,
			219, 219, 219, 236, 236, 236, 237, 253, 253, 254, 254, 254, 255,
			169, 169, 169, 169, 186, 186, 186, 186, 186, 187, 187, 203, 203,
			203, 204, 204, 204, 220, 220, 221, 221, 221, 221, 237, 237, 238,
			238, 238, 238, 254, 255, 255, 203, 203, 203, 204, 204, 204, 204,
			204, 220, 220, 220, 221, 221, 221, 221, 221, 237, 237, 238, 238,
			238, 238, 238, 238, 254, 255, 255, 255, 255, 255, 255, 255,
		},
		fsggml.TensorTypeQ6_K: {
			96, 110, 92, 90, 88, 70, 68, 50, 48, 46, 44, 42, 24, 22, 4, 2, 80,
			95, 78, 77, 76, 59, 58, 57, 40, 39, 38, 21, 20, 19, 2, 1, 75, 75,
			74, 57, 57, 56, 55, 39, 38, 37, 21, 20, 20, 19, 2, 2, 72, 55, 55,
			54, 54, 37, 37, 36, 36, 19, 19, 18, 18, 1, 1, 0, 35, 35, 35, 35,
			34, 18, 18, 18, 17, 17, 17, 1, 1, 0, 0, 0, 35, 35, 34, 34, 18,
			18, 18, 17, 17, 17, 17, 1, 0, 0, 0, 0, 35, 35, 35, 19, 19, 18, 18,
			18, 18, 18, 1, 1, 1, 1, 1, 1, 34, 34, 18, 18, 18, 18, 17, 17, 17,
			17, 1, 1, 0, 0, 0, 0, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
			0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 248, 240, 231, 224, 216, 208, 200, 192, 184, 176,
			166, 160, 152, 144, 136, 128, 235, 43,
		},
		fsggml.TensorTypeQ3_K: {
			1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 20, 23, 23, 7, 7, 6, 6, 6, 2,
			1, 1, 1, 1, 0, 0, 22, 22, 6, 6, 5, 5, 5, 1, 1, 1, 1, 1, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 238, 204, 170, 136, 102, 68,
			34, 1, 5, 5, 5, 5, 189, 63,
		},
	}
)
