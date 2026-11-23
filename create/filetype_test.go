package create

import "testing"

func TestSourceFloatFileType(t *testing.T) {
	for _, tc := range []struct {
		name    string
		tensors map[string]SourceTensor
		want    string
	}{
		{"bf16 weights with f32 norms", map[string]SourceTensor{
			"w":    {Dtype: "BF16", Shape: []int32{2048, 2048}},
			"norm": {Dtype: "F32", Shape: []int32{2048}},
			"bias": {Dtype: "F32", Shape: []int32{2048}},
		}, "BF16"},
		{"f16", map[string]SourceTensor{"w": {Dtype: "F16", Shape: []int32{4, 4}}}, "F16"},
		{"f32", map[string]SourceTensor{"w": {Dtype: "F32", Shape: []int32{4}}}, "F32"},
		{"empty", map[string]SourceTensor{}, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := sourceFloatFileType(Inventory{Tensors: tc.tensors}); got != tc.want {
				t.Errorf("sourceFloatFileType = %q, want %q", got, tc.want)
			}
		})
	}
}
