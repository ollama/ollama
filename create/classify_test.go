package create

import (
	"strings"
	"testing"
)

func newInventory(cfg sourceModelConfig, tensors map[string]string) Inventory {
	m := make(map[string]SourceTensor)
	for name, dtype := range tensors {
		m[name] = SourceTensor{Name: name, Dtype: dtype, Shape: []int32{128, 128}, File: "model.safetensors"}
	}
	return Inventory{Dir: "test", Config: cfg, Tensors: m}
}

// withShapes overrides the default [128, 128] shape of the named tensors.
func withShapes(inv Inventory, shapes map[string][]int32) Inventory {
	for name, shape := range shapes {
		t := inv.Tensors[name]
		t.Shape = shape
		inv.Tensors[name] = t
	}
	return inv
}

func TestClassify(t *testing.T) {
	tests := []struct {
		name      string
		cfg       sourceModelConfig
		tensors   map[string]string
		shapes    map[string][]int32
		requested string
		wantKind  SourceKind
		wantQuant string
	}{
		{
			name:     "float, no quantize",
			tensors:  map[string]string{"model.embed.weight": "BF16", "model.layers.0.weight": "BF16"},
			wantKind: SourceFloat,
		},
		{
			name:      "float, quantize int4",
			tensors:   map[string]string{"model.layers.0.weight": "BF16"},
			requested: "int4",
			wantKind:  SourceFloat,
			wantQuant: "int4",
		},
		{
			name:      "float, quantize alias fp8 resolves to int8",
			tensors:   map[string]string{"model.layers.0.weight": "F32"},
			requested: "fp8",
			wantKind:  SourceFloat,
			wantQuant: "int8",
		},
		{
			name:      "mlx prequantized (.scales)",
			cfg:       sourceModelConfig{Quantization: sourceQuantization{Bits: 4, Mode: "affine", GroupSize: 32}},
			tensors:   map[string]string{"model.layers.0.weight": "U32", "model.layers.0.scales": "BF16"},
			wantKind:  SourcePrequantized,
			wantQuant: "int4",
		},
		{
			name:     "mlx prequantized without quantization metadata",
			tensors:  map[string]string{"model.layers.0.weight": "U32", "model.layers.0.scales": "BF16"},
			wantKind: SourcePrequantized,
		},
		{
			// ModelOpt NVFP4 whose hf_quant_config.json sidecar is absent:
			// recognized from the packed weight + scale companion (finding #7).
			name:      "modelopt nvfp4 without config sidecar",
			tensors:   map[string]string{"model.layers.0.weight": "U8", "model.layers.0.weight_scale": "F8_E4M3"},
			wantKind:  SourcePrequantized,
			wantQuant: "nvfp4",
		},
		{
			name:      "compressed-tensors nvfp4 (.weight_packed)",
			tensors:   map[string]string{"model.layers.0.weight_packed": "U8", "model.layers.0.weight_scale": "F8_E4M3"},
			wantKind:  SourcePrequantized,
			wantQuant: "nvfp4",
		},
		{
			name: "mixed prequantized formats have no single file type",
			cfg:  sourceModelConfig{Quantization: sourceQuantization{Bits: 4, Mode: "affine", GroupSize: 32}},
			tensors: map[string]string{
				"model.layers.0.weight":        "U32",
				"model.layers.0.scales":        "BF16",
				"model.layers.1.weight_packed": "U8",
				"model.layers.1.weight_scale":  "F8_E4M3",
			},
			wantKind: SourcePrequantized,
		},
		{
			name:      "per-tensor fp8 (ModelOpt)",
			tensors:   map[string]string{"model.layers.0.weight": "F8_E4M3", "model.layers.0.weight_scale": "F32", "model.layers.0.input_scale": "F32"},
			shapes:    map[string][]int32{"model.layers.0.weight_scale": {}, "model.layers.0.input_scale": {1}},
			wantKind:  SourceFP8,
			wantQuant: "mxfp8",
		},
		{
			name: "nvfp4 beside per-tensor fp8 is prequantized",
			tensors: map[string]string{
				"model.layers.0.weight":         "U8",
				"model.layers.0.weight_scale":   "F8_E4M3",
				"model.layers.0.weight_scale_2": "F32",
				"model.layers.1.weight":         "F8_E4M3",
				"model.layers.1.weight_scale":   "F32",
			},
			shapes:    map[string][]int32{"model.layers.0.weight_scale_2": {}, "model.layers.1.weight_scale": {}},
			wantKind:  SourcePrequantized,
			wantQuant: "nvfp4",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Classify(withShapes(newInventory(tt.cfg, tt.tensors), tt.shapes), tt.requested)
			if err != nil {
				t.Fatalf("Classify() error = %v", err)
			}
			if got.Kind != tt.wantKind {
				t.Errorf("Kind = %v, want %v", got.Kind, tt.wantKind)
			}
			if got.Quantize != tt.wantQuant {
				t.Errorf("Quantize = %q, want %q", got.Quantize, tt.wantQuant)
			}
		})
	}
}

func TestClassifyErrors(t *testing.T) {
	tests := []struct {
		name      string
		cfg       sourceModelConfig
		tensors   map[string]string
		shapes    map[string][]int32
		requested string
		wantErr   string
	}{
		{
			name:      "invalid quantize type",
			tensors:   map[string]string{"model.layers.0.weight": "BF16"},
			requested: "int3",
			wantErr:   "unsupported quantize type",
		},
		{
			name:      "mlx prequantized rejects requantize",
			tensors:   map[string]string{"model.layers.0.weight": "U32", "model.layers.0.scales": "BF16"},
			requested: "int4",
			wantErr:   "cannot requantize",
		},
		{
			name:      "modelopt nvfp4 rejects requantize",
			tensors:   map[string]string{"model.layers.0.weight": "U8", "model.layers.0.weight_scale": "F8_E4M3"},
			requested: "nvfp4",
			wantErr:   "cannot requantize",
		},
		{
			name:      "fp8 rejects quantize flag",
			tensors:   map[string]string{"model.layers.0.weight": "F8_E4M3", "model.layers.0.weight_scale": "F32"},
			shapes:    map[string][]int32{"model.layers.0.weight_scale": {}},
			requested: "mxfp8",
			wantErr:   "cannot quantize an fp8 source",
		},
		{
			name:    "per-channel fp8 (compressed-tensors) rejected",
			tensors: map[string]string{"model.layers.0.weight": "F8_E4M3", "model.layers.0.weight_scale": "BF16"},
			shapes:  map[string][]int32{"model.layers.0.weight_scale": {128, 1}},
			wantErr: "per-channel FP8 import is not supported yet",
		},
		{
			name:    "block-scaled fp8 rejected",
			tensors: map[string]string{"model.layers.0.weight": "F8_E4M3", "model.layers.0.weight_scale_inv": "BF16"},
			shapes:  map[string][]int32{"model.layers.0.weight": {256, 256}, "model.layers.0.weight_scale_inv": {2, 2}},
			wantErr: "block-scaled FP8 cannot be stored without re-rounding every weight",
		},
		{
			name:    "fp8 weight without a float scale",
			tensors: map[string]string{"model.layers.0.weight": "F8_E4M3"},
			wantErr: "has no float scale companion",
		},
		{
			name:    "e5m2 fp8 unsupported",
			tensors: map[string]string{"model.layers.0.weight": "F8_E5M2"},
			wantErr: "F8_E5M2",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := Classify(withShapes(newInventory(tt.cfg, tt.tensors), tt.shapes), tt.requested)
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("Classify() error = %v, want substring %q", err, tt.wantErr)
			}
		})
	}
}
