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

func fp8BlockConfig(rows, cols int32) sourceModelConfig {
	return sourceModelConfig{
		QuantizationConfig: sourceQuantization{QuantMethod: "fp8", WeightBlockSize: []int32{rows, cols}},
	}
}

func TestClassify(t *testing.T) {
	tests := []struct {
		name      string
		cfg       sourceModelConfig
		tensors   map[string]string
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
			name:     "mlx prequantized (.scales)",
			tensors:  map[string]string{"model.layers.0.weight": "U32", "model.layers.0.scales": "BF16"},
			wantKind: SourcePrequantized,
		},
		{
			// ModelOpt NVFP4 whose hf_quant_config.json sidecar is absent:
			// recognized from the packed weight + scale companion (finding #7).
			name:     "modelopt nvfp4 without config sidecar",
			tensors:  map[string]string{"model.layers.0.weight": "U8", "model.layers.0.weight_scale": "F8_E4M3"},
			wantKind: SourcePrequantized,
		},
		{
			name:     "compressed-tensors nvfp4 (.weight_packed)",
			tensors:  map[string]string{"model.layers.0.weight_packed": "U8", "model.layers.0.weight_scale": "F8_E4M3"},
			wantKind: SourcePrequantized,
		},
		{
			name:      "block-fp8 auto-converts to mxfp8",
			cfg:       fp8BlockConfig(128, 128),
			tensors:   map[string]string{"model.layers.0.weight": "F8_E4M3", "model.layers.0.weight_scale_inv": "F32"},
			wantKind:  SourceBlockFP8,
			wantQuant: "mxfp8",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Classify(newInventory(tt.cfg, tt.tensors), tt.requested)
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
			name:      "block-fp8 rejects quantize flag",
			cfg:       fp8BlockConfig(128, 128),
			tensors:   map[string]string{"model.layers.0.weight": "F8_E4M3", "model.layers.0.weight_scale_inv": "F32"},
			requested: "nvfp4",
			wantErr:   "cannot quantize an fp8 source",
		},
		{
			name:    "block-fp8 missing block size",
			tensors: map[string]string{"model.layers.0.weight": "F8_E4M3", "model.layers.0.weight_scale_inv": "F32"},
			wantErr: "missing weight_block_size",
		},
		{
			name:    "block-fp8 unsupported block size",
			cfg:     fp8BlockConfig(64, 64),
			tensors: map[string]string{"model.layers.0.weight": "F8_E4M3", "model.layers.0.weight_scale_inv": "F32"},
			wantErr: "unsupported fp8 source block size",
		},
		{
			name:    "e5m2 fp8 unsupported",
			tensors: map[string]string{"model.layers.0.weight": "F8_E5M2"},
			wantErr: "F8_E5M2",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := Classify(newInventory(tt.cfg, tt.tensors), tt.requested)
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("Classify() error = %v, want substring %q", err, tt.wantErr)
			}
		})
	}
}
