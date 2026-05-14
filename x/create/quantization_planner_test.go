package create

import (
	"slices"
	"testing"
)

func TestSafetensorsQuantizationPlannerFromFilesUsesModelPolicies(t *testing.T) {
	gemma4Config := []byte(`{
		"architectures": ["Gemma4ForConditionalGeneration"],
		"text_config": {"num_hidden_layers": 64, "num_experts": 8}
	}`)
	qwenConfig := []byte(`{
		"architectures": ["Qwen3_5MoeForConditionalGeneration"],
		"text_config": {"model_type": "qwen3"}
	}`)
	genericConfig := []byte(`{
		"architectures": ["LlamaForCausalLM"],
		"model_type": "llama"
	}`)

	tests := []struct {
		name     string
		config   []byte
		tensor   string
		shape    []int32
		quantize string
		want     string
	}{
		{
			name:     "Gemma4 embeds promote NVFP4 to MXFP8",
			config:   gemma4Config,
			tensor:   "model.language_model.embed_tokens.weight",
			shape:    []int32{1024, 128},
			quantize: "nvfp4",
			want:     "mxfp8",
		},
		{
			name:     "Gemma4 router stays BF16",
			config:   gemma4Config,
			tensor:   "model.language_model.layers.0.moe.router.proj.weight",
			shape:    []int32{1024, 128},
			quantize: "nvfp4",
			want:     "",
		},
		{
			name:     "Gemma4 eight expert K projection promotes to MXFP8",
			config:   gemma4Config,
			tensor:   "model.language_model.layers.0.self_attn.k_proj.weight",
			shape:    []int32{1024, 128},
			quantize: "nvfp4",
			want:     "mxfp8",
		},
		{
			name:     "Qwen direct non-affine embed stays BF16",
			config:   qwenConfig,
			tensor:   "language_model.model.embed_tokens.weight",
			shape:    []int32{1024, 128},
			quantize: "nvfp4",
			want:     "",
		},
		{
			name:     "Qwen vision tower stays BF16",
			config:   qwenConfig,
			tensor:   "vision_tower.blocks.0.attn.proj.weight",
			shape:    []int32{1024, 128},
			quantize: "nvfp4",
			want:     "",
		},
		{
			name:     "Qwen ordinary projection uses requested quantization",
			config:   qwenConfig,
			tensor:   "language_model.model.layers.0.self_attn.q_proj.weight",
			shape:    []int32{1024, 128},
			quantize: "nvfp4",
			want:     "nvfp4",
		},
		{
			name:     "Generic model uses generic mixed precision policy",
			config:   genericConfig,
			tensor:   "model.layers.0.self_attn.v_proj.weight",
			shape:    []int32{1024, 128},
			quantize: "int4",
			want:     "int8",
		},
		{
			name:     "Generic model quantizes ordinary projection",
			config:   genericConfig,
			tensor:   "model.layers.0.self_attn.q_proj.weight",
			shape:    []int32{1024, 128},
			quantize: "nvfp4",
			want:     "nvfp4",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			planner, err := NewSafetensorsQuantizationPlannerFromFiles(map[string][]byte{
				"config.json": tt.config,
			})
			if err != nil {
				t.Fatalf("NewSafetensorsQuantizationPlannerFromFiles() error = %v", err)
			}

			if got := planner.QuantizationType(tt.tensor, tt.shape, tt.quantize); got != tt.want {
				t.Fatalf("QuantizationType(%q, %v, %q) = %q, want %q", tt.tensor, tt.shape, tt.quantize, got, tt.want)
			}
		})
	}
}

func TestResolveUploadedSourceQuantization(t *testing.T) {
	genericConfig := []byte(`{
		"architectures": ["LlamaForCausalLM"],
		"model_type": "llama"
	}`)
	fp8Config := []byte(`{
		"architectures": ["Qwen3_5ForCausalLM"],
		"model_type": "qwen3",
		"quantization_config": {
			"quant_method": "fp8",
			"weight_block_size": [128, 128]
		}
	}`)
	modelOptConfig := []byte(`{
		"producer": {"name": "nvidia-modelopt", "version": "1.0"},
		"quantization": {"quant_algo": "NVFP4", "group_size": 16}
	}`)

	tests := []struct {
		name      string
		fileNames []string
		files     map[string][]byte
		requested string
		want      string
		wantErr   string
	}{
		{
			name:      "no source quantization keeps requested value",
			fileNames: []string{"config.json", "model.layers.0.self_attn.q_proj.weight"},
			files: map[string][]byte{
				"config.json": genericConfig,
			},
			requested: "int4",
			want:      "int4",
		},
		{
			name:      "ModelOpt uploaded config preserves source quantization",
			fileNames: []string{"config.json", "hf_quant_config.json", "model.layers.0.self_attn.q_proj.weight"},
			files: map[string][]byte{
				"config.json":          genericConfig,
				"hf_quant_config.json": modelOptConfig,
			},
			want: "",
		},
		{
			name:      "ModelOpt uploaded config rejects requantization",
			fileNames: []string{"config.json", "hf_quant_config.json", "model.layers.0.self_attn.q_proj.weight"},
			files: map[string][]byte{
				"config.json":          genericConfig,
				"hf_quant_config.json": modelOptConfig,
			},
			requested: "int4",
			wantErr:   `cannot requantize already-quantized source model with --quantize "int4"`,
		},
		{
			name:      "uploaded fp8 companions default to mxfp8",
			fileNames: []string{"config.json", "model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.q_proj.weight_scale_inv"},
			files: map[string][]byte{
				"config.json": fp8Config,
			},
			want: "mxfp8",
		},
		{
			name:      "uploaded fp8 companions allow nvfp4 conversion",
			fileNames: []string{"config.json", "model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.q_proj.weight_scale_inv"},
			files: map[string][]byte{
				"config.json": fp8Config,
			},
			requested: "nvfp4",
			want:      "nvfp4",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ResolveUploadedSourceQuantization(tt.fileNames, tt.files, tt.requested)
			if tt.wantErr != "" {
				if err == nil || err.Error() != tt.wantErr {
					t.Fatalf("ResolveUploadedSourceQuantization() error = %v, want %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("ResolveUploadedSourceQuantization() error = %v", err)
			}
			if got != tt.want {
				t.Fatalf("ResolveUploadedSourceQuantization() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestAnalyzeUploadedSafetensorsSource(t *testing.T) {
	config := []byte(`{
		"architectures": ["Gemma4ForConditionalGeneration"],
		"text_config": {"num_hidden_layers": 64, "num_experts": 8},
		"vision_config": {}
	}`)

	analysis, err := AnalyzeUploadedSafetensorsSource(
		[]string{"config.json", "model.language_model.embed_tokens.weight"},
		map[string][]byte{"config.json": config},
		"nvfp4",
		"",
	)
	if err != nil {
		t.Fatalf("AnalyzeUploadedSafetensorsSource() error = %v", err)
	}

	wantCaps := []string{"completion", "vision"}
	if !slices.Equal(analysis.Capabilities, wantCaps) {
		t.Fatalf("Capabilities = %#v, want %#v", analysis.Capabilities, wantCaps)
	}
	if analysis.EffectiveQuantize != "nvfp4" {
		t.Fatalf("EffectiveQuantize = %q, want %q", analysis.EffectiveQuantize, "nvfp4")
	}
	if analysis.QuantizationPlanner == nil {
		t.Fatal("QuantizationPlanner is nil")
	}

	if got := analysis.QuantizationPlanner.QuantizationType("model.language_model.embed_tokens.weight", []int32{1024, 128}, analysis.EffectiveQuantize); got != "mxfp8" {
		t.Fatalf("QuantizationType() = %q, want %q", got, "mxfp8")
	}
}
