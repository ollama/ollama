package create

import "testing"

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
