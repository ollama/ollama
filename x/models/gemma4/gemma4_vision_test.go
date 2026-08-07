package gemma4

import (
	"strings"
	"testing"
)

func TestParseVisionConfig(t *testing.T) {
	t.Run("small vision tower", func(t *testing.T) {
		cfg, err := parseVisionConfig([]byte(`{
			"vision_config": {
				"model_type": "gemma4_vision",
				"hidden_size": 768,
				"num_hidden_layers": 16,
				"num_attention_heads": 12,
				"patch_size": 16,
				"pooling_kernel_size": 3,
				"default_output_length": 280
			}
		}`))
		if err != nil {
			t.Fatal(err)
		}
		if cfg == nil {
			t.Fatal("expected supported vision config")
		}
		if cfg.ImageSeqLength != 280 {
			t.Fatalf("image sequence length = %d, want 280", cfg.ImageSeqLength)
		}
	})

	t.Run("unified vision tower", func(t *testing.T) {
		cfg, err := parseVisionConfig([]byte(`{
			"vision_config": {
				"model_type": "gemma4_unified_vision",
				"mm_embed_dim": 3840,
				"mm_posemb_size": 1120,
				"model_patch_size": 48,
				"patch_size": 16,
				"pooling_kernel_size": 3,
				"num_soft_tokens": 280
			}
		}`))
		if err != nil {
			t.Fatal(err)
		}
		if cfg == nil {
			t.Fatal("expected supported unified vision config")
		}
		if cfg.HiddenSize != 3840 || cfg.PositionEmbeddingSize != 1120 {
			t.Fatalf("embedding dimensions = (%d, %d), want (3840, 1120)", cfg.HiddenSize, cfg.PositionEmbeddingSize)
		}
		if cfg.PatchSize != 48 || cfg.PoolingKernelSize != 1 {
			t.Fatalf("patch configuration = (%d, %d), want (48, 1)", cfg.PatchSize, cfg.PoolingKernelSize)
		}
		if cfg.ImageSeqLength != 280 {
			t.Fatalf("image sequence length = %d, want 280", cfg.ImageSeqLength)
		}
	})

	t.Run("no vision config is text only", func(t *testing.T) {
		cfg, err := parseVisionConfig([]byte(`{"hidden_size": 768}`))
		if err != nil {
			t.Fatal(err)
		}
		if cfg != nil {
			t.Fatalf("vision config = %+v, want nil for a text-only model", cfg)
		}
	})

	t.Run("unsupported vision tower is an error", func(t *testing.T) {
		// The manifest advertises vision for any config carrying a
		// vision_config, so a silent text-only fallback here would leave
		// /api/show claiming a tower the runner cannot execute.
		cfg, err := parseVisionConfig([]byte(`{"vision_config": {"model_type": "gemma4_future_vision"}}`))
		if err == nil {
			t.Fatalf("vision config = %+v, want an error naming the tower", cfg)
		}
		if !strings.Contains(err.Error(), "gemma4_future_vision") {
			t.Fatalf("error = %v, want it to name the unsupported tower", err)
		}
	})

	t.Run("large vision tower", func(t *testing.T) {
		cfg, err := parseVisionConfig([]byte(`{
			"vision_config": {
				"model_type": "gemma4_vision",
				"hidden_size": 1152,
				"num_hidden_layers": 27,
				"num_attention_heads": 16,
				"head_dim": 72,
				"standardize": true
			}
		}`))
		if err != nil {
			t.Fatal(err)
		}
		if cfg == nil {
			t.Fatal("expected supported vision config")
		}
		if !cfg.Standardize {
			t.Fatal("expected vision output standardization")
		}
	})
}

func TestVisionTokenCount(t *testing.T) {
	m := &Model{VisionConfig: &VisionConfig{
		PatchSize:         16,
		PoolingKernelSize: 3,
		ImageSeqLength:    280,
	}}

	if got := m.VisionTokenCount(48, 48); got != 1 {
		t.Fatalf("48x48 token count = %d, want 1", got)
	}
	if got := m.VisionTokenCount(48, 96); got != 2 {
		t.Fatalf("96x48 token count = %d, want 2", got)
	}

	m.VisionConfig = &VisionConfig{
		ModelType:         "gemma4_unified_vision",
		PatchSize:         48,
		PoolingKernelSize: 1,
		ImageSeqLength:    280,
	}
	if got := m.VisionTokenCount(48, 96); got != 2 {
		t.Fatalf("unified 96x48 token count = %d, want 2", got)
	}
}

func TestGemma4ImageSize(t *testing.T) {
	const alignment = 48

	for _, tt := range []struct {
		name                  string
		height, width         int
		minTokens, maxTokens  int
		wantHeight, wantWidth int
	}{
		{name: "small square", height: 100, width: 100, minTokens: 280, maxTokens: 1120, wantHeight: 768, wantWidth: 768},
		{name: "small landscape", height: 300, width: 600, minTokens: 280, maxTokens: 1120, wantHeight: 528, wantWidth: 1104},
		{name: "preserve high resolution", height: 1080, width: 1920, minTokens: 280, maxTokens: 1120, wantHeight: 1104, wantWidth: 1920},
		{name: "downscale high resolution", height: 1080, width: 1920, minTokens: 280, maxTokens: 280, wantHeight: 576, wantWidth: 1056},
		{name: "very wide", height: 1, width: 10000, minTokens: 280, maxTokens: 1120, wantHeight: 48, wantWidth: 13440},
		{name: "very tall", height: 10000, width: 1, minTokens: 280, maxTokens: 1120, wantHeight: 13440, wantWidth: 48},
		{name: "extreme wide", height: 1, width: 1000000000, minTokens: 280, maxTokens: 1120, wantHeight: 48, wantWidth: 53760},
	} {
		t.Run(tt.name, func(t *testing.T) {
			minPixels := tt.minTokens * alignment * alignment
			maxPixels := tt.maxTokens * alignment * alignment
			gotH, gotW := gemma4ImageSize(tt.height, tt.width, alignment, minPixels, maxPixels)
			if gotH != tt.wantHeight || gotW != tt.wantWidth {
				t.Fatalf("size = %dx%d, want %dx%d", gotH, gotW, tt.wantHeight, tt.wantWidth)
			}
			if gotH%alignment != 0 || gotW%alignment != 0 {
				t.Fatalf("size %dx%d is not aligned to %d", gotH, gotW, alignment)
			}
			if gotH*gotW > maxPixels {
				t.Fatalf("size %dx%d exceeds pixel budget %d", gotH, gotW, maxPixels)
			}
		})
	}
}

// TestGemma4ImageTokenLimit pins every fidelity step and the boundary on each
// side of it, so neither a threshold nor the order of the switch can be changed
// without the test noticing. The measurements the thresholds derive from are
// above the budget constants in vision_prompt.go.
func TestGemma4ImageTokenLimit(t *testing.T) {
	for _, tt := range []struct {
		name     string
		headroom int
		want     int
	}{
		{name: "no budget reported falls back", headroom: 0, want: gemma4FallbackImageTokens},
		{name: "just below the standard step", headroom: gemma4StandardResHeadroom - 1, want: gemma4FallbackImageTokens},
		{name: "exactly the standard step", headroom: gemma4StandardResHeadroom, want: gemma4StandardImageTokens},
		{name: "just below the high step", headroom: gemma4HighResHeadroom - 1, want: gemma4StandardImageTokens},
		{name: "exactly the high step", headroom: gemma4HighResHeadroom, want: gemma4MaxImageTokens},
		{name: "well above the high step", headroom: 96 << 30, want: gemma4MaxImageTokens},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := gemma4ImageTokenLimit(tt.headroom); got != tt.want {
				t.Fatalf("token limit = %d, want %d", got, tt.want)
			}
		})
	}
}
