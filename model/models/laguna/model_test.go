package laguna

import (
	"iter"
	"math"
	"testing"
)

type testConfig map[string]any

func (c testConfig) Architecture() string { return "laguna" }

func (c testConfig) key(key string) string {
	switch {
	case len(key) >= len("tokenizer.") && key[:len("tokenizer.")] == "tokenizer.":
		return key
	case len(key) >= len("general.") && key[:len("general.")] == "general.":
		return key
	default:
		return "laguna." + key
	}
}

func (c testConfig) String(key string, defaultValue ...string) string {
	if v, ok := c[c.key(key)].(string); ok {
		return v
	}
	if len(defaultValue) > 0 {
		return defaultValue[0]
	}
	return ""
}

func (c testConfig) Uint(key string, defaultValue ...uint32) uint32 {
	switch v := c[c.key(key)].(type) {
	case uint32:
		return v
	case int:
		return uint32(v)
	}
	if len(defaultValue) > 0 {
		return defaultValue[0]
	}
	return 0
}

func (c testConfig) Float(key string, defaultValue ...float32) float32 {
	if v, ok := c[c.key(key)].(float32); ok {
		return v
	}
	if len(defaultValue) > 0 {
		return defaultValue[0]
	}
	return 0
}

func (c testConfig) Bool(key string, defaultValue ...bool) bool {
	if v, ok := c[c.key(key)].(bool); ok {
		return v
	}
	if len(defaultValue) > 0 {
		return defaultValue[0]
	}
	return false
}

func (c testConfig) Strings(key string, defaultValue ...[]string) []string {
	if v, ok := c[c.key(key)].([]string); ok {
		return v
	}
	if len(defaultValue) > 0 {
		return defaultValue[0]
	}
	return nil
}

func (c testConfig) Ints(key string, defaultValue ...[]int32) []int32 {
	if v, ok := c[c.key(key)].([]int32); ok {
		return v
	}
	if len(defaultValue) > 0 {
		return defaultValue[0]
	}
	return nil
}

func (c testConfig) Uints(key string, defaultValue ...[]uint32) []uint32 {
	if v, ok := c[c.key(key)].([]uint32); ok {
		return v
	}
	if len(defaultValue) > 0 {
		return defaultValue[0]
	}
	return nil
}

func (c testConfig) Floats(key string, defaultValue ...[]float32) []float32 {
	if v, ok := c[c.key(key)].([]float32); ok {
		return v
	}
	if len(defaultValue) > 0 {
		return defaultValue[0]
	}
	return nil
}

func (c testConfig) Bools(key string, defaultValue ...[]bool) []bool {
	if v, ok := c[c.key(key)].([]bool); ok {
		return v
	}
	if len(defaultValue) > 0 {
		return defaultValue[0]
	}
	return nil
}

func (c testConfig) Len() int { return len(c) }

func (c testConfig) Keys() iter.Seq[string] {
	return func(yield func(string) bool) {
		for key := range c {
			if !yield(key) {
				return
			}
		}
	}
}

func (c testConfig) Value(key string) any { return c[key] }

func TestNewOptionsLayerConfig(t *testing.T) {
	cfg := testConfig{
		"laguna.block_count":                          uint32(4),
		"laguna.embedding_length":                     uint32(128),
		"laguna.attention.key_length":                 uint32(16),
		"laguna.attention.head_count":                 []uint32{8, 16, 16, 16},
		"laguna.attention.head_count_kv":              uint32(4),
		"laguna.attention.layer_norm_rms_epsilon":     float32(1e-6),
		"laguna.attention.sliding_window":             uint32(512),
		"laguna.attention.sliding_window_pattern":     []bool{false, true, true, true},
		"laguna.rope.dimension_count":                 uint32(8),
		"laguna.rope.freq_base":                       float32(500000),
		"laguna.rope.scaling.factor":                  float32(32),
		"laguna.rope.scaling.original_context_length": uint32(4096),
		"laguna.rope.swa.dimension_count":             uint32(16),
		"laguna.rope.swa.freq_base":                   float32(10000),
		"laguna.expert_count":                         uint32(32),
		"laguna.expert_used_count":                    uint32(4),
		"laguna.expert_weights_norm":                  true,
		"laguna.expert_weights_scale":                 float32(2.5),
		"laguna.decoder_sparse_step":                  uint32(1),
		"laguna.dense_layers":                         []uint32{0},
	}

	opts := newOptions(cfg, 4)
	if got := opts.numHeadsForLayer(0); got != 8 {
		t.Fatalf("layer 0 heads = %d, want 8", got)
	}
	if got := opts.numHeadsForLayer(1); got != 16 {
		t.Fatalf("layer 1 heads = %d, want 16", got)
	}
	if opts.layerIsSliding(0) {
		t.Fatal("layer 0 should be full attention")
	}
	if !opts.layerIsSliding(1) {
		t.Fatal("layer 1 should be sliding attention")
	}
	if opts.layerUsesMoE(0) {
		t.Fatal("layer 0 should be dense")
	}
	if !opts.layerUsesMoE(1) {
		t.Fatal("layer 1 should use MoE")
	}
	if opts.fullRopeDim != 8 || opts.swaRopeDim != 16 {
		t.Fatalf("rope dims = full %d swa %d, want 8/16", opts.fullRopeDim, opts.swaRopeDim)
	}
}

func TestNewOptionsYarnAttentionFactorFallback(t *testing.T) {
	cfg := testConfig{
		"laguna.block_count":             uint32(1),
		"laguna.embedding_length":        uint32(128),
		"laguna.attention.key_length":    uint32(16),
		"laguna.attention.head_count":    uint32(8),
		"laguna.attention.head_count_kv": uint32(4),
		"laguna.rope.scaling.type":       "yarn",
		"laguna.rope.scaling.factor":     float32(32),
	}

	opts := newOptions(cfg, 1)
	want := float32(0.1*math.Log(32) + 1)
	if got := opts.fullRopeAttentionFactor; math.Abs(float64(got-want)) > 1e-6 {
		t.Fatalf("fullRopeAttentionFactor = %v, want %v", got, want)
	}
}

func TestNewRejectsUnsupportedLagunaVariants(t *testing.T) {
	tests := []struct {
		name string
		cfg  testConfig
	}{
		{
			name: "attention sinks",
			cfg: testConfig{
				"laguna.attention.sink_enabled": true,
			},
		},
		{
			name: "non per-head gate",
			cfg: testConfig{
				"laguna.attention.gating_type": uint32(0),
			},
		},
		{
			name: "missing qk norm",
			cfg: testConfig{
				"laguna.attention.gating_type": uint32(1),
			},
		},
		{
			name: "non sigmoid experts",
			cfg: testConfig{
				"laguna.attention.gating_type": uint32(1),
				"laguna.attention.qk_norm":     true,
				"laguna.expert_gating_func":    uint32(1),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := New(tt.cfg); err == nil {
				t.Fatal("expected unsupported variant error")
			}
		})
	}
}
