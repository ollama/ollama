package dflash

import (
	"encoding/json"
	"maps"
	"math"
	"slices"
	"testing"
)

func TestDraftConfigBoundsAndCausality(t *testing.T) {
	config := map[string]any{
		"hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 4, "head_dim": 8,
		"is_causal": false, "layer_types": []string{"sliding_attention", "sliding_attention"}, "sliding_window": 16,
		"dflash_config": map[string]any{"block_size": 8, "mask_token_id": 7, "target_layer_ids": []int{1, 3}},
	}
	for _, tt := range []struct {
		name, key string
		value     any
		valid     bool
	}{
		{"publisher noncausal", "is_causal", false, true},
		{"explicit causal", "is_causal", true, true},
		{"negative layers", "num_hidden_layers", -1, false},
		{"too many layers", "num_hidden_layers", maxDraftLayers + 1, false},
		{"zero heads", "num_key_value_heads", 0, false},
		{"mismatched heads", "num_key_value_heads", 3, false},
		{"overflowing projection", "head_dim", 2147483646, false},
		{"overflowing fused projection", "head_dim", 200000000, false},
		{"odd head dim", "head_dim", 7, false},
		{"zero hidden size", "hidden_size", 0, false},
		{"oversized block", "dflash_config", map[string]any{"block_size": maxBlockSize + 1, "mask_token_id": 7, "target_layer_ids": []int{1, 3}}, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			c := maps.Clone(config)
			c[tt.key] = tt.value
			b, err := json.Marshal(c)
			if err != nil {
				t.Fatal(err)
			}
			got, err := parseConfig(b)
			if (err == nil) != tt.valid {
				t.Fatalf("parseConfig error=%v valid=%v", err, tt.valid)
			}
			// Only DFlash2 reads the top-level is_causal; the shared parse must not apply it.
			if tt.valid && (got.Causal != nil || got.isCausal == nil || *got.isCausal != tt.value.(bool)) {
				t.Fatalf("causal=%v is_causal=%v, want only is_causal %v", got.Causal, got.isCausal, tt.value)
			}
		})
	}
}

func FuzzDraftConfig(f *testing.F) {
	f.Add([]byte(`{"hidden_size":32,"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,"is_causal":false,"vocab_size":64,"dflash_config":{"block_size":8,"mask_token_id":7,"target_layer_ids":[1,3],"conv_group_size":16,"conv_kernel_size":2,"selector_rank":4,"selector_top_k":2}}`))
	f.Add([]byte(`{"num_hidden_layers":-1}`))
	// The largest fused attention projection that fits int32.
	f.Add([]byte(`{"hidden_size":32,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"head_dim":715827882,"dflash_config":{"block_size":2,"mask_token_id":0,"target_layer_ids":[0]}}`))
	f.Fuzz(func(t *testing.T, data []byte) {
		c, err := parseConfig(data)
		if err != nil {
			return
		}
		if c.NumHiddenLayers <= 0 || c.NumHiddenLayers > maxDraftLayers || len(c.LayerTypes) != int(c.NumHiddenLayers) || c.BlockSize < 2 || c.BlockSize > maxBlockSize {
			t.Fatalf("unsafe accepted dimensions: %+v", c)
		}
		if c.HiddenSize <= 0 || c.NumAttentionHeads <= 0 || c.NumKeyValueHeads <= 0 || c.HeadDim <= 0 ||
			c.NumAttentionHeads%c.NumKeyValueHeads != 0 || c.HeadDim%2 != 0 ||
			(int64(c.NumAttentionHeads)+2*int64(c.NumKeyValueHeads))*int64(c.HeadDim) > math.MaxInt32 {
			t.Fatalf("unsafe accepted attention dimensions: %+v", c)
		}
		if len(c.TargetLayerIDs) == 0 || !slices.IsSorted(c.TargetLayerIDs) {
			t.Fatalf("invalid target taps: %v", c.TargetLayerIDs)
		}
		if c.validateDFlash2() != nil {
			return
		}
		// LoadWeights sizes the convolutions and the selector from these.
		if c.ConvGroupSize < 1 || int(c.HiddenSize)%c.ConvGroupSize != 0 || c.ConvKernelSize < 1 || c.ConvKernelSize > c.BlockSize ||
			2*int64(c.ConvKernelSize)*int64(c.HiddenSize)/int64(c.ConvGroupSize) > math.MaxInt32 ||
			c.SelectorTopK < 1 || c.SelectorTopK > min(int(c.VocabSize), maxSelectorTopK) || c.SelectorRank < 1 {
			t.Fatalf("unsafe accepted dflash2 dimensions: %+v", c)
		}
	})
}
