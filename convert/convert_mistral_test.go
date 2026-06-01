package convert

import "testing"

func TestMistral3KVUsesLlamaCppRopeScalingKeys(t *testing.T) {
	mscale := float32(0.75)
	mscaleAllDim := float32(0)
	temperatureScale := float32(0.125)

	multimodal := &mistral3Model{}
	multimodal.TextModel.NumAttentionHeads = 1
	multimodal.TextModel.HeadDim = 64
	multimodal.TextModel.RopeParameters.BetaFast = 32
	multimodal.TextModel.RopeParameters.BetaSlow = 1
	multimodal.TextModel.RopeParameters.Mscale = &mscale
	multimodal.TextModel.RopeParameters.MscaleAllDim = &mscaleAllDim
	multimodal.TextModel.RopeParameters.Llama4ScalingBeta = &temperatureScale

	causal := &mistral3CausalModel{NumAttentionHeads: 1, HeadDim: 64}
	causal.RopeParameters.BetaFast = 32
	causal.RopeParameters.BetaSlow = 1
	causal.RopeParameters.Mscale = &mscale
	causal.RopeParameters.MscaleAllDim = &mscaleAllDim
	causal.RopeParameters.Llama4ScalingBeta = &temperatureScale

	tests := []struct {
		name string
		kv   KV
	}{
		{name: "multimodal", kv: multimodal.KV(mistralTestTokenizer())},
		{name: "causal", kv: causal.KV(mistralTestTokenizer())},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assertKVEquals(t, tt.kv, "mistral3.rope.scaling.yarn_beta_fast", float32(32))
			assertKVEquals(t, tt.kv, "mistral3.rope.scaling.yarn_beta_slow", float32(1))
			assertKVEquals(t, tt.kv, "mistral3.rope.scaling.yarn_log_multiplier", mscaleAllDim)
			assertKVEquals(t, tt.kv, "mistral3.attention.temperature_scale", temperatureScale)

			for _, key := range []string{
				"mistral3.rope.scaling.beta_fast",
				"mistral3.rope.scaling.beta_slow",
				"mistral3.rope.scaling.mscale",
				"mistral3.rope.scaling.mscale_all_dim",
				"mistral3.rope.scaling_beta",
			} {
				if _, ok := tt.kv[key]; ok {
					t.Fatalf("unexpected legacy key %q", key)
				}
			}
		})
	}
}

func mistralTestTokenizer() *Tokenizer {
	return &Tokenizer{Vocabulary: &Vocabulary{}}
}

func assertKVEquals[T comparable](t *testing.T, kv KV, key string, want T) {
	t.Helper()

	got, ok := kv[key]
	if !ok {
		t.Fatalf("missing key %q", key)
	}
	if got != want {
		t.Fatalf("%s = %v, want %v", key, got, want)
	}
}
