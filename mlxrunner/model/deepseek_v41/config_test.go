package deepseek_v41

import (
	"encoding/json"
	"strings"
	"testing"
)

func testConfig() Config {
	ratios := make([]int, 43)
	for i := 2; i < 20; i++ {
		ratios[i] = 2
	}
	for i := 20; i < 40; i++ {
		ratios[i] = 1
	}
	return Config{
		ModelType: "deepseek_v41_text", VocabSize: 129280, HiddenSize: 5120,
		MoeIntermediateSize: 2304, NumHiddenLayers: 40, NumAttentionHeads: 64,
		NumKeyValueHeads: 1, HeadDim: 512, QKRoPEHeadDim: 64, QLoRARank: 1280,
		OLoRARank: 1024, OGroups: 8, RMSNormEps: 1e-20,
		HiddenAct: "silu", SwiGLULimit: 10, UseCache: true,
		MaxPositionEmbeddings: 1048576, RoPETheta: 10000,
		RoPEScaling:      ropeScaling{Type: "yarn", Factor: 16, OriginalMaxPositionEmbeddings: 65536},
		NumRoutedExperts: 384, NumSharedExperts: 1, NumExpertsPerToken: 6,
		ScoringFunc: "sqrtsoftplus", TopKMethod: "noaux_tc", NormTopKProb: true,
		RoutedScalingFactor: 1.5,
		SlidingWindow:       128, CompressRatios: ratios, CompressRoPETheta: 160000,
		KVSourceLayerIDs:    []int{2, 8, 14, 20},
		IndexSourceLayerIDs: []int{2, 8, 14, 20, 24, 28, 32, 36},
		IndexNumHeads:       32, IndexHeadDim: 128, IndexTopK: 512,
		CandidateSourceLayerID: 20, CandidateTopKBlocks: 2048, CandidateBlockSize: 8,
		HCMult: 4, HCSinkhornIters: 20, HCEps: 1e-6,
		EngramLayerIDs: []int{1, 14}, EngramNumEmbeddings: []int64{384006168, 384016682},
		EngramMaxNGramSize: 4, EngramVocabSize: 16000000, EngramNumHeads: 8,
		EngramHeadDim: 256, EngramPadTokenID: 2, EngramCompressedVocabSize: 99092,
		NumNextNPredictLayers: 3, DSparkBlockSize: 5,
		DSparkTargetLayerIDs:   []int{37, 38, 39},
		DSparkNumRoutedExperts: 128, DSparkNumExpertsPerToken: 3,
	}
}

func configJSON(t interface {
	Helper()
	Fatal(...any)
}, cfg Config, q quantizationConfig,
) []byte {
	t.Helper()
	data, err := json.Marshal(struct {
		Architectures []string           `json:"architectures"`
		ModelType     string             `json:"model_type"`
		Quantization  quantizationConfig `json:"quantization_config"`
		Text          Config             `json:"text_config"`
	}{[]string{"DeepseekV41ForCausalLM"}, "deepseek_v41", q, cfg})
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func testQuantization() quantizationConfig {
	return quantizationConfig{Method: "fp8", ActivationScheme: "dynamic", WeightBlockSize: []int{32, 32}, ScaleFormat: "ue8m0", ExpertDType: "fp4"}
}

func TestParseConfig(t *testing.T) {
	cfg, err := parseConfig(configJSON(t, testConfig(), testQuantization()))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.NumHiddenLayers != 40 || cfg.EngramNumEmbeddings[1] != 384016682 {
		t.Fatalf("unexpected model dimensions: layers=%d, engram rows=%v", cfg.NumHiddenLayers, cfg.EngramNumEmbeddings)
	}
}

func TestParseConfigRejectsUnsupportedMetadata(t *testing.T) {
	tests := []struct {
		name string
		edit func(*Config, *quantizationConfig)
		want string
	}{
		{"unrecognized scale", func(_ *Config, q *quantizationConfig) { q.ScaleFormat = "float" }, "quantization"},
		{"missing output groups", func(c *Config, _ *quantizationConfig) { c.OGroups = 0 }, "attention dimensions"},
		{"index key wider than cache", func(c *Config, _ *quantizationConfig) { c.IndexHeadDim = c.HeadDim + 1 }, "sparse index"},
		{"KV source without index key", func(c *Config, _ *quantizationConfig) { c.IndexSourceLayerIDs = c.IndexSourceLayerIDs[1:] }, "KV source layer 2"},
		{"index source changes ratio", func(c *Config, _ *quantizationConfig) { c.CompressRatios[24] = 2 }, "matching KV source"},
		{"candidate source without indexer", func(c *Config, _ *quantizationConfig) { c.CandidateSourceLayerID = 23 }, "candidate source layer"},
		{"missing compression layer", func(c *Config, _ *quantizationConfig) { c.CompressRatios = c.CompressRatios[:40] }, "compress_ratios"},
		{"duplicate engram layer", func(c *Config, _ *quantizationConfig) { c.EngramLayerIDs[1] = c.EngramLayerIDs[0] }, "duplicate"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg, q := testConfig(), testQuantization()
			tt.edit(&cfg, &q)
			_, err := parseConfig(configJSON(t, cfg, q))
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("parseConfig error = %v, want %q", err, tt.want)
			}
		})
	}
}

func FuzzParseDeepSeekV41Config(f *testing.F) {
	f.Add(configJSON(f, testConfig(), testQuantization()))
	f.Add([]byte(`{"architectures":[],"text_config":{}}`))
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) > 1<<20 {
			t.Skip("config exceeds fuzz input budget")
		}
		cfg, err := parseConfig(data)
		if err == nil && (cfg.NumHiddenLayers <= 0 || cfg.VocabSize <= 0 || len(cfg.EngramLayerIDs) == 0) {
			t.Fatalf("accepted config with invalid dimensions: layers=%d vocab=%d engram=%v", cfg.NumHiddenLayers, cfg.VocabSize, cfg.EngramLayerIDs)
		}
	})
}
