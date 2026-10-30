package lfm3speech

import (
	"slices"
	"strings"
	"testing"
)

const omniConfig = `{
 "num_codebooks": 12, "codebook_size": 2017, "audio_eos_code": 2016, "maskgit_schedule": [4, 4, 4],
 "interleaved_n_text": 6, "interleaved_n_audio": 12,
 "depth_config": {"hidden_size": 1024, "intermediate_size": 2816, "num_hidden_layers": 3,
  "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 32, "norm_eps": 1e-05,
  "max_position_embeddings": 12, "layer_types": ["full_attention", "full_attention", "full_attention"],
  "rope_parameters": {"full_attention": {"partial_rotary_factor": 1.0, "rope_theta": 1000000.0, "rope_type": "default"}},
  "use_sandwich_norm": false}
}`

const audioConfig = `{
 "codebooks": 12, "audio_vocab_size": 2017, "maskgit_steps": 3,
 "interleaved_n_text": 6, "interleaved_n_audio": 12,
 "depthformer": {"hidden_size": 1024, "intermediate_size": 2816, "num_hidden_layers": 3,
  "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 32, "norm_eps": 1e-05,
  "max_position_embeddings": 12, "layer_types": ["full_attention", "full_attention", "full_attention"],
  "rope_parameters": {"full_attention": {"rope_type": "default", "rope_theta": 1000000.0, "partial_rotary_factor": 1.0}},
  "use_sandwich_norm": false, "is_causal": false}
}`

func TestParseConfigLayoutsAgree(t *testing.T) {
	omni, err := ParseConfig([]byte(omniConfig))
	if err != nil {
		t.Fatal(err)
	}
	audio, err := ParseConfig([]byte(audioConfig))
	if err != nil {
		t.Fatal(err)
	}
	for _, cfg := range []*Config{omni, audio} {
		if cfg.NumCodebooks != 12 || cfg.CodebookSize != 2017 || cfg.EOSCode != 2016 || !slices.Equal(cfg.Schedule, []int{4, 4, 4}) {
			t.Fatalf("codebooks: %+v", cfg)
		}
		d := cfg.Depth
		if d.HiddenSize != 1024 || d.HeadDim != 32 || d.NumHeads != 32 || d.NumKVHeads != 8 || d.RopeTheta != 1e6 || d.NumHiddenLayers != 3 {
			t.Fatalf("depth: %+v", d)
		}
	}
}

func TestParseConfigAudioSchedule(t *testing.T) {
	// An uneven step count puts the remainder on the last pass.
	cfg, err := ParseConfig([]byte(strings.Replace(audioConfig, `"maskgit_steps": 3`, `"maskgit_steps": 5`, 1)))
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(cfg.Schedule, []int{2, 2, 2, 2, 4}) {
		t.Fatalf("schedule %v", cfg.Schedule)
	}
}

func TestParseConfigRejects(t *testing.T) {
	for name, change := range map[string][3]string{
		"schedule sum":  {`[4, 4, 4]`, `[4, 4, 3]`, "does not cover 12 codebooks"},
		"eos range":     {`"audio_eos_code": 2016`, `"audio_eos_code": 2017`, "EOS 2017"},
		"causal layer":  {`"layer_types": ["full_attention", "full_attention", "full_attention"]`, `"layer_types": ["full_attention", "sliding_attention", "full_attention"]`, "layer 1 is \"sliding_attention\""},
		"partial rope":  {`"partial_rotary_factor": 1.0`, `"partial_rotary_factor": 0.5`, "partial rotary factor 0.5"},
		"sandwich":      {`"use_sandwich_norm": false`, `"use_sandwich_norm": true`, "sandwich norms"},
		"heads":         {`"num_key_value_heads": 8`, `"num_key_value_heads": 5`, "5 kv"},
		"no interleave": {`"interleaved_n_text": 6,`, ``, "needs interleaved_n_text"},
		"both layouts":  {`"depth_config"`, `"depthformer": {}, "depth_config"`, "exactly one of depth_config or depthformer"},
	} {
		t.Run(name, func(t *testing.T) {
			changed := strings.Replace(omniConfig, change[0], change[1], 1)
			if changed == omniConfig {
				t.Fatal("test edit did not apply")
			}
			if _, err := ParseConfig([]byte(changed)); err == nil || !strings.Contains(err.Error(), change[2]) {
				t.Fatalf("error %v, want one mentioning %q", err, change[2])
			}
		})
	}
}

// The released codec config, minus fields the decoder does not read.
const codecConfig = `{"architectures": ["LiquidCodecModel"], "model_type": "liquid_codec", "encoder_dim": 512,
 "encoder_conv_dim": 256, "stft_n_fft": 1280, "stft_hop_length": 320, "stft_win_length": 1280, "stft_frame_stack": 6,
 "quantizer_dim": 256, "fsq_levels": [8, 7, 6, 6], "n_q": 12, "decoder_window": 30, "sampling_rate": 24000,
 "sample_rate": 24000, "frame_samples": 1920}`

func TestParseCodecConfig(t *testing.T) {
	c, err := ParseCodecConfig([]byte(codecConfig))
	if err != nil {
		t.Fatal(err)
	}
	if c.CodebookSize() != 2016 || c.FrameSamples() != 1920 || c.Window != 30 || c.SampleRate != 24000 {
		t.Fatalf("codec config %+v", c)
	}
	for name, change := range map[string][3]string{
		"model type": {`"liquid_codec"`, `"other"`, `model_type "other"`},
		"window":     {`"stft_win_length": 1280`, `"stft_win_length": 1000`, "win 1000"},
		"levels":     {`[8, 7, 6, 6]`, `[8, 1]`, "fsq level 1"},
		"nested":     {`"n_q": 12`, `"n_q": 12, "model": {}`, "nested LiquidCodec config"},
	} {
		changed := strings.Replace(codecConfig, change[0], change[1], 1)
		if changed == codecConfig {
			t.Fatalf("%s: test edit did not apply", name)
		}
		if _, err := ParseCodecConfig([]byte(changed)); err == nil || !strings.Contains(err.Error(), change[2]) {
			t.Errorf("%s: error %v, want one mentioning %q", name, err, change[2])
		}
	}
}
