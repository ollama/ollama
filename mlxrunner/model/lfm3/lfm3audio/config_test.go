package lfm3audio

import (
	"fmt"
	"slices"
	"strings"
	"testing"
)

const encoderLFM2 = `{
	"hidden_size": 768, "intermediate_size": 2560, "num_hidden_layers": 4,
	"num_attention_heads": 8, "num_key_value_heads": 4, "norm_eps": 1e-05,
	"conv_L_cache": 3, "conv_bias": false,
	"block_auto_adjust_ff_dim": true, "block_ffn_dim_multiplier": 1.0, "block_multiple_of": 256,
	"layer_types": ["conv", "conv", "full_attention", "full_attention"],
	"rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"}
}`

var omniConfig = `{
	"audio_encoder_config": {"conv_channels": 256, "hidden_size": 768, "num_mel_bins": 128, "sliding_window": 32,
		"text_config": ` + encoderLFM2 + `},
	"text_config": {"hidden_size": 2048}
}`

const omniProcessor = `{"feature_extractor": {"encoder_stride": 8, "feature_size": 128, "hop_length": 160,
	"log_floor": -12.0, "n_fft": 400, "normalization_divisor": 4.0, "normalization_offset": 4.0,
	"sampling_rate": 16000, "win_length": 400}}`

var audioConfig = `{
	"encoder": {"dim_in": 128, "dim_conv": 256, "dim_out": 768, "sliding_window": 32, "lfm": ` + encoderLFM2 + `},
	"lfm": {"hidden_size": 2048},
	"preprocessor": {"feature_type": "log_mel", "sample_rate": 16000, "channel_reduction": "mean", "n_mels": 128,
		"n_fft": 400, "win_length": 400, "hop_length": 160, "window": "hann", "window_periodic": true,
		"center": false, "normalized": false, "onesided": true, "power": 2.0, "f_min": 0.0, "f_max": 8000.0,
		"mel_scale": "slaney", "mel_norm": "slaney", "log_base": 10, "logmel_floor": -12.0,
		"normalization": {"type": "affine", "offset": 4.0, "divisor": 4.0}}
}`

func TestParseConfigLayouts(t *testing.T) {
	want := Config{
		SampleRate: 16000, NumMelBins: 128, NFFT: 400, HopLength: 160, WinLength: 400,
		LogFloor: -12, NormOffset: 4, NormDivisor: 4, EncoderStride: 8,
		ConvChannels: 256, HiddenSize: 768, IntermediateSize: 1792,
		NumAttentionHeads: 8, NumKeyValueHeads: 4, HeadDim: 96, ConvKernel: 3,
		NormEps: 1e-5, RopeTheta: 1e6, SlidingWindow: 32,
		LayerTypes: []string{"conv", "conv", "full_attention", "full_attention"},
		OutputSize: 2048, ProjectorNormEps: 1e-5,
	}
	for _, tc := range []struct {
		name            string
		config, process string
	}{
		{"omni", omniConfig, omniProcessor},
		{"omni without processor", omniConfig, ""},
		{"audio", audioConfig, ""},
		{"audio with flat preprocessor", audioConfig, `{"feature_size": 128, "sampling_rate": 16000, "encoder_stride": 8}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ParseConfig([]byte(tc.config), []byte(tc.process))
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(got.LayerTypes, want.LayerTypes) {
				t.Fatalf("layer types %v", got.LayerTypes)
			}
			got.LayerTypes = want.LayerTypes
			if fmt.Sprintf("%+v", got) != fmt.Sprintf("%+v", want) {
				t.Fatalf("got  %+v\nwant %+v", got, want)
			}
		})
	}
}

func TestParseConfigRejects(t *testing.T) {
	for _, tc := range []struct {
		name, config, processor, err string
	}{
		{"no encoder", `{"text_config": {"hidden_size": 2048}}`, "", "neither"},
		{"centered stft", strings.Replace(audioConfig, `"center": false`, `"center": true`, 1), "", "center"},
		{"htk mel", strings.Replace(audioConfig, `"mel_scale": "slaney"`, `"mel_scale": "htk"`, 1), "", "mel_scale"},
		{"sample rate", omniConfig, `{"feature_extractor": {"sampling_rate": 24000}}`, "sample rate"},
		{"mel mismatch", omniConfig, `{"feature_extractor": {"feature_size": 80}}`, "mel bins"},
		{"layer type", strings.Replace(omniConfig, `"conv", "conv"`, `"conv", "sliding_attention"`, 1), "", "layer type"},
		{"layer count", strings.Replace(omniConfig, `"num_hidden_layers": 4`, `"num_hidden_layers": 5`, 1), "", "layer_types"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := ParseConfig([]byte(tc.config), []byte(tc.processor))
			if err == nil || !strings.Contains(err.Error(), tc.err) {
				t.Fatalf("err = %v, want it to mention %q", err, tc.err)
			}
		})
	}
}
