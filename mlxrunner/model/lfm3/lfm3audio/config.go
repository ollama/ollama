// Package lfm3audio implements the speech-input path shared by LFM3-Audio and
// LFM3-Omni: log-mel features, the strided-convolution downsampler, the LFM2
// speech encoder, and the projector into the language model.
package lfm3audio

import (
	"encoding/json"
	"fmt"
)

// Config holds the front-end and encoder hyperparameters.
type Config struct {
	SampleRate    int
	NumMelBins    int
	NFFT          int
	HopLength     int
	WinLength     int
	LogFloor      float64
	NormOffset    float64
	NormDivisor   float64
	EncoderStride int

	ConvChannels      int32
	HiddenSize        int32
	IntermediateSize  int32
	NumAttentionHeads int32
	NumKeyValueHeads  int32
	HeadDim           int32
	ConvKernel        int32
	NormEps           float32
	RopeTheta         float32
	SlidingWindow     int32
	LayerTypes        []string

	// OutputSize is the language model width the projector maps into.
	OutputSize       int32
	ProjectorNormEps float32
}

type lfm2Config struct {
	HiddenSize            int32    `json:"hidden_size"`
	IntermediateSize      int32    `json:"intermediate_size"`
	NumHiddenLayers       int32    `json:"num_hidden_layers"`
	NumAttentionHeads     int32    `json:"num_attention_heads"`
	NumKeyValueHeads      int32    `json:"num_key_value_heads"`
	NormEps               float32  `json:"norm_eps"`
	ConvLCache            int32    `json:"conv_L_cache"`
	ConvBias              bool     `json:"conv_bias"`
	LayerTypes            []string `json:"layer_types"`
	BlockAutoAdjustFFDim  *bool    `json:"block_auto_adjust_ff_dim"`
	BlockFFNDimMultiplier *float64 `json:"block_ffn_dim_multiplier"`
	BlockMultipleOf       *int32   `json:"block_multiple_of"`
	RopeParameters        struct {
		RopeType  string  `json:"rope_type"`
		RopeTheta float32 `json:"rope_theta"`
	} `json:"rope_parameters"`
}

// encoderConfig accepts the Omni names and the legacy Audio export names.
type encoderConfig struct {
	NumMelBins    int32       `json:"num_mel_bins"`
	ConvChannels  int32       `json:"conv_channels"`
	HiddenSize    int32       `json:"hidden_size"`
	SlidingWindow int32       `json:"sliding_window"`
	TextConfig    *lfm2Config `json:"text_config"`

	DimIn   int32       `json:"dim_in"`
	DimConv int32       `json:"dim_conv"`
	DimOut  int32       `json:"dim_out"`
	LFM     *lfm2Config `json:"lfm"`
}

type legacyPreprocessor struct {
	FeatureType    *string  `json:"feature_type"`
	SampleRate     *int     `json:"sample_rate"`
	NMels          *int     `json:"n_mels"`
	NFFT           *int     `json:"n_fft"`
	WinLength      *int     `json:"win_length"`
	HopLength      *int     `json:"hop_length"`
	Window         *string  `json:"window"`
	WindowPeriodic *bool    `json:"window_periodic"`
	Center         *bool    `json:"center"`
	Normalized     *bool    `json:"normalized"`
	Onesided       *bool    `json:"onesided"`
	Power          *float64 `json:"power"`
	FMin           *float64 `json:"f_min"`
	FMax           *float64 `json:"f_max"`
	MelScale       *string  `json:"mel_scale"`
	MelNorm        *string  `json:"mel_norm"`
	LogBase        *float64 `json:"log_base"`
	LogmelFloor    *float64 `json:"logmel_floor"`
	ChannelReduce  *string  `json:"channel_reduction"`
	Normalization  *struct {
		Type    string   `json:"type"`
		Offset  *float64 `json:"offset"`
		Divisor *float64 `json:"divisor"`
	} `json:"normalization"`
}

type featureExtractorConfig struct {
	FeatureSize          *int     `json:"feature_size"`
	SamplingRate         *int     `json:"sampling_rate"`
	NFFT                 *int     `json:"n_fft"`
	HopLength            *int     `json:"hop_length"`
	WinLength            *int     `json:"win_length"`
	LogFloor             *float64 `json:"log_floor"`
	NormalizationOffset  *float64 `json:"normalization_offset"`
	NormalizationDivisor *float64 `json:"normalization_divisor"`
	EncoderStride        *int     `json:"encoder_stride"`
}

// ParseConfig reads the speech-input configuration from a checkpoint's
// config.json: the Omni layout (audio_encoder_config, text_config) or the
// legacy Audio export layout (encoder, lfm, preprocessor). processorJSON is
// the optional processor_config.json or preprocessor_config.json holding the
// feature extractor settings; absent fields keep the publisher defaults.
func ParseConfig(configJSON, processorJSON []byte) (Config, error) {
	var root struct {
		AudioEncoder *encoderConfig `json:"audio_encoder_config"`
		Encoder      *encoderConfig `json:"encoder"`
		TextConfig   *struct {
			HiddenSize int32 `json:"hidden_size"`
		} `json:"text_config"`
		LFM *struct {
			HiddenSize int32 `json:"hidden_size"`
		} `json:"lfm"`
		Preprocessor *legacyPreprocessor `json:"preprocessor"`
	}
	if err := json.Unmarshal(configJSON, &root); err != nil {
		return Config{}, fmt.Errorf("parse LFM3 audio config: %w", err)
	}

	cfg := Config{
		SampleRate:       16000,
		NumMelBins:       128,
		NFFT:             400,
		HopLength:        160,
		WinLength:        400,
		LogFloor:         -12,
		NormOffset:       4,
		NormDivisor:      4,
		EncoderStride:    8,
		ProjectorNormEps: 1e-5,
	}

	enc := root.AudioEncoder
	switch {
	case enc != nil && root.TextConfig != nil:
		cfg.OutputSize = root.TextConfig.HiddenSize
	case root.Encoder != nil && root.LFM != nil:
		enc = root.Encoder
		cfg.OutputSize = root.LFM.HiddenSize
		if enc.TextConfig == nil {
			enc.TextConfig = enc.LFM
		}
		if enc.NumMelBins == 0 {
			enc.NumMelBins = enc.DimIn
		}
		if enc.ConvChannels == 0 {
			enc.ConvChannels = enc.DimConv
		}
		if enc.HiddenSize == 0 {
			enc.HiddenSize = enc.DimOut
		}
	default:
		return Config{}, fmt.Errorf("LFM3 audio config has neither audio_encoder_config nor a legacy encoder block")
	}
	if enc.TextConfig == nil {
		return Config{}, fmt.Errorf("LFM3 audio encoder config has no LFM2 model block")
	}
	if err := cfg.applyEncoder(enc); err != nil {
		return Config{}, err
	}

	if p := root.Preprocessor; p != nil {
		if err := cfg.applyLegacyPreprocessor(p); err != nil {
			return Config{}, err
		}
	}
	if len(processorJSON) > 0 {
		var nested struct {
			FeatureExtractor *featureExtractorConfig `json:"feature_extractor"`
		}
		if err := json.Unmarshal(processorJSON, &nested); err != nil {
			return Config{}, fmt.Errorf("parse LFM3 audio processor config: %w", err)
		}
		fe := nested.FeatureExtractor
		if fe == nil {
			fe = &featureExtractorConfig{}
			if err := json.Unmarshal(processorJSON, fe); err != nil {
				return Config{}, fmt.Errorf("parse LFM3 audio processor config: %w", err)
			}
		}
		cfg.applyFeatureExtractor(fe)
	}
	if err := cfg.validate(enc.NumMelBins); err != nil {
		return Config{}, err
	}
	return cfg, nil
}

func (c *Config) applyEncoder(enc *encoderConfig) error {
	t := enc.TextConfig
	c.ConvChannels = enc.ConvChannels
	c.HiddenSize = enc.HiddenSize
	c.SlidingWindow = enc.SlidingWindow
	if t.HiddenSize != enc.HiddenSize {
		return fmt.Errorf("LFM3 audio encoder width %d differs from its LFM2 width %d", enc.HiddenSize, t.HiddenSize)
	}
	if t.NumAttentionHeads <= 0 || t.NumKeyValueHeads <= 0 || t.NumAttentionHeads%t.NumKeyValueHeads != 0 || t.HiddenSize%t.NumAttentionHeads != 0 {
		return fmt.Errorf("invalid LFM3 audio encoder heads: query=%d kv=%d width=%d", t.NumAttentionHeads, t.NumKeyValueHeads, t.HiddenSize)
	}
	if t.NumHiddenLayers <= 0 || len(t.LayerTypes) != int(t.NumHiddenLayers) {
		return fmt.Errorf("LFM3 audio encoder layer_types has %d entries for %d layers", len(t.LayerTypes), t.NumHiddenLayers)
	}
	for i, kind := range t.LayerTypes {
		if kind != "conv" && kind != "full_attention" {
			return fmt.Errorf("unsupported LFM3 audio encoder layer type %q at %d", kind, i)
		}
	}
	if t.ConvBias {
		return fmt.Errorf("LFM3 audio encoder convolution bias is not supported")
	}
	if rt := t.RopeParameters.RopeType; rt != "" && rt != "default" {
		return fmt.Errorf("unsupported LFM3 audio encoder rope_type %q", rt)
	}
	c.NumAttentionHeads = t.NumAttentionHeads
	c.NumKeyValueHeads = t.NumKeyValueHeads
	c.HeadDim = t.HiddenSize / t.NumAttentionHeads
	c.ConvKernel = t.ConvLCache
	c.NormEps = t.NormEps
	c.RopeTheta = t.RopeParameters.RopeTheta
	c.LayerTypes = t.LayerTypes

	// LFM2 shrinks the configured feed-forward width to 2/3, scaled and
	// rounded up to a multiple; absent fields take the LFM2 defaults.
	c.IntermediateSize = t.IntermediateSize
	if t.BlockAutoAdjustFFDim == nil || *t.BlockAutoAdjustFFDim {
		multiplier, multiple := 1.0, int32(256)
		if t.BlockFFNDimMultiplier != nil {
			multiplier = *t.BlockFFNDimMultiplier
		}
		if t.BlockMultipleOf != nil {
			multiple = *t.BlockMultipleOf
		}
		if multiple <= 0 {
			return fmt.Errorf("invalid LFM3 audio encoder block_multiple_of %d", multiple)
		}
		size := int32(multiplier * float64(2*int64(t.IntermediateSize)/3))
		c.IntermediateSize = multiple * ((size + multiple - 1) / multiple)
	}
	return nil
}

func (c *Config) applyLegacyPreprocessor(p *legacyPreprocessor) error {
	for _, check := range []struct {
		name string
		ok   bool
	}{
		{"feature_type", p.FeatureType == nil || *p.FeatureType == "log_mel"},
		{"window", p.Window == nil || *p.Window == "hann"},
		{"window_periodic", p.WindowPeriodic == nil || *p.WindowPeriodic},
		{"center", p.Center == nil || !*p.Center},
		{"normalized", p.Normalized == nil || !*p.Normalized},
		{"onesided", p.Onesided == nil || *p.Onesided},
		{"power", p.Power == nil || *p.Power == 2},
		{"f_min", p.FMin == nil || *p.FMin == 0},
		{"mel_scale", p.MelScale == nil || *p.MelScale == "slaney"},
		{"mel_norm", p.MelNorm == nil || *p.MelNorm == "slaney"},
		{"log_base", p.LogBase == nil || *p.LogBase == 10},
		{"channel_reduction", p.ChannelReduce == nil || *p.ChannelReduce == "mean"},
		{"normalization", p.Normalization == nil || p.Normalization.Type == "" || p.Normalization.Type == "affine"},
	} {
		if !check.ok {
			return fmt.Errorf("unsupported LFM3 audio preprocessor setting %s", check.name)
		}
	}
	setInt(&c.SampleRate, p.SampleRate)
	setInt(&c.NumMelBins, p.NMels)
	setInt(&c.NFFT, p.NFFT)
	setInt(&c.WinLength, p.WinLength)
	setInt(&c.HopLength, p.HopLength)
	setFloat(&c.LogFloor, p.LogmelFloor)
	if n := p.Normalization; n != nil {
		setFloat(&c.NormOffset, n.Offset)
		setFloat(&c.NormDivisor, n.Divisor)
	}
	if p.FMax != nil && *p.FMax != float64(c.SampleRate)/2 {
		return fmt.Errorf("LFM3 audio mel filters must span to Nyquist, got f_max %g", *p.FMax)
	}
	return nil
}

func (c *Config) applyFeatureExtractor(fe *featureExtractorConfig) {
	setInt(&c.NumMelBins, fe.FeatureSize)
	setInt(&c.SampleRate, fe.SamplingRate)
	setInt(&c.NFFT, fe.NFFT)
	setInt(&c.HopLength, fe.HopLength)
	setInt(&c.WinLength, fe.WinLength)
	setFloat(&c.LogFloor, fe.LogFloor)
	setFloat(&c.NormOffset, fe.NormalizationOffset)
	setFloat(&c.NormDivisor, fe.NormalizationDivisor)
	setInt(&c.EncoderStride, fe.EncoderStride)
}

func (c *Config) validate(encoderMelBins int32) error {
	switch {
	case c.SampleRate != 16000:
		return fmt.Errorf("unsupported LFM3 audio sample rate %d", c.SampleRate)
	case c.NFFT <= 0 || c.NFFT%2 != 0 || c.WinLength != c.NFFT || c.HopLength <= 0:
		return fmt.Errorf("unsupported LFM3 audio STFT: n_fft=%d win_length=%d hop_length=%d", c.NFFT, c.WinLength, c.HopLength)
	case c.NumMelBins <= 0 || int32(c.NumMelBins) != encoderMelBins:
		return fmt.Errorf("LFM3 audio feature size %d differs from encoder mel bins %d", c.NumMelBins, encoderMelBins)
	case c.EncoderStride != 8 || c.NumMelBins%8 != 0:
		return fmt.Errorf("unsupported LFM3 audio encoder stride %d for %d mel bins", c.EncoderStride, c.NumMelBins)
	case c.NormDivisor == 0:
		return fmt.Errorf("invalid LFM3 audio log-mel normalization")
	case c.ConvChannels <= 0 || c.HiddenSize <= 0 || c.IntermediateSize <= 0 || c.OutputSize <= 0:
		return fmt.Errorf("invalid LFM3 audio encoder dimensions: conv=%d hidden=%d intermediate=%d output=%d", c.ConvChannels, c.HiddenSize, c.IntermediateSize, c.OutputSize)
	case c.ConvKernel <= 0:
		return fmt.Errorf("invalid LFM3 audio encoder conv_L_cache %d", c.ConvKernel)
	case c.NormEps <= 0 || c.RopeTheta <= 0:
		return fmt.Errorf("invalid LFM3 audio encoder norm_eps %g or rope_theta %g", c.NormEps, c.RopeTheta)
	case c.SlidingWindow <= 0:
		return fmt.Errorf("invalid LFM3 audio encoder sliding_window %d", c.SlidingWindow)
	}
	return nil
}

func setInt(dst *int, v *int) {
	if v != nil {
		*dst = *v
	}
}

func setFloat(dst *float64, v *float64) {
	if v != nil {
		*dst = *v
	}
}
