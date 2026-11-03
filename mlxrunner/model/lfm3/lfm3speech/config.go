// Package lfm3speech implements the speech-output half of LFM3-Audio and
// LFM3-Omni: the audio-code feedback embedding, the MaskGIT depth decoder that
// turns a backbone hidden state into one frame of codec codes, and the
// LiquidCodec decoder that turns code frames into 24 kHz audio.
package lfm3speech

import (
	"encoding/json"
	"fmt"
	"slices"
)

// Config holds the speech-output settings of a checkpoint's config.json.
type Config struct {
	NumCodebooks int
	CodebookSize int
	// EOSCode in any codebook ends the speech segment.
	EOSCode int32
	// Schedule is how many codebooks each MaskGIT pass reveals.
	Schedule []int
	// InterleavedText and InterleavedAudio are the slot counts of one
	// text block and one speech block in interleaved generation.
	InterleavedText  int
	InterleavedAudio int
	Depth            DepthConfig
}

// DepthConfig describes the bidirectional LFM3 trunk of the depth decoder.
type DepthConfig struct {
	HiddenSize       int32    `json:"hidden_size"`
	IntermediateSize int32    `json:"intermediate_size"`
	NumHiddenLayers  int32    `json:"num_hidden_layers"`
	NumHeads         int32    `json:"num_attention_heads"`
	NumKVHeads       int32    `json:"num_key_value_heads"`
	HeadDim          int32    `json:"head_dim"`
	NormEps          float32  `json:"norm_eps"`
	MaxPositions     int32    `json:"max_position_embeddings"`
	LayerTypes       []string `json:"-"`
	UseSandwichNorm  bool     `json:"-"`
	RopeTheta        float32  `json:"-"`
}

type depthSource struct {
	DepthConfig
	LayerTypes      []string `json:"layer_types"`
	UseSandwichNorm *bool    `json:"use_sandwich_norm"`
	RopeParameters  map[string]struct {
		RopeType            string   `json:"rope_type"`
		RopeTheta           float32  `json:"rope_theta"`
		PartialRotaryFactor *float32 `json:"partial_rotary_factor"`
	} `json:"rope_parameters"`
}

// ParseConfig reads the speech settings from either published layout: the
// Omni layout (depth_config, num_codebooks, codebook_size, audio_eos_code,
// maskgit_schedule) or the Audio training-export layout (depthformer,
// codebooks, audio_vocab_size, maskgit_steps).
func ParseConfig(data []byte) (*Config, error) {
	var raw struct {
		DepthConfig      *depthSource `json:"depth_config"`
		Depthformer      *depthSource `json:"depthformer"`
		NumCodebooks     *int         `json:"num_codebooks"`
		Codebooks        *int         `json:"codebooks"`
		CodebookSize     *int         `json:"codebook_size"`
		AudioVocabSize   *int         `json:"audio_vocab_size"`
		AudioEOSCode     *int32       `json:"audio_eos_code"`
		MaskgitSchedule  []int        `json:"maskgit_schedule"`
		MaskgitSteps     *int         `json:"maskgit_steps"`
		InterleavedText  *int         `json:"interleaved_n_text"`
		InterleavedAudio *int         `json:"interleaved_n_audio"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("parse LFM3 speech config: %w", err)
	}
	cfg := &Config{}
	var depth *depthSource
	switch {
	case raw.DepthConfig != nil && raw.Depthformer == nil:
		depth = raw.DepthConfig
		if raw.NumCodebooks == nil || raw.CodebookSize == nil || raw.AudioEOSCode == nil || raw.MaskgitSchedule == nil {
			return nil, fmt.Errorf("LFM3 speech config needs num_codebooks, codebook_size, audio_eos_code and maskgit_schedule")
		}
		cfg.NumCodebooks, cfg.CodebookSize, cfg.EOSCode = *raw.NumCodebooks, *raw.CodebookSize, *raw.AudioEOSCode
		cfg.Schedule = slices.Clone(raw.MaskgitSchedule)
	case raw.Depthformer != nil && raw.DepthConfig == nil:
		// The training export derives the EOS code and an even schedule.
		depth = raw.Depthformer
		if raw.Codebooks == nil || raw.AudioVocabSize == nil || raw.MaskgitSteps == nil {
			return nil, fmt.Errorf("LFM3 audio config needs codebooks, audio_vocab_size and maskgit_steps")
		}
		cfg.NumCodebooks, cfg.CodebookSize = *raw.Codebooks, *raw.AudioVocabSize
		cfg.EOSCode = int32(cfg.CodebookSize - 1)
		steps := *raw.MaskgitSteps
		if steps < 1 || steps > cfg.NumCodebooks {
			return nil, fmt.Errorf("maskgit_steps %d out of range for %d codebooks", steps, cfg.NumCodebooks)
		}
		cfg.Schedule = make([]int, steps)
		for i := range cfg.Schedule {
			cfg.Schedule[i] = cfg.NumCodebooks / steps
		}
		cfg.Schedule[steps-1] += cfg.NumCodebooks % steps
	default:
		return nil, fmt.Errorf("LFM3 speech config needs exactly one of depth_config or depthformer")
	}
	if raw.InterleavedText == nil || raw.InterleavedAudio == nil {
		return nil, fmt.Errorf("LFM3 speech config needs interleaved_n_text and interleaved_n_audio")
	}
	cfg.InterleavedText, cfg.InterleavedAudio = *raw.InterleavedText, *raw.InterleavedAudio

	cfg.Depth = depth.DepthConfig
	cfg.Depth.LayerTypes = depth.LayerTypes
	if depth.UseSandwichNorm == nil {
		return nil, fmt.Errorf("depth decoder config needs use_sandwich_norm")
	}
	cfg.Depth.UseSandwichNorm = *depth.UseSandwichNorm
	rope, ok := depth.RopeParameters["full_attention"]
	if !ok || len(depth.RopeParameters) != 1 {
		return nil, fmt.Errorf("depth decoder needs exactly full_attention rope parameters")
	}
	if rope.RopeType != "default" {
		return nil, fmt.Errorf("unsupported depth decoder rope type %q", rope.RopeType)
	}
	if p := rope.PartialRotaryFactor; p != nil && *p != 1 {
		return nil, fmt.Errorf("unsupported depth decoder partial rotary factor %v", *p)
	}
	cfg.Depth.RopeTheta = rope.RopeTheta
	return cfg, cfg.validate()
}

func (c *Config) validate() error {
	if c.NumCodebooks < 1 || c.CodebookSize < 2 || c.EOSCode < 0 || int(c.EOSCode) >= c.CodebookSize {
		return fmt.Errorf("invalid LFM3 codebooks: %d of size %d, EOS %d", c.NumCodebooks, c.CodebookSize, c.EOSCode)
	}
	total := 0
	for _, n := range c.Schedule {
		if n < 1 {
			return fmt.Errorf("invalid maskgit_schedule %v", c.Schedule)
		}
		total += n
	}
	if total != c.NumCodebooks {
		return fmt.Errorf("maskgit_schedule %v does not cover %d codebooks", c.Schedule, c.NumCodebooks)
	}
	if c.InterleavedText < 1 || c.InterleavedAudio < 1 {
		return fmt.Errorf("invalid interleave sizes %d/%d", c.InterleavedText, c.InterleavedAudio)
	}
	d := c.Depth
	if d.HiddenSize <= 0 || d.IntermediateSize <= 0 || d.NumHiddenLayers <= 0 || d.NormEps <= 0 || d.RopeTheta <= 0 {
		return fmt.Errorf("invalid depth decoder dimensions %+v", d)
	}
	if d.HeadDim <= 0 || d.HeadDim%2 != 0 || d.NumHeads <= 0 || d.NumKVHeads <= 0 || d.NumHeads%d.NumKVHeads != 0 {
		return fmt.Errorf("invalid depth decoder heads: %d query, %d kv, width %d", d.NumHeads, d.NumKVHeads, d.HeadDim)
	}
	if int(d.MaxPositions) < c.NumCodebooks {
		return fmt.Errorf("depth decoder max_position_embeddings %d < %d codebooks", d.MaxPositions, c.NumCodebooks)
	}
	if len(d.LayerTypes) != int(d.NumHiddenLayers) {
		return fmt.Errorf("depth decoder has %d layer types for %d layers", len(d.LayerTypes), d.NumHiddenLayers)
	}
	for i, t := range d.LayerTypes {
		if t != "full_attention" {
			return fmt.Errorf("depth decoder layer %d is %q, want full_attention", i, t)
		}
	}
	if d.UseSandwichNorm {
		return fmt.Errorf("depth decoder with sandwich norms is not supported")
	}
	return nil
}

// CodecConfig is the LiquidCodec config.json shipped in the codec subfolder.
type CodecConfig struct {
	EncoderDim   int   `json:"encoder_dim"`
	NFFT         int   `json:"stft_n_fft"`
	HopLength    int   `json:"stft_hop_length"`
	WinLength    int   `json:"stft_win_length"`
	FrameStack   int   `json:"stft_frame_stack"`
	QuantizerDim int   `json:"quantizer_dim"`
	FSQLevels    []int `json:"fsq_levels"`
	NumCodebooks int   `json:"n_q"`
	Window       int   `json:"decoder_window"`
	SampleRate   int   `json:"sampling_rate"`
}

// ParseCodecConfig reads and validates a LiquidCodec config.json.
func ParseCodecConfig(data []byte) (*CodecConfig, error) {
	var raw struct {
		CodecConfig
		ModelType string `json:"model_type"`
		Nested    any    `json:"model"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("parse LiquidCodec config: %w", err)
	}
	if raw.ModelType != "liquid_codec" {
		return nil, fmt.Errorf("codec model_type %q, want liquid_codec", raw.ModelType)
	}
	if raw.Nested != nil {
		return nil, fmt.Errorf("nested LiquidCodec config layout is not supported")
	}
	c := raw.CodecConfig
	if c.NFFT <= 0 || c.NFFT%2 != 0 || c.WinLength != c.NFFT || c.HopLength <= 0 || c.WinLength%c.HopLength != 0 || 2*c.HopLength > c.WinLength {
		return nil, fmt.Errorf("unsupported codec STFT: n_fft %d, win %d, hop %d", c.NFFT, c.WinLength, c.HopLength)
	}
	if (c.WinLength-c.HopLength)%2 != 0 {
		return nil, fmt.Errorf("codec STFT trim is not whole: win %d hop %d", c.WinLength, c.HopLength)
	}
	// One code frame must cover the leading trim for streaming to emit.
	if c.HopLength*c.FrameStack < (c.WinLength-c.HopLength)/2 {
		return nil, fmt.Errorf("codec frame of %d samples is shorter than its trim", c.HopLength*c.FrameStack)
	}
	if c.FrameStack <= 0 || c.EncoderDim <= 0 || c.QuantizerDim <= 0 || c.NumCodebooks <= 0 || c.Window <= 0 || c.SampleRate <= 0 {
		return nil, fmt.Errorf("invalid codec config %+v", c)
	}
	if len(c.FSQLevels) == 0 {
		return nil, fmt.Errorf("codec config has no fsq_levels")
	}
	for _, l := range c.FSQLevels {
		if l < 2 {
			return nil, fmt.Errorf("invalid fsq level %d", l)
		}
	}
	return &c, nil
}

// CodebookSize is the number of distinct codes per codebook.
func (c *CodecConfig) CodebookSize() int {
	n := 1
	for _, l := range c.FSQLevels {
		n *= l
	}
	return n
}

// FrameSamples is the number of output samples per code frame.
func (c *CodecConfig) FrameSamples() int { return c.HopLength * c.FrameStack }
