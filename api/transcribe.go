package api

import (
	"time"
)

// TranscribeRequest describes a request sent to transcribe audio
type TranscribeRequest struct {
	// Model is the whisper model name to use
	Model string `json:"model"`

	// Audio is the audio data in base64 or raw bytes
	// Supported formats: wav, mp3, flac, ogg (16kHz mono preferred)
	Audio []byte `json:"audio,omitempty"`

	// Language is the source language code (e.g., "en", "fr")
	// Leave empty for auto-detection
	Language string `json:"language,omitempty"`

	// Translate when true, translates to English
	Translate bool `json:"translate,omitempty"`

	// ResponseFormat specifies the output format
	// Options: "json", "text", "srt", "vtt", "verbose_json"
	ResponseFormat string `json:"response_format,omitempty"`

	// TimestampGranularity specifies timestamp detail level
	// Options: "segment", "word"
	TimestampGranularity string `json:"timestamp_granularity,omitempty"`

	// Prompt provides context or previous text to guide the model
	Prompt string `json:"prompt,omitempty"`

	// Stream enables streaming responses
	Stream *bool `json:"stream,omitempty"`

	// KeepAlive controls how long the model stays loaded
	KeepAlive *Duration `json:"keep_alive,omitempty"`

	// ============================================
	// Whisper-specific parameters
	// ============================================

	// SamplingStrategy: "greedy" or "beam_search" (default: greedy)
	SamplingStrategy string `json:"sampling_strategy,omitempty"`

	// NumThreads for processing (0 = auto)
	NumThreads int `json:"num_threads,omitempty"`

	// MaxTextContext: max tokens to use from past text as prompt (default: model default)
	MaxTextContext int `json:"max_text_ctx,omitempty"`

	// OffsetMs: start offset in milliseconds
	OffsetMs int `json:"offset_ms,omitempty"`

	// DurationMs: audio duration to process in ms (0 = full audio)
	DurationMs int `json:"duration_ms,omitempty"`

	// NoContext: do not use past transcription as initial prompt
	NoContext bool `json:"no_context,omitempty"`

	// NoTimestamps: do not generate timestamps
	NoTimestamps bool `json:"no_timestamps,omitempty"`

	// SingleSegment: force single segment output (useful for streaming)
	SingleSegment bool `json:"single_segment,omitempty"`

	// TokenTimestamps: enable token-level timestamps
	TokenTimestamps bool `json:"token_timestamps,omitempty"`

	// MaxSegmentLength: max segment length in characters (0 = no limit)
	MaxSegmentLength int `json:"max_segment_length,omitempty"`

	// SplitOnWord: split on word rather than on token (when used with MaxSegmentLength)
	SplitOnWord bool `json:"split_on_word,omitempty"`

	// MaxTokensPerSegment: max tokens per segment (0 = no limit)
	MaxTokensPerSegment int `json:"max_tokens_per_segment,omitempty"`

	// AudioContext: overwrite the audio context size (0 = use default)
	AudioContext int `json:"audio_ctx,omitempty"`

	// SpeakerDiarization: enable tinydiarize speaker turn detection
	SpeakerDiarization bool `json:"speaker_diarization,omitempty"`

	// SuppressRegex: regex pattern to suppress matching tokens
	SuppressRegex string `json:"suppress_regex,omitempty"`

	// SuppressBlank: suppress blank tokens at the beginning
	SuppressBlank *bool `json:"suppress_blank,omitempty"`

	// SuppressNonSpeech: suppress non-speech tokens
	SuppressNonSpeech *bool `json:"suppress_non_speech,omitempty"`

	// ============================================
	// Temperature and sampling parameters
	// ============================================

	// Temperature for sampling (0.0 to 1.0, default: 0.0)
	Temperature float32 `json:"temperature,omitempty"`

	// TemperatureIncrement: increment for temperature fallback
	TemperatureIncrement float32 `json:"temperature_increment,omitempty"`

	// MaxInitialTimestamp: max initial timestamp (default: 1.0)
	MaxInitialTimestamp float32 `json:"max_initial_ts,omitempty"`

	// LengthPenalty: length penalty for beam search (default: -1 = disabled)
	LengthPenalty float32 `json:"length_penalty,omitempty"`

	// EntropyThreshold: entropy threshold for decoder fallback (default: 2.4)
	EntropyThreshold float32 `json:"entropy_threshold,omitempty"`

	// LogprobThreshold: log probability threshold for decoder fallback (default: -1.0)
	LogprobThreshold float32 `json:"logprob_threshold,omitempty"`

	// NoSpeechThreshold: no speech probability threshold (default: 0.6)
	NoSpeechThreshold float32 `json:"no_speech_threshold,omitempty"`

	// ============================================
	// Greedy sampling parameters
	// ============================================

	// BestOf: number of candidates for greedy sampling (default: 1)
	BestOf int `json:"best_of,omitempty"`

	// ============================================
	// Beam search parameters
	// ============================================

	// BeamSize: beam size for beam search (default: 5)
	BeamSize int `json:"beam_size,omitempty"`

	// Patience: beam search patience factor (default: -1 = disabled)
	Patience float32 `json:"patience,omitempty"`

	// ============================================
	// Word-level timestamp parameters
	// ============================================

	// WordTimestampThreshold: timestamp token probability threshold (default: 0.01)
	WordTimestampThreshold float32 `json:"word_timestamp_threshold,omitempty"`

	// WordTimestampSumThreshold: timestamp token sum probability threshold (default: 0.01)
	WordTimestampSumThreshold float32 `json:"word_timestamp_sum_threshold,omitempty"`

	// Options for additional model-specific settings
	Options map[string]any `json:"options,omitempty"`
}

// TranscribeResponse is the response from transcription
type TranscribeResponse struct {
	// Model used for transcription
	Model string `json:"model"`

	// Text is the full transcribed text
	Text string `json:"text"`

	// Language detected or specified
	Language string `json:"language"`

	// Duration of the audio in seconds
	Duration float64 `json:"duration"`

	// Segments contains detailed segment information (for verbose_json)
	Segments []TranscribeSegment `json:"segments,omitempty"`

	// Words contains word-level timestamps (if requested)
	Words []TranscribeWord `json:"words,omitempty"`

	// Task is either "transcribe" or "translate"
	Task string `json:"task,omitempty"`

	// Metrics
	TotalDuration      time.Duration `json:"total_duration,omitempty"`
	LoadDuration       time.Duration `json:"load_duration,omitempty"`
	ProcessingDuration time.Duration `json:"processing_duration,omitempty"`

	// Done indicates if this is the final response (for streaming)
	Done bool `json:"done,omitempty"`
}

// TranscribeSegment represents a transcribed segment with timestamps
type TranscribeSegment struct {
	// ID is the segment index
	ID int `json:"id"`

	// Seek is the seek offset
	Seek int `json:"seek"`

	// Start time in seconds
	Start float64 `json:"start"`

	// End time in seconds
	End float64 `json:"end"`

	// Text content of this segment
	Text string `json:"text"`

	// Tokens in this segment
	Tokens []int `json:"tokens,omitempty"`

	// Temperature used for this segment
	Temperature float64 `json:"temperature,omitempty"`

	// AvgLogprob is the average log probability
	AvgLogprob float64 `json:"avg_logprob,omitempty"`

	// CompressionRatio for this segment
	CompressionRatio float64 `json:"compression_ratio,omitempty"`

	// NoSpeechProb is probability of no speech
	NoSpeechProb float64 `json:"no_speech_prob,omitempty"`
}

// TranscribeWord represents word-level timestamp information
type TranscribeWord struct {
	// Word text
	Word string `json:"word"`

	// Start time in seconds
	Start float64 `json:"start"`

	// End time in seconds
	End float64 `json:"end"`

	// Probability of this word
	Probability float64 `json:"probability,omitempty"`
}

// TranscribeStreamResponse for streaming transcription
type TranscribeStreamResponse struct {
	// Segment being streamed
	Segment *TranscribeSegment `json:"segment,omitempty"`

	// PartialText is text in progress
	PartialText string `json:"partial_text,omitempty"`

	// Done indicates transcription is complete
	Done bool `json:"done"`

	// Error if any
	Error string `json:"error,omitempty"`
}

// TranslateRequest is similar to TranscribeRequest but specifically for translation
type TranslateRequest struct {
	TranscribeRequest
}

// VADRequest for voice activity detection
type VADRequest struct {
	// Model for VAD (optional, uses built-in if empty)
	Model string `json:"model,omitempty"`

	// Audio data
	Audio []byte `json:"audio"`

	// Threshold for speech detection (0.0 to 1.0)
	Threshold float32 `json:"threshold,omitempty"`

	// MinSpeechDurationMs minimum speech segment duration
	MinSpeechDurationMs int `json:"min_speech_duration_ms,omitempty"`

	// MinSilenceDurationMs minimum silence to split segments
	MinSilenceDurationMs int `json:"min_silence_duration_ms,omitempty"`
}

// VADResponse contains detected speech segments
type VADResponse struct {
	// Segments of detected speech
	Segments []VADSegment `json:"segments"`
}

// VADSegment represents a speech segment
type VADSegment struct {
	// Start time in seconds
	Start float64 `json:"start"`

	// End time in seconds
	End float64 `json:"end"`

	// Probability of speech
	Probability float64 `json:"probability"`
}

// WhisperModelDetails provides details about a whisper model
type WhisperModelDetails struct {
	// Type of model (tiny, base, small, medium, large, etc.)
	Type string `json:"type"`

	// Multilingual indicates if the model supports multiple languages
	Multilingual bool `json:"multilingual"`

	// Languages supported
	Languages []string `json:"languages,omitempty"`

	// ParameterSize human-readable size
	ParameterSize string `json:"parameter_size"`

	// QuantizationLevel if quantized
	QuantizationLevel string `json:"quantization_level,omitempty"`
}

// AudioInfo contains metadata about audio input
type AudioInfo struct {
	// Format of the audio
	Format string `json:"format"`

	// SampleRate in Hz
	SampleRate int `json:"sample_rate"`

	// Channels count
	Channels int `json:"channels"`

	// Duration in seconds
	Duration float64 `json:"duration"`

	// BitDepth of samples
	BitDepth int `json:"bit_depth,omitempty"`
}

// Helper function to check if transcription request wants streaming
func (r *TranscribeRequest) IsStreaming() bool {
	return r.Stream != nil && *r.Stream
}

// GetResponseFormat returns the response format with default
func (r *TranscribeRequest) GetResponseFormat() string {
	if r.ResponseFormat == "" {
		return "json"
	}
	return r.ResponseFormat
}

// SupportedAudioFormats returns list of supported audio formats
func SupportedAudioFormats() []string {
	return []string{
		"wav",
		"mp3",
		"flac",
		"ogg",
		"m4a",
		"webm",
	}
}

// SupportedWhisperModels returns available whisper model sizes
func SupportedWhisperModels() []string {
	return []string{
		"whisper:tiny",
		"whisper:tiny.en",
		"whisper:base",
		"whisper:base.en",
		"whisper:small",
		"whisper:small.en",
		"whisper:medium",
		"whisper:medium.en",
		"whisper:large",
		"whisper:large-v2",
		"whisper:large-v3",
		"whisper:large-v3-turbo",
	}
}

// SupportedLanguages returns ISO language codes supported by whisper
func SupportedLanguages() []string {
	return []string{
		"en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr",
		"pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
		"he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
		"th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk",
		"te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk",
		"br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
		"gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
		"ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
		"ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
		"mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su",
	}
}

