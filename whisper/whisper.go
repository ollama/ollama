package whisper

/*
#cgo CFLAGS: -std=c11
#cgo windows CFLAGS: -Wno-dll-attribute-on-redeclaration
#cgo CXXFLAGS: -std=c++17
#cgo CPPFLAGS: -I${SRCDIR}/whisper.cpp/include
#cgo CPPFLAGS: -I${SRCDIR}/whisper.cpp/src
#cgo CPPFLAGS: -I${SRCDIR}/whisper.cpp/ggml/include
#cgo CPPFLAGS: -I${SRCDIR}/../ml/backend/ggml/ggml/include

#cgo LDFLAGS: -L${SRCDIR}/whisper.cpp/build/src -L${SRCDIR}/whisper.cpp/build/ggml/src
#cgo LDFLAGS: -lwhisper -lggml -lggml-cpu -lggml-base
#cgo LDFLAGS: -lstdc++ -lm
#cgo windows LDFLAGS: -static -static-libgcc -static-libstdc++
#cgo linux LDFLAGS: -lgomp -lpthread
#cgo darwin LDFLAGS: -framework Accelerate

#include <stdlib.h>
#include <stdbool.h>
#include "whisper.h"

extern void whisperLogCallback(int level, char* text, void* user_data);
extern void whisperProgressCallback(void* user_data, int progress);
extern void whisperNewSegmentCallback(void* user_data, int n_new);
extern bool whisperEncoderBeginCallback(void* user_data);

// Progress callback wrapper
static void whisper_progress_cb_wrapper(struct whisper_context* ctx, struct whisper_state* state, int progress, void* user_data) {
    if(user_data != NULL && ctx != NULL) {
        whisperProgressCallback(user_data, progress);
    }
}

// New segment callback wrapper
static void whisper_new_segment_cb_wrapper(struct whisper_context* ctx, struct whisper_state* state, int n_new, void* user_data) {
    if(user_data != NULL && ctx != NULL) {
        whisperNewSegmentCallback(user_data, n_new);
    }
}

// Encoder begin callback wrapper
static bool whisper_encoder_begin_cb_wrapper(struct whisper_context* ctx, struct whisper_state* state, void* user_data) {
    if(user_data != NULL && ctx != NULL) {
        return whisperEncoderBeginCallback(user_data);
    }
    return true;
}

// Get default params with callbacks set
static struct whisper_full_params whisper_full_default_params_with_cb(enum whisper_sampling_strategy strategy) {
    struct whisper_full_params params = whisper_full_default_params(strategy);
    params.progress_callback = whisper_progress_cb_wrapper;
    params.new_segment_callback = whisper_new_segment_cb_wrapper;
    params.encoder_begin_callback = whisper_encoder_begin_cb_wrapper;
    return params;
}
*/
import "C"

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"runtime"
	"sync"
	"time"
	"unsafe"
)

// Constants from whisper.h
const (
	SampleRate = C.WHISPER_SAMPLE_RATE // 16000 Hz
	NumFFT     = C.WHISPER_N_FFT       // 400
	HopLength  = C.WHISPER_HOP_LENGTH  // 160
	ChunkSize  = C.WHISPER_CHUNK_SIZE  // 30 seconds
)

// SamplingStrategy represents the decoding strategy
type SamplingStrategy int

const (
	SamplingGreedy     SamplingStrategy = C.WHISPER_SAMPLING_GREEDY
	SamplingBeamSearch SamplingStrategy = C.WHISPER_SAMPLING_BEAM_SEARCH
)

// Errors
var (
	ErrModelLoad       = errors.New("failed to load whisper model")
	ErrTranscription   = errors.New("transcription failed")
	ErrInvalidLanguage = errors.New("invalid language")
	ErrContextCreation = errors.New("failed to create whisper context")
)

// WhisperContext wraps the whisper context
type WhisperContext struct {
	c  *C.struct_whisper_context
	mu sync.Mutex
}

// Segment represents a transcribed segment with timestamps
type Segment struct {
	Text   string        `json:"text"`
	Start  time.Duration `json:"start"`
	End    time.Duration `json:"end"`
	Tokens []Token       `json:"tokens,omitempty"`
}

// Token represents a single token with probability info
type Token struct {
	ID          int     `json:"id"`
	Text        string  `json:"text"`
	Probability float32 `json:"probability"`
	Start       int64   `json:"start"`
	End         int64   `json:"end"`
}

// TranscribeParams holds all parameters for transcription
type TranscribeParams struct {
	// Basic parameters
	Language      string           // Source language (auto-detect if empty)
	Translate     bool             // Translate to English
	Strategy      SamplingStrategy // Sampling strategy
	NumThreads    int              // Number of threads (0 = auto)

	// Context and timing
	MaxTextContext int // Max tokens from past text as prompt (-1 = default)
	OffsetMs       int // Start offset in milliseconds
	DurationMs     int // Audio duration to process (0 = full)
	AudioContext   int // Audio context size override (0 = default)

	// Output control
	NoContext     bool // Do not use past transcription as prompt
	NoTimestamps  bool // Disable timestamps
	SingleSegment bool // Force single segment output
	PrintProgress bool // Print progress information
	PrintSpecial  bool // Print special tokens

	// Token-level timestamps
	TokenTimestamps           bool    // Enable token-level timestamps
	WordTimestampThreshold    float32 // Timestamp token probability threshold (~0.01)
	WordTimestampSumThreshold float32 // Timestamp token sum probability threshold (~0.01)

	// Segment control
	MaxSegmentLength    int  // Max segment length in characters (0 = no limit)
	SplitOnWord         bool // Split on word rather than token
	MaxTokensPerSegment int  // Max tokens per segment (0 = no limit)

	// Special features
	SpeakerDiarization bool   // Enable tinydiarize speaker detection
	SuppressRegex      string // Regex to suppress tokens
	InitialPrompt      string // Initial prompt text

	// Token suppression
	SuppressBlank     bool // Suppress blank tokens
	SuppressNonSpeech bool // Suppress non-speech tokens

	// Temperature and sampling
	Temperature          float32 // Sampling temperature (default: 0.0)
	TemperatureIncrement float32 // Temperature increment for fallback
	MaxInitialTimestamp  float32 // Max initial timestamp (default: 1.0)
	LengthPenalty        float32 // Length penalty (-1 = disabled)

	// Fallback thresholds
	EntropyThreshold  float32 // Entropy threshold for fallback (default: 2.4)
	LogprobThreshold  float32 // Log probability threshold (default: -1.0)
	NoSpeechThreshold float32 // No speech probability threshold (default: 0.6)

	// Greedy parameters
	BestOf int // Number of candidates (default: 1)

	// Beam search parameters
	BeamSize int     // Beam size (default: 5)
	Patience float32 // Patience factor (-1 = disabled)

	// Callbacks
	ProgressFunc   func(int)     // Progress callback
	NewSegmentFunc func(Segment) // New segment callback
	EncoderBeginFn func() bool   // Encoder begin callback (return false to abort)
}

// DefaultTranscribeParams returns default transcription parameters
func DefaultTranscribeParams() TranscribeParams {
	return TranscribeParams{
		Language:              "",    // Auto-detect
		Translate:             false,
		Strategy:              SamplingGreedy,
		NumThreads:            0,    // Auto
		MaxTextContext:        -1,   // Use default
		NoTimestamps:          false,
		SingleSegment:         false,
		TokenTimestamps:       false,
		MaxSegmentLength:      0,    // No limit
		MaxTokensPerSegment:   0,    // No limit
		SuppressBlank:         true,
		SuppressNonSpeech:     true,
		Temperature:           0.0,
		TemperatureIncrement:  0.2,
		MaxInitialTimestamp:   1.0,
		LengthPenalty:         -1.0, // Disabled
		EntropyThreshold:      2.4,
		LogprobThreshold:      -1.0,
		NoSpeechThreshold:     0.6,
		BestOf:                1,
		BeamSize:              5,
		Patience:              -1.0, // Disabled
		WordTimestampThreshold:    0.01,
		WordTimestampSumThreshold: 0.01,
	}
}

// ContextParams for model loading
type ContextParams struct {
	UseGPU      bool
	FlashAttn   bool
	GPUDevice   int
	DTWTokenTS  bool // Token-level timestamps with DTW
}

// DefaultContextParams returns default context parameters
func DefaultContextParams() ContextParams {
	return ContextParams{
		UseGPU:    true,
		FlashAttn: false,
		GPUDevice: 0,
	}
}

func init() {
	// Set up logging callback
	// C.whisper_log_set(C.ggml_log_callback(C.whisperLogCallback), nil)
}

//export whisperLogCallback
func whisperLogCallback(level C.int, text *C.char, _ unsafe.Pointer) {
	if slog.Default().Enabled(context.TODO(), slog.Level(int(level)*4)) {
		fmt.Fprint(os.Stderr, C.GoString(text))
	}
}

// Callback management
var (
	callbackMu       sync.RWMutex
	progressCallbacks = make(map[uintptr]func(int))
	segmentCallbacks  = make(map[uintptr]func(Segment))
	encoderCallbacks  = make(map[uintptr]func() bool)
)

func registerCallbacks(ctx *WhisperContext, params TranscribeParams) uintptr {
	ptr := uintptr(unsafe.Pointer(ctx.c))
	callbackMu.Lock()
	defer callbackMu.Unlock()
	
	if params.ProgressFunc != nil {
		progressCallbacks[ptr] = params.ProgressFunc
	}
	if params.NewSegmentFunc != nil {
		segmentCallbacks[ptr] = params.NewSegmentFunc
	}
	if params.EncoderBeginFn != nil {
		encoderCallbacks[ptr] = params.EncoderBeginFn
	}
	return ptr
}

func unregisterCallbacks(ptr uintptr) {
	callbackMu.Lock()
	defer callbackMu.Unlock()
	delete(progressCallbacks, ptr)
	delete(segmentCallbacks, ptr)
	delete(encoderCallbacks, ptr)
}

//export whisperProgressCallback
func whisperProgressCallback(userData unsafe.Pointer, progress C.int) {
	ptr := uintptr(userData)
	callbackMu.RLock()
	fn, ok := progressCallbacks[ptr]
	callbackMu.RUnlock()
	if ok && fn != nil {
		fn(int(progress))
	}
}

//export whisperNewSegmentCallback
func whisperNewSegmentCallback(userData unsafe.Pointer, nNew C.int) {
	// This callback is called when new segments are available
	// We would need the context to get segment details, so this is a simplified version
	ptr := uintptr(userData)
	callbackMu.RLock()
	_, ok := segmentCallbacks[ptr]
	callbackMu.RUnlock()
	if ok {
		// Segment details would be fetched separately
	}
}

//export whisperEncoderBeginCallback
func whisperEncoderBeginCallback(userData unsafe.Pointer) C.bool {
	ptr := uintptr(userData)
	callbackMu.RLock()
	fn, ok := encoderCallbacks[ptr]
	callbackMu.RUnlock()
	if ok && fn != nil {
		return C.bool(fn())
	}
	return C.bool(true)
}

// NewContext creates a new whisper context from a model file
func NewContext(modelPath string, params ContextParams) (*WhisperContext, error) {
	cPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cPath))

	cParams := C.whisper_context_default_params()
	cParams.use_gpu = C.bool(params.UseGPU)
	cParams.flash_attn = C.bool(params.FlashAttn)
	cParams.gpu_device = C.int(params.GPUDevice)
	cParams.dtw_token_timestamps = C.bool(params.DTWTokenTS)

	ctx := C.whisper_init_from_file_with_params(cPath, cParams)
	if ctx == nil {
		return nil, fmt.Errorf("%w: %s", ErrModelLoad, modelPath)
	}

	return &WhisperContext{c: ctx}, nil
}

// NewContextFromBuffer creates a context from model data in memory
func NewContextFromBuffer(data []byte, params ContextParams) (*WhisperContext, error) {
	if len(data) == 0 {
		return nil, errors.New("empty model data")
	}

	cParams := C.whisper_context_default_params()
	cParams.use_gpu = C.bool(params.UseGPU)
	cParams.flash_attn = C.bool(params.FlashAttn)
	cParams.gpu_device = C.int(params.GPUDevice)

	ctx := C.whisper_init_from_buffer_with_params(
		unsafe.Pointer(&data[0]),
		C.size_t(len(data)),
		cParams,
	)
	if ctx == nil {
		return nil, ErrModelLoad
	}

	return &WhisperContext{c: ctx}, nil
}

// Free releases resources associated with the context
func (ctx *WhisperContext) Free() {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()

	if ctx.c != nil {
		C.whisper_free(ctx.c)
		ctx.c = nil
	}
}

// IsMultilingual returns true if the model supports multiple languages
func (ctx *WhisperContext) IsMultilingual() bool {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	return int(C.whisper_is_multilingual(ctx.c)) != 0
}

// Languages returns all supported language codes
func Languages() []string {
	maxID := int(C.whisper_lang_max_id())
	langs := make([]string, 0, maxID+1)
	for i := 0; i <= maxID; i++ {
		lang := C.GoString(C.whisper_lang_str(C.int(i)))
		if lang != "" {
			langs = append(langs, lang)
		}
	}
	return langs
}

// LanguageID returns the ID for a language code
func LanguageID(lang string) int {
	cLang := C.CString(lang)
	defer C.free(unsafe.Pointer(cLang))
	return int(C.whisper_lang_id(cLang))
}

// Transcribe transcribes audio samples (PCM float32, 16kHz mono)
func (ctx *WhisperContext) Transcribe(samples []float32, params TranscribeParams) ([]Segment, error) {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()

	if len(samples) == 0 {
		return nil, errors.New("no audio samples provided")
	}

	// Get default params based on strategy
	cParams := C.whisper_full_default_params_with_cb(C.enum_whisper_sampling_strategy(params.Strategy))

	// ============================================
	// Basic parameters
	// ============================================

	// Set language
	if params.Language != "" {
		cLang := C.CString(params.Language)
		defer C.free(unsafe.Pointer(cLang))
		cParams.language = cLang
	}

	cParams.translate = C.bool(params.Translate)

	// Set thread count
	numThreads := params.NumThreads
	if numThreads <= 0 {
		numThreads = runtime.NumCPU()
	}
	cParams.n_threads = C.int(numThreads)

	// ============================================
	// Context and timing
	// ============================================

	if params.MaxTextContext >= 0 {
		cParams.n_max_text_ctx = C.int(params.MaxTextContext)
	}

	if params.OffsetMs > 0 {
		cParams.offset_ms = C.int(params.OffsetMs)
	}

	if params.DurationMs > 0 {
		cParams.duration_ms = C.int(params.DurationMs)
	}

	if params.AudioContext > 0 {
		cParams.audio_ctx = C.int(params.AudioContext)
	}

	// ============================================
	// Output control
	// ============================================

	cParams.no_context = C.bool(params.NoContext)
	cParams.no_timestamps = C.bool(params.NoTimestamps)
	cParams.single_segment = C.bool(params.SingleSegment)
	cParams.print_progress = C.bool(params.PrintProgress)
	cParams.print_special = C.bool(params.PrintSpecial)

	// ============================================
	// Token-level timestamps
	// ============================================

	cParams.token_timestamps = C.bool(params.TokenTimestamps)

	if params.WordTimestampThreshold > 0 {
		cParams.thold_pt = C.float(params.WordTimestampThreshold)
	}

	if params.WordTimestampSumThreshold > 0 {
		cParams.thold_ptsum = C.float(params.WordTimestampSumThreshold)
	}

	// ============================================
	// Segment control
	// ============================================

	if params.MaxSegmentLength > 0 {
		cParams.max_len = C.int(params.MaxSegmentLength)
	}

	cParams.split_on_word = C.bool(params.SplitOnWord)

	if params.MaxTokensPerSegment > 0 {
		cParams.max_tokens = C.int(params.MaxTokensPerSegment)
	}

	// ============================================
	// Special features
	// ============================================

	cParams.tdrz_enable = C.bool(params.SpeakerDiarization)

	if params.SuppressRegex != "" {
		cRegex := C.CString(params.SuppressRegex)
		defer C.free(unsafe.Pointer(cRegex))
		cParams.suppress_regex = cRegex
	}

	if params.InitialPrompt != "" {
		cPrompt := C.CString(params.InitialPrompt)
		defer C.free(unsafe.Pointer(cPrompt))
		cParams.initial_prompt = cPrompt
	}

	// ============================================
	// Token suppression
	// ============================================

	cParams.suppress_blank = C.bool(params.SuppressBlank)
	cParams.suppress_nst = C.bool(params.SuppressNonSpeech)

	// ============================================
	// Temperature and sampling
	// ============================================

	cParams.temperature = C.float(params.Temperature)

	if params.TemperatureIncrement > 0 {
		cParams.temperature_inc = C.float(params.TemperatureIncrement)
	}

	if params.MaxInitialTimestamp > 0 {
		cParams.max_initial_ts = C.float(params.MaxInitialTimestamp)
	}

	if params.LengthPenalty != 0 {
		cParams.length_penalty = C.float(params.LengthPenalty)
	}

	// ============================================
	// Fallback thresholds
	// ============================================

	if params.EntropyThreshold > 0 {
		cParams.entropy_thold = C.float(params.EntropyThreshold)
	}

	if params.LogprobThreshold != 0 {
		cParams.logprob_thold = C.float(params.LogprobThreshold)
	}

	if params.NoSpeechThreshold > 0 {
		cParams.no_speech_thold = C.float(params.NoSpeechThreshold)
	}

	// ============================================
	// Greedy parameters
	// ============================================

	if params.BestOf > 0 {
		cParams.greedy.best_of = C.int(params.BestOf)
	}

	// ============================================
	// Beam search parameters
	// ============================================

	if params.BeamSize > 0 {
		cParams.beam_search.beam_size = C.int(params.BeamSize)
	}

	if params.Patience > 0 {
		cParams.beam_search.patience = C.float(params.Patience)
	}

	// ============================================
	// Set up callbacks
	// ============================================

	ptr := registerCallbacks(ctx, params)
	defer unregisterCallbacks(ptr)

	cParams.progress_callback_user_data = unsafe.Pointer(ptr)
	cParams.new_segment_callback_user_data = unsafe.Pointer(ptr)
	cParams.encoder_begin_callback_user_data = unsafe.Pointer(ptr)

	// ============================================
	// Run transcription
	// ============================================

	result := C.whisper_full(
		ctx.c,
		cParams,
		(*C.float)(unsafe.Pointer(&samples[0])),
		C.int(len(samples)),
	)

	if result != 0 {
		return nil, fmt.Errorf("%w: whisper_full returned %d", ErrTranscription, result)
	}

	// Extract segments
	return ctx.getSegments(), nil
}

// getSegments extracts all segments from the context
func (ctx *WhisperContext) getSegments() []Segment {
	nSegments := int(C.whisper_full_n_segments(ctx.c))
	segments := make([]Segment, nSegments)

	for i := 0; i < nSegments; i++ {
		t0 := int64(C.whisper_full_get_segment_t0(ctx.c, C.int(i)))
		t1 := int64(C.whisper_full_get_segment_t1(ctx.c, C.int(i)))
		text := C.GoString(C.whisper_full_get_segment_text(ctx.c, C.int(i)))

		// Convert timestamps (in centiseconds) to duration
		segments[i] = Segment{
			Text:  text,
			Start: time.Duration(t0*10) * time.Millisecond,
			End:   time.Duration(t1*10) * time.Millisecond,
		}

		// Get tokens for this segment
		nTokens := int(C.whisper_full_n_tokens(ctx.c, C.int(i)))
		segments[i].Tokens = make([]Token, nTokens)
		for j := 0; j < nTokens; j++ {
			tokenData := C.whisper_full_get_token_data(ctx.c, C.int(i), C.int(j))
			segments[i].Tokens[j] = Token{
				ID:          int(tokenData.id),
				Text:        C.GoString(C.whisper_full_get_token_text(ctx.c, C.int(i), C.int(j))),
				Probability: float32(tokenData.p),
				Start:       int64(tokenData.t0),
				End:         int64(tokenData.t1),
			}
		}
	}

	return segments
}

// DetectedLanguage returns the detected language after transcription
func (ctx *WhisperContext) DetectedLanguage() string {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	langID := int(C.whisper_full_lang_id(ctx.c))
	return C.GoString(C.whisper_lang_str(C.int(langID)))
}

// AutoDetectLanguage detects the language from audio
func (ctx *WhisperContext) AutoDetectLanguage(samples []float32, numThreads int) (string, float32, error) {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()

	if numThreads <= 0 {
		numThreads = runtime.NumCPU()
	}

	// Convert to mel spectrogram first
	result := C.whisper_pcm_to_mel(
		ctx.c,
		(*C.float)(unsafe.Pointer(&samples[0])),
		C.int(len(samples)),
		C.int(numThreads),
	)
	if result != 0 {
		return "", 0, errors.New("failed to compute mel spectrogram")
	}

	// Get language probabilities
	maxLangs := int(C.whisper_lang_max_id()) + 1
	probs := make([]float32, maxLangs)

	result = C.whisper_lang_auto_detect(
		ctx.c,
		C.int(0), // offset_ms
		C.int(numThreads),
		(*C.float)(unsafe.Pointer(&probs[0])),
	)
	if result < 0 {
		return "", 0, errors.New("language auto-detection failed")
	}

	// Find best language
	bestLang := 0
	bestProb := probs[0]
	for i := 1; i < maxLangs; i++ {
		if probs[i] > bestProb {
			bestProb = probs[i]
			bestLang = i
		}
	}

	langStr := C.GoString(C.whisper_lang_str(C.int(bestLang)))
	return langStr, bestProb, nil
}

// PrintTimings prints performance timings to stderr
func (ctx *WhisperContext) PrintTimings() {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	C.whisper_print_timings(ctx.c)
}

// ResetTimings resets the timing statistics
func (ctx *WhisperContext) ResetTimings() {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	C.whisper_reset_timings(ctx.c)
}

// SystemInfo returns system information string
func SystemInfo() string {
	return C.GoString(C.whisper_print_system_info())
}

// ModelInfo contains information about the loaded model
type ModelInfo struct {
	Type          string
	Multilingual  bool
	VocabSize     int
	AudioCtxSize  int
	TextCtxSize   int
	MelFilterSize int
}

// GetModelInfo returns information about the loaded model
func (ctx *WhisperContext) GetModelInfo() ModelInfo {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()

	return ModelInfo{
		Multilingual:  int(C.whisper_is_multilingual(ctx.c)) != 0,
		VocabSize:     int(C.whisper_n_vocab(ctx.c)),
		AudioCtxSize:  int(C.whisper_n_audio_ctx(ctx.c)),
		TextCtxSize:   int(C.whisper_n_text_ctx(ctx.c)),
		MelFilterSize: int(C.whisper_n_len(ctx.c)),
	}
}

// TokenToString converts a token ID to its string representation
func (ctx *WhisperContext) TokenToString(token int) string {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	return C.GoString(C.whisper_token_to_str(ctx.c, C.whisper_token(token)))
}

// SpecialTokens contains special token IDs
type SpecialTokens struct {
	EOT        int // End of transcript
	SOT        int // Start of transcript
	Prev       int // Previous token
	SOLM       int // Start of language model
	NOT        int // Not token
	BEG        int // Beginning
	Translate  int
	Transcribe int
}

// GetSpecialTokens returns the special token IDs for this model
func (ctx *WhisperContext) GetSpecialTokens() SpecialTokens {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()

	return SpecialTokens{
		EOT:        int(C.whisper_token_eot(ctx.c)),
		SOT:        int(C.whisper_token_sot(ctx.c)),
		Prev:       int(C.whisper_token_prev(ctx.c)),
		SOLM:       int(C.whisper_token_solm(ctx.c)),
		NOT:        int(C.whisper_token_not(ctx.c)),
		BEG:        int(C.whisper_token_beg(ctx.c)),
		Translate:  int(C.whisper_token_translate(ctx.c)),
		Transcribe: int(C.whisper_token_transcribe(ctx.c)),
	}
}

// Ensure context is freed when garbage collected
var finalizer = func(ctx *WhisperContext) {
	ctx.Free()
}

func init() {
	// Set finalizer for automatic cleanup
	// runtime.SetFinalizer is called in NewContext
}

