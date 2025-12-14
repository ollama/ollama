package stt

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/whisper"
)

// WhisperServer interface defines the contract for speech-to-text servers
type WhisperServer interface {
	ModelPath() string
	Load(ctx context.Context, gpus []ml.DeviceInfo) error
	Transcribe(ctx context.Context, req TranscribeRequest) (*TranscribeResponse, error)
	DetectLanguage(ctx context.Context, samples []float32) (string, float32, error)
	Close() error
	Ping(ctx context.Context) error
	GetModelInfo() whisper.ModelInfo
	IsMultilingual() bool
}

// TranscribeRequest - simplified with Options pattern for advanced parameters
type TranscribeRequest struct {
	// Required: Audio samples (PCM float32, 16kHz mono)
	Samples []float32

	// Basic parameters (commonly used)
	Language      string  // Source language (empty = auto-detect)
	Translate     bool    // Translate to English
	InitialPrompt string  // Context/prompt for the model
	Temperature   float32 // Sampling temperature (0.0 = greedy)
	NoTimestamps  bool    // Disable timestamps

	// Timestamp granularity
	TokenTimestamps bool // Enable word-level timestamps

	// Callbacks for streaming
	ProgressFunc   func(int)
	NewSegmentFunc func(whisper.Segment)

	// Advanced options (map for extensibility)
	Options map[string]any
}

// TranscribeResponse for internal use
type TranscribeResponse struct {
	Segments []whisper.Segment
	Language string
	Duration time.Duration
}

// Option keys for advanced parameters
const (
	OptSamplingStrategy       = "sampling_strategy"        // "greedy" or "beam_search"
	OptNumThreads             = "num_threads"              // int
	OptBeamSize               = "beam_size"                // int
	OptBestOf                 = "best_of"                  // int
	OptNoContext              = "no_context"               // bool
	OptSingleSegment          = "single_segment"           // bool
	OptMaxSegmentLength       = "max_segment_length"       // int
	OptMaxTokensPerSegment    = "max_tokens_per_segment"   // int
	OptSplitOnWord            = "split_on_word"            // bool
	OptSuppressBlank          = "suppress_blank"           // bool
	OptSuppressNonSpeech      = "suppress_non_speech"      // bool
	OptSpeakerDiarization     = "speaker_diarization"      // bool
	OptEntropyThreshold       = "entropy_threshold"        // float32
	OptLogprobThreshold       = "logprob_threshold"        // float32
	OptNoSpeechThreshold      = "no_speech_threshold"      // float32
	OptTemperatureIncrement   = "temperature_increment"    // float32
	OptMaxInitialTimestamp    = "max_initial_timestamp"    // float32
	OptWordTimestampThreshold = "word_timestamp_threshold" // float32
)

// GetOption returns an option value with type assertion
func (r *TranscribeRequest) GetOption(key string, defaultVal any) any {
	if r.Options == nil {
		return defaultVal
	}
	if v, ok := r.Options[key]; ok {
		return v
	}
	return defaultVal
}

// GetStringOption returns a string option
func (r *TranscribeRequest) GetStringOption(key, defaultVal string) string {
	v := r.GetOption(key, defaultVal)
	if s, ok := v.(string); ok {
		return s
	}
	return defaultVal
}

// GetIntOption returns an int option
func (r *TranscribeRequest) GetIntOption(key string, defaultVal int) int {
	v := r.GetOption(key, defaultVal)
	switch n := v.(type) {
	case int:
		return n
	case float64:
		return int(n)
	case float32:
		return int(n)
	}
	return defaultVal
}

// GetFloat32Option returns a float32 option
func (r *TranscribeRequest) GetFloat32Option(key string, defaultVal float32) float32 {
	v := r.GetOption(key, defaultVal)
	switch n := v.(type) {
	case float32:
		return n
	case float64:
		return float32(n)
	case int:
		return float32(n)
	}
	return defaultVal
}

// GetBoolOption returns a bool option
func (r *TranscribeRequest) GetBoolOption(key string, defaultVal bool) bool {
	v := r.GetOption(key, defaultVal)
	if b, ok := v.(bool); ok {
		return b
	}
	return defaultVal
}

// whisperServer is the implementation of WhisperServer
type whisperServer struct {
	modelPath string
	ctx       *whisper.WhisperContext
	params    whisper.ContextParams
	mu        sync.RWMutex
	loaded    bool
	loadStart time.Time
}

// NewWhisperServer creates a new whisper server instance and loads the model
func NewWhisperServer(modelPath string, params whisper.ContextParams) (WhisperServer, error) {
	if _, err := os.Stat(modelPath); err != nil {
		return nil, fmt.Errorf("model not found: %s", modelPath)
	}

	s := &whisperServer{
		modelPath: modelPath,
		params:    params,
		loadStart: time.Now(),
	}

	slog.Info("loading whisper model", "model", modelPath, "use_gpu", params.UseGPU)

	ctx, err := whisper.NewContext(modelPath, params)
	if err != nil {
		return nil, fmt.Errorf("failed to load whisper model: %w", err)
	}

	s.ctx = ctx
	s.loaded = true

	slog.Info("whisper model loaded", "multilingual", ctx.IsMultilingual())
	return s, nil
}

func (s *whisperServer) ModelPath() string { return s.modelPath }

func (s *whisperServer) Load(ctx context.Context, gpus []ml.DeviceInfo) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.loaded {
		return nil
	}

	if len(gpus) > 0 {
		s.params.UseGPU = true
		s.params.GPUDevice = 0
	} else {
		s.params.UseGPU = false
	}

	slog.Info("loading whisper model", "model", s.modelPath, "use_gpu", s.params.UseGPU)

	var err error
	s.ctx, err = whisper.NewContext(s.modelPath, s.params)
	if err != nil {
		return fmt.Errorf("failed to load whisper model: %w", err)
	}

	s.loaded = true
	slog.Info("whisper model loaded", "duration", time.Since(s.loadStart), "multilingual", s.ctx.IsMultilingual())
	return nil
}

// Transcribe performs speech-to-text
func (s *whisperServer) Transcribe(ctx context.Context, req TranscribeRequest) (*TranscribeResponse, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if !s.loaded || s.ctx == nil {
		return nil, errors.New("model not loaded")
	}

	params := whisper.DefaultTranscribeParams()

	// Basic parameters
	params.Language = req.Language
	params.Translate = req.Translate
	params.InitialPrompt = req.InitialPrompt
	params.NoTimestamps = req.NoTimestamps
	params.TokenTimestamps = req.TokenTimestamps

	if req.Temperature > 0 {
		params.Temperature = req.Temperature
	}

	// Advanced options from map
	strategy := req.GetStringOption(OptSamplingStrategy, "greedy")
	if strategy == "beam_search" {
		params.Strategy = whisper.SamplingBeamSearch
	}

	if threads := req.GetIntOption(OptNumThreads, 0); threads > 0 {
		params.NumThreads = threads
	}
	if beamSize := req.GetIntOption(OptBeamSize, 0); beamSize > 0 {
		params.BeamSize = beamSize
	}
	if bestOf := req.GetIntOption(OptBestOf, 0); bestOf > 0 {
		params.BestOf = bestOf
	}
	if maxLen := req.GetIntOption(OptMaxSegmentLength, 0); maxLen > 0 {
		params.MaxSegmentLength = maxLen
	}
	if maxTokens := req.GetIntOption(OptMaxTokensPerSegment, 0); maxTokens > 0 {
		params.MaxTokensPerSegment = maxTokens
	}

	params.NoContext = req.GetBoolOption(OptNoContext, false)
	params.SingleSegment = req.GetBoolOption(OptSingleSegment, false)
	params.SplitOnWord = req.GetBoolOption(OptSplitOnWord, false)
	params.SuppressBlank = req.GetBoolOption(OptSuppressBlank, true)
	params.SuppressNonSpeech = req.GetBoolOption(OptSuppressNonSpeech, true)
	params.SpeakerDiarization = req.GetBoolOption(OptSpeakerDiarization, false)

	if entropy := req.GetFloat32Option(OptEntropyThreshold, 0); entropy > 0 {
		params.EntropyThreshold = entropy
	}
	if logprob := req.GetFloat32Option(OptLogprobThreshold, 0); logprob != 0 {
		params.LogprobThreshold = logprob
	}
	if noSpeech := req.GetFloat32Option(OptNoSpeechThreshold, 0); noSpeech > 0 {
		params.NoSpeechThreshold = noSpeech
	}
	if tempInc := req.GetFloat32Option(OptTemperatureIncrement, 0); tempInc > 0 {
		params.TemperatureIncrement = tempInc
	}
	if maxTs := req.GetFloat32Option(OptMaxInitialTimestamp, 0); maxTs > 0 {
		params.MaxInitialTimestamp = maxTs
	}
	if wordThresh := req.GetFloat32Option(OptWordTimestampThreshold, 0); wordThresh > 0 {
		params.WordTimestampThreshold = wordThresh
	}

	// Callbacks
	params.ProgressFunc = req.ProgressFunc
	params.NewSegmentFunc = req.NewSegmentFunc

	start := time.Now()
	segments, err := s.ctx.Transcribe(req.Samples, params)
	if err != nil {
		return nil, err
	}

	detectedLang := s.ctx.DetectedLanguage()
	if req.Language != "" {
		detectedLang = req.Language
	}

	return &TranscribeResponse{
		Segments: segments,
		Language: detectedLang,
		Duration: time.Since(start),
	}, nil
}

func (s *whisperServer) DetectLanguage(ctx context.Context, samples []float32) (string, float32, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if !s.loaded || s.ctx == nil {
		return "", 0, errors.New("model not loaded")
	}
	return s.ctx.AutoDetectLanguage(samples, 0)
}

func (s *whisperServer) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.ctx != nil {
		s.ctx.Free()
		s.ctx = nil
	}
	s.loaded = false
	return nil
}

func (s *whisperServer) Ping(ctx context.Context) error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if !s.loaded {
		return errors.New("model not loaded")
	}
	return nil
}

func (s *whisperServer) GetModelInfo() whisper.ModelInfo {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.ctx == nil {
		return whisper.ModelInfo{}
	}
	return s.ctx.GetModelInfo()
}

func (s *whisperServer) IsMultilingual() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.ctx == nil {
		return false
	}
	return s.ctx.IsMultilingual()
}

// ============================================================================
// Audio Conversion
// ============================================================================

type AudioConverter struct {
	ffmpegPath string
}

func NewAudioConverter() (*AudioConverter, error) {
	ffmpegPath, err := exec.LookPath("ffmpeg")
	if err != nil {
		for _, p := range []string{"/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg", "C:\\ffmpeg\\bin\\ffmpeg.exe"} {
			if _, err := os.Stat(p); err == nil {
				ffmpegPath = p
				break
			}
		}
	}
	return &AudioConverter{ffmpegPath: ffmpegPath}, nil
}

func (c *AudioConverter) ToFloat32Samples(ctx context.Context, audio []byte, format string) ([]float32, error) {
	if format == "wav" || format == "" {
		if samples, err := parseWAV(audio); err == nil {
			return samples, nil
		}
	}
	if c.ffmpegPath == "" {
		return nil, errors.New("ffmpeg not found - required for audio format conversion")
	}
	return c.convertWithFFmpeg(ctx, audio)
}

func (c *AudioConverter) convertWithFFmpeg(ctx context.Context, audio []byte) ([]float32, error) {
	tmpIn, err := os.CreateTemp("", "whisper_in_*")
	if err != nil {
		return nil, err
	}
	defer os.Remove(tmpIn.Name())

	if _, err := tmpIn.Write(audio); err != nil {
		tmpIn.Close()
		return nil, err
	}
	tmpIn.Close()

	tmpOut, err := os.CreateTemp("", "whisper_out_*.raw")
	if err != nil {
		return nil, err
	}
	defer os.Remove(tmpOut.Name())
	tmpOut.Close()

	cmd := exec.CommandContext(ctx, c.ffmpegPath,
		"-i", tmpIn.Name(),
		"-ar", "16000", "-ac", "1", "-f", "f32le", "-acodec", "pcm_f32le", "-y",
		tmpOut.Name(),
	)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("ffmpeg failed: %v - %s", err, stderr.String())
	}

	rawData, err := os.ReadFile(tmpOut.Name())
	if err != nil {
		return nil, err
	}

	samples := make([]float32, len(rawData)/4)
	reader := bytes.NewReader(rawData)
	for i := range samples {
		if err := binary.Read(reader, binary.LittleEndian, &samples[i]); err != nil {
			break
		}
	}
	return samples, nil
}

func parseWAV(data []byte) ([]float32, error) {
	if len(data) < 44 || string(data[0:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return nil, errors.New("invalid WAV file")
	}

	pos := 12
	var numChannels, bitsPerSample uint16
	var sampleRate uint32

	for pos < len(data)-8 {
		chunkID := string(data[pos : pos+4])
		chunkSize := binary.LittleEndian.Uint32(data[pos+4 : pos+8])

		if chunkID == "fmt " && chunkSize >= 16 {
			numChannels = binary.LittleEndian.Uint16(data[pos+10 : pos+12])
			sampleRate = binary.LittleEndian.Uint32(data[pos+12 : pos+16])
			bitsPerSample = binary.LittleEndian.Uint16(data[pos+22 : pos+24])
		} else if chunkID == "data" {
			dataEnd := pos + 8 + int(chunkSize)
			if dataEnd > len(data) {
				dataEnd = len(data)
			}
			return convertToFloat32(data[pos+8:dataEnd], numChannels, bitsPerSample, sampleRate)
		}
		pos += 8 + int(chunkSize)
		if chunkSize%2 == 1 {
			pos++
		}
	}
	return nil, errors.New("no data chunk found")
}

func convertToFloat32(data []byte, channels, bitsPerSample uint16, sampleRate uint32) ([]float32, error) {
	bytesPerSample := int(bitsPerSample / 8)
	numSamples := len(data) / bytesPerSample / int(channels)
	samples := make([]float32, numSamples)
	reader := bytes.NewReader(data)

	for i := 0; i < numSamples; i++ {
		var channelSum float32
		for c := 0; c < int(channels); c++ {
			var sample float32
			switch bitsPerSample {
			case 8:
				var s uint8
				binary.Read(reader, binary.LittleEndian, &s)
				sample = (float32(s) - 128) / 128.0
			case 16:
				var s int16
				binary.Read(reader, binary.LittleEndian, &s)
				sample = float32(s) / 32768.0
			case 32:
				var s int32
				binary.Read(reader, binary.LittleEndian, &s)
				sample = float32(s) / 2147483648.0
			default:
				return nil, fmt.Errorf("unsupported bit depth: %d", bitsPerSample)
			}
			channelSum += sample
		}
		samples[i] = channelSum / float32(channels)
	}

	// Resample if not 16kHz
	if sampleRate != 16000 {
		ratio := float64(sampleRate) / 16000.0
		newLen := int(float64(len(samples)) / ratio)
		resampled := make([]float32, newLen)
		for i := 0; i < newLen; i++ {
			srcIdx := float64(i) * ratio
			idx := int(srcIdx)
			if idx+1 < len(samples) {
				frac := float32(srcIdx - float64(idx))
				resampled[i] = samples[idx]*(1-frac) + samples[idx+1]*frac
			} else if idx < len(samples) {
				resampled[i] = samples[idx]
			}
		}
		return resampled, nil
	}
	return samples, nil
}

func GetAudioDuration(samples []float32) time.Duration {
	return time.Duration(float64(len(samples)) / float64(whisper.SampleRate) * float64(time.Second))
}

// ============================================================================
// Model Size Utilities
// ============================================================================

type WhisperModelSize string

const (
	WhisperTiny       WhisperModelSize = "tiny"
	WhisperTinyEn     WhisperModelSize = "tiny.en"
	WhisperBase       WhisperModelSize = "base"
	WhisperBaseEn     WhisperModelSize = "base.en"
	WhisperSmall      WhisperModelSize = "small"
	WhisperSmallEn    WhisperModelSize = "small.en"
	WhisperMedium     WhisperModelSize = "medium"
	WhisperMediumEn   WhisperModelSize = "medium.en"
	WhisperLarge      WhisperModelSize = "large"
	WhisperLargeV2    WhisperModelSize = "large-v2"
	WhisperLargeV3    WhisperModelSize = "large-v3"
	WhisperLargeTurbo WhisperModelSize = "large-v3-turbo"
)

func ModelSizeFromName(name string) WhisperModelSize {
	name = strings.ToLower(name)
	for _, size := range []WhisperModelSize{
		WhisperLargeTurbo, WhisperLargeV3, WhisperLargeV2, WhisperLarge,
		WhisperMediumEn, WhisperMedium, WhisperSmallEn, WhisperSmall,
		WhisperBaseEn, WhisperBase, WhisperTinyEn, WhisperTiny,
	} {
		if strings.Contains(name, string(size)) {
			return size
		}
	}
	return WhisperBase
}

func EstimateVRAM(size WhisperModelSize) uint64 {
	switch size {
	case WhisperTiny, WhisperTinyEn:
		return 100 * format.MebiByte
	case WhisperBase, WhisperBaseEn:
		return 200 * format.MebiByte
	case WhisperSmall, WhisperSmallEn:
		return 500 * format.MebiByte
	case WhisperMedium, WhisperMediumEn:
		return 1500 * format.MebiByte
	case WhisperLarge, WhisperLargeV2, WhisperLargeV3:
		return 3000 * format.MebiByte
	case WhisperLargeTurbo:
		return 2000 * format.MebiByte
	default:
		return 200 * format.MebiByte
	}
}
