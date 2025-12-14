package stt

import (
	"bytes"
	"encoding/binary"
	"testing"
	"time"

	"github.com/ollama/ollama/whisper"
)

// TestGetAudioDuration tests duration calculation
func TestGetAudioDuration(t *testing.T) {
	// 1 second of audio at 16kHz
	samples := make([]float32, 16000)
	duration := GetAudioDuration(samples)
	
	if duration != time.Second {
		t.Errorf("GetAudioDuration() = %v, want 1s", duration)
	}
	
	// 2.5 seconds
	samples = make([]float32, 40000)
	duration = GetAudioDuration(samples)
	expected := 2500 * time.Millisecond
	
	if duration != expected {
		t.Errorf("GetAudioDuration() = %v, want %v", duration, expected)
	}
}

// TestModelSizeFromName tests model size detection
func TestModelSizeFromName(t *testing.T) {
	tests := []struct {
		name     string
		expected WhisperModelSize
	}{
		{"whisper:tiny", WhisperTiny},
		{"whisper:tiny.en", WhisperTinyEn},
		{"whisper:base", WhisperBase},
		{"whisper:base.en", WhisperBaseEn},
		{"whisper:small", WhisperSmall},
		{"whisper:medium", WhisperMedium},
		{"whisper:large-v3", WhisperLargeV3},
		{"whisper:large-v3-turbo", WhisperLargeTurbo},
		{"unknown-model", WhisperBase}, // Default
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ModelSizeFromName(tt.name)
			if result != tt.expected {
				t.Errorf("ModelSizeFromName(%q) = %v, want %v", tt.name, result, tt.expected)
			}
		})
	}
}

// TestEstimateVRAM tests VRAM estimation
func TestEstimateVRAM(t *testing.T) {
	// Tiny should use less VRAM than large
	tinyVRAM := EstimateVRAM(WhisperTiny)
	largeVRAM := EstimateVRAM(WhisperLargeV3)
	
	if tinyVRAM >= largeVRAM {
		t.Errorf("Tiny VRAM (%d) should be less than Large VRAM (%d)", tinyVRAM, largeVRAM)
	}
	
	// All estimates should be positive
	for _, size := range []WhisperModelSize{WhisperTiny, WhisperBase, WhisperSmall, WhisperMedium, WhisperLargeV3} {
		vram := EstimateVRAM(size)
		if vram == 0 {
			t.Errorf("EstimateVRAM(%v) = 0, want positive", size)
		}
	}
}

// TestNewAudioConverter tests converter creation
func TestNewAudioConverter(t *testing.T) {
	converter, err := NewAudioConverter()
	if err != nil {
		t.Fatalf("NewAudioConverter() error = %v", err)
	}
	
	if converter == nil {
		t.Fatal("NewAudioConverter() returned nil")
	}
}

// TestParseWAV tests WAV parsing
func TestParseWAV(t *testing.T) {
	// Create a minimal valid WAV file
	wav := createTestWAV(16000, 16, 1, 1.0) // 1 second, 16-bit, mono
	
	samples, err := parseWAV(wav)
	if err != nil {
		t.Fatalf("parseWAV() error = %v", err)
	}
	
	// Should have 16000 samples for 1 second at 16kHz
	if len(samples) != 16000 {
		t.Errorf("parseWAV() returned %d samples, want 16000", len(samples))
	}
	
	// All samples should be in valid range
	for i, s := range samples {
		if s < -1.0 || s > 1.0 {
			t.Errorf("Sample %d = %f, want between -1 and 1", i, s)
			break
		}
	}
}

// TestParseWAV_InvalidInput tests error handling
func TestParseWAV_InvalidInput(t *testing.T) {
	tests := []struct {
		name string
		data []byte
	}{
		{"too small", []byte{0x00, 0x01, 0x02}},
		{"not RIFF", []byte("NOT_RIFF0000WAVE" + string(make([]byte, 32)))},
		{"not WAVE", []byte("RIFF0000NOTW" + string(make([]byte, 32)))},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := parseWAV(tt.data)
			if err == nil {
				t.Error("parseWAV() expected error, got nil")
			}
		})
	}
}

// TestConvertToFloat32 tests sample conversion
func TestConvertToFloat32(t *testing.T) {
	// Test 16-bit conversion
	data := make([]byte, 4) // 2 samples
	binary.LittleEndian.PutUint16(data[0:2], 0x7FFF) // Max positive
	binary.LittleEndian.PutUint16(data[2:4], 0x8001) // Max negative
	
	samples, err := convertToFloat32(data, 1, 16, 16000)
	if err != nil {
		t.Fatalf("convertToFloat32() error = %v", err)
	}
	
	if len(samples) != 2 {
		t.Errorf("convertToFloat32() returned %d samples, want 2", len(samples))
	}
	
	// First sample should be close to 1.0
	if samples[0] < 0.99 {
		t.Errorf("samples[0] = %f, want close to 1.0", samples[0])
	}
	
	// Second sample should be close to -1.0
	if samples[1] > -0.99 {
		t.Errorf("samples[1] = %f, want close to -1.0", samples[1])
	}
}

// TestTranscribeRequest tests request struct
func TestTranscribeRequest(t *testing.T) {
	req := TranscribeRequest{
		Samples:      make([]float32, 16000),
		Language:     "en",
		Translate:    false,
		NoTimestamps: false,
		Temperature:  0.0,
	}

	if len(req.Samples) != 16000 {
		t.Errorf("Samples length = %d, want 16000", len(req.Samples))
	}

	if req.Language != "en" {
		t.Errorf("Language = %q, want \"en\"", req.Language)
	}
}

// TestTranscribeRequestOptions tests the Options pattern
func TestTranscribeRequestOptions(t *testing.T) {
	req := TranscribeRequest{
		Samples:  make([]float32, 16000),
		Language: "en",
		Options: map[string]any{
			OptSamplingStrategy:  "beam_search",
			OptBeamSize:          5,
			OptNumThreads:        4,
			OptSuppressBlank:     true,
			OptEntropyThreshold:  2.4,
			OptNoContext:         false,
		},
	}

	// Test string option
	strategy := req.GetStringOption(OptSamplingStrategy, "greedy")
	if strategy != "beam_search" {
		t.Errorf("GetStringOption(SamplingStrategy) = %q, want \"beam_search\"", strategy)
	}

	// Test int option
	beamSize := req.GetIntOption(OptBeamSize, 1)
	if beamSize != 5 {
		t.Errorf("GetIntOption(BeamSize) = %d, want 5", beamSize)
	}

	// Test float option
	entropy := req.GetFloat32Option(OptEntropyThreshold, 0.0)
	if entropy != 2.4 {
		t.Errorf("GetFloat32Option(EntropyThreshold) = %f, want 2.4", entropy)
	}

	// Test bool option
	suppressBlank := req.GetBoolOption(OptSuppressBlank, false)
	if !suppressBlank {
		t.Error("GetBoolOption(SuppressBlank) = false, want true")
	}

	// Test default value for missing option
	patience := req.GetFloat32Option("patience", 1.5)
	if patience != 1.5 {
		t.Errorf("GetFloat32Option(patience) = %f, want 1.5 (default)", patience)
	}

	// Test nil Options map
	emptyReq := TranscribeRequest{}
	defaultVal := emptyReq.GetStringOption(OptSamplingStrategy, "greedy")
	if defaultVal != "greedy" {
		t.Errorf("GetStringOption with nil Options = %q, want \"greedy\"", defaultVal)
	}
}

// TestGetIntOptionWithFloatValue tests int conversion from JSON float
func TestGetIntOptionWithFloatValue(t *testing.T) {
	req := TranscribeRequest{
		Options: map[string]any{
			OptBeamSize: float64(10), // JSON numbers are float64
		},
	}

	beamSize := req.GetIntOption(OptBeamSize, 1)
	if beamSize != 10 {
		t.Errorf("GetIntOption with float64 value = %d, want 10", beamSize)
	}
}

// TestTranscribeResponse tests response struct
func TestTranscribeResponse(t *testing.T) {
	resp := TranscribeResponse{
		Segments: []whisper.Segment{
			{Text: "Hello", Start: 0, End: time.Second},
			{Text: "World", Start: time.Second, End: 2 * time.Second},
		},
		Language: "en",
		Duration: 100 * time.Millisecond,
	}
	
	if len(resp.Segments) != 2 {
		t.Errorf("Segments length = %d, want 2", len(resp.Segments))
	}
	
	if resp.Language != "en" {
		t.Errorf("Language = %q, want \"en\"", resp.Language)
	}
}

// Helper function to create a test WAV file
func createTestWAV(sampleRate, bitsPerSample, channels int, durationSec float64) []byte {
	numSamples := int(float64(sampleRate) * durationSec)
	bytesPerSample := bitsPerSample / 8
	dataSize := numSamples * bytesPerSample * channels
	
	buf := new(bytes.Buffer)
	
	// RIFF header
	buf.WriteString("RIFF")
	binary.Write(buf, binary.LittleEndian, uint32(36+dataSize))
	buf.WriteString("WAVE")
	
	// fmt chunk
	buf.WriteString("fmt ")
	binary.Write(buf, binary.LittleEndian, uint32(16))                          // Chunk size
	binary.Write(buf, binary.LittleEndian, uint16(1))                           // Audio format (PCM)
	binary.Write(buf, binary.LittleEndian, uint16(channels))                    // Channels
	binary.Write(buf, binary.LittleEndian, uint32(sampleRate))                  // Sample rate
	binary.Write(buf, binary.LittleEndian, uint32(sampleRate*channels*bytesPerSample)) // Byte rate
	binary.Write(buf, binary.LittleEndian, uint16(channels*bytesPerSample))     // Block align
	binary.Write(buf, binary.LittleEndian, uint16(bitsPerSample))               // Bits per sample
	
	// data chunk
	buf.WriteString("data")
	binary.Write(buf, binary.LittleEndian, uint32(dataSize))
	
	// Audio data (silence)
	for i := 0; i < numSamples*channels; i++ {
		if bitsPerSample == 16 {
			binary.Write(buf, binary.LittleEndian, int16(0))
		} else if bitsPerSample == 8 {
			buf.WriteByte(128) // 8-bit is unsigned, 128 = silence
		}
	}
	
	return buf.Bytes()
}

// BenchmarkParseWAV benchmarks WAV parsing
func BenchmarkParseWAV(b *testing.B) {
	wav := createTestWAV(16000, 16, 1, 1.0)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		parseWAV(wav)
	}
}

// BenchmarkConvertToFloat32 benchmarks sample conversion
func BenchmarkConvertToFloat32(b *testing.B) {
	data := make([]byte, 32000) // 1 second of 16-bit audio
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		convertToFloat32(data, 1, 16, 16000)
	}
}

