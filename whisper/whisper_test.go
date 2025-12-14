package whisper

import (
	"testing"
	"time"
)

// TestSampleRateConstant verifies the sample rate is correct
func TestSampleRateConstant(t *testing.T) {
	if SampleRate != 16000 {
		t.Errorf("SampleRate = %d, want 16000", SampleRate)
	}
}

// TestDefaultTranscribeParams verifies default parameters
func TestDefaultTranscribeParams(t *testing.T) {
	params := DefaultTranscribeParams()
	
	if params.Language != "" {
		t.Errorf("Language = %q, want empty string for auto-detect", params.Language)
	}
	
	if params.Translate {
		t.Error("Translate should be false by default")
	}
	
	if params.Strategy != SamplingGreedy {
		t.Errorf("Strategy = %v, want SamplingGreedy", params.Strategy)
	}
	
	if params.Temperature != 0.0 {
		t.Errorf("Temperature = %f, want 0.0", params.Temperature)
	}
}

// TestDefaultContextParams verifies default context parameters
func TestDefaultContextParams(t *testing.T) {
	params := DefaultContextParams()
	
	if !params.UseGPU {
		t.Error("UseGPU should be true by default")
	}
	
	if params.GPUDevice != 0 {
		t.Errorf("GPUDevice = %d, want 0", params.GPUDevice)
	}
}

// TestLanguageID tests language code to ID conversion
func TestLanguageID(t *testing.T) {
	tests := []struct {
		lang     string
		wantPositive bool
	}{
		{"en", true},
		{"fr", true},
		{"de", true},
		{"invalid_lang", false},
	}
	
	for _, tt := range tests {
		id := LanguageID(tt.lang)
		if tt.wantPositive && id < 0 {
			t.Errorf("LanguageID(%q) = %d, want positive", tt.lang, id)
		}
		if !tt.wantPositive && id >= 0 {
			t.Errorf("LanguageID(%q) = %d, want negative", tt.lang, id)
		}
	}
}

// TestSegment verifies Segment struct
func TestSegment(t *testing.T) {
	seg := Segment{
		Text:  "Hello, world!",
		Start: 0,
		End:   2 * time.Second,
	}
	
	if seg.Text != "Hello, world!" {
		t.Errorf("Segment.Text = %q, want \"Hello, world!\"", seg.Text)
	}
	
	if seg.End-seg.Start != 2*time.Second {
		t.Errorf("Segment duration = %v, want 2s", seg.End-seg.Start)
	}
}

// TestToken verifies Token struct
func TestToken(t *testing.T) {
	token := Token{
		ID:          42,
		Text:        "test",
		Probability: 0.95,
		Start:       0,
		End:         100,
	}
	
	if token.ID != 42 {
		t.Errorf("Token.ID = %d, want 42", token.ID)
	}
	
	if token.Probability < 0 || token.Probability > 1 {
		t.Errorf("Token.Probability = %f, want between 0 and 1", token.Probability)
	}
}

// TestModelInfo verifies ModelInfo struct
func TestModelInfo(t *testing.T) {
	info := ModelInfo{
		Multilingual: true,
		VocabSize:    51864,
		AudioCtxSize: 1500,
		TextCtxSize:  448,
	}
	
	if !info.Multilingual {
		t.Error("ModelInfo.Multilingual should be true")
	}
	
	if info.VocabSize <= 0 {
		t.Errorf("ModelInfo.VocabSize = %d, want positive", info.VocabSize)
	}
}

// TestSpecialTokens verifies SpecialTokens struct
func TestSpecialTokens(t *testing.T) {
	tokens := SpecialTokens{
		EOT:        50256,
		SOT:        50257,
		Translate:  50258,
		Transcribe: 50259,
	}
	
	if tokens.EOT == tokens.SOT {
		t.Error("EOT and SOT should be different")
	}
	
	if tokens.Translate == tokens.Transcribe {
		t.Error("Translate and Transcribe tokens should be different")
	}
}

// BenchmarkDefaultTranscribeParams benchmarks parameter creation
func BenchmarkDefaultTranscribeParams(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = DefaultTranscribeParams()
	}
}

