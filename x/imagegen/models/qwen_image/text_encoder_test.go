//go:build mlx

package qwen_image

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/safetensors"
)

// TinyTextEncoderConfig holds config for the tiny test text encoder
type TinyTextEncoderConfig struct {
	HiddenSize        int32   `json:"hidden_size"`
	NumHiddenLayers   int32   `json:"num_hidden_layers"`
	IntermediateSize  int32   `json:"intermediate_size"`
	NumAttentionHeads int32   `json:"num_attention_heads"`
	NumKeyValueHeads  int32   `json:"num_key_value_heads"`
	VocabSize         int32   `json:"vocab_size"`
	RMSNormEps        float32 `json:"rms_norm_eps"`
	RopeTheta         float32 `json:"rope_theta"`
	HeadDim           int32   `json:"head_dim"`
	MRoPESection      []int32 `json:"mrope_section"`
}

// loadTinyTextEncoder loads the tiny text encoder from testdata
func loadTinyTextEncoder(t *testing.T) (*Qwen25VL, *TinyTextEncoderConfig) {
	t.Helper()

	testdataDir := filepath.Join("testdata", "tiny_text_encoder")

	// Load config
	configData, err := os.ReadFile(filepath.Join(testdataDir, "config.json"))
	if err != nil {
		t.Skipf("Skipping: tiny weights not found. Regenerate with Python (see models/CLAUDE.md)")
	}

	var tinyCfg TinyTextEncoderConfig
	if err := json.Unmarshal(configData, &tinyCfg); err != nil {
		t.Fatalf("Failed to parse config: %v", err)
	}

	// Create encoder config (using Qwen25VLConfig)
	cfg := &Qwen25VLConfig{
		HiddenSize:        tinyCfg.HiddenSize,
		NumHiddenLayers:   tinyCfg.NumHiddenLayers,
		IntermediateSize:  tinyCfg.IntermediateSize,
		NumAttentionHeads: tinyCfg.NumAttentionHeads,
		NumKeyValueHeads:  tinyCfg.NumKeyValueHeads,
		VocabSize:         tinyCfg.VocabSize,
		RMSNormEps:        tinyCfg.RMSNormEps,
		RopeTheta:         tinyCfg.RopeTheta,
		HeadDim:           tinyCfg.HeadDim,
		MRoPESection:      tinyCfg.MRoPESection,
	}

	// Load weights
	weights, err := safetensors.LoadModelWeights(testdataDir)
	if err != nil {
		t.Fatalf("Failed to load weights: %v", err)
	}

	if err := weights.Load(mlx.DtypeBFloat16); err != nil {
		t.Fatalf("Failed to bulk load weights: %v", err)
	}

	// Build encoder
	embedding, err := weights.Get("model.embed_tokens.weight")
	if err != nil {
		t.Fatalf("Failed to get embedding: %v", err)
	}

	blocks := make([]*VLTextBlock, cfg.NumHiddenLayers)
	for i := int32(0); i < cfg.NumHiddenLayers; i++ {
		block, err := newVLTextBlock(weights, int(i), cfg)
		if err != nil {
			t.Fatalf("Failed to load block %d: %v", i, err)
		}
		blocks[i] = block
	}

	finalNorm, err := weights.Get("model.norm.weight")
	if err != nil {
		t.Fatalf("Failed to get final norm: %v", err)
	}

	encoder := &Qwen25VL{
		Config:    cfg,
		Embedding: embedding,
		Blocks:    blocks,
		FinalNorm: finalNorm,
		HasVision: false, // Text-only mode
	}

	return encoder, &tinyCfg
}

// TestTextEncoderForward verifies the text encoder forward pass with tiny weights.
func TestTextEncoderForward(t *testing.T) {
	encoder, cfg := loadTinyTextEncoder(t)

	// Create test tokens (within vocab range)
	tokens := []int32{1, 2, 3, 4, 5}

	// Forward pass using EncodeTextOnly
	out := encoder.EncodeTextOnly(tokens)
	mlx.Eval(out)

	// Verify output shape: [batch, seq_len, hidden_size]
	wantShape := []int32{1, 5, cfg.HiddenSize}
	if !slices.Equal(out.Shape(), wantShape) {
		t.Errorf("output shape: got %v, want %v", out.Shape(), wantShape)
	}

	// Verify output is finite (not NaN or Inf)
	data := out.Data()
	for i, v := range data {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Errorf("output[%d] is not finite: %v", i, v)
			break
		}
	}
}

// TestTextEncoderBatch tests batch processing.
func TestTextEncoderBatch(t *testing.T) {
	encoder, cfg := loadTinyTextEncoder(t)

	// For batch test, we'll use EncodeTextOnly with a single sequence
	// (EncodeTextOnly doesn't support batch, but we can verify single sequence works)
	tokens := []int32{1, 2, 3}

	out := encoder.EncodeTextOnly(tokens)
	mlx.Eval(out)

	wantShape := []int32{1, 3, cfg.HiddenSize}
	if !slices.Equal(out.Shape(), wantShape) {
		t.Errorf("shape: got %v, want %v", out.Shape(), wantShape)
	}
}

// TestMRoPEComputation verifies M-RoPE frequency computation produces valid values.
func TestMRoPEComputation(t *testing.T) {
	encoder, cfg := loadTinyTextEncoder(t)

	cossin := encoder.computeTextRoPE(10, 1)
	mlx.Eval(cossin[0], cossin[1])

	// Verify shapes: [3, B, L, head_dim]
	wantShape := []int32{3, 1, 10, cfg.HeadDim}
	if !slices.Equal(cossin[0].Shape(), wantShape) {
		t.Errorf("cos shape: got %v, want %v", cossin[0].Shape(), wantShape)
	}
	if !slices.Equal(cossin[1].Shape(), wantShape) {
		t.Errorf("sin shape: got %v, want %v", cossin[1].Shape(), wantShape)
	}

	// Verify cos/sin values are in valid range [-1, 1]
	cosData := cossin[0].Data()
	sinData := cossin[1].Data()
	for i := 0; i < min(100, len(cosData)); i++ {
		if cosData[i] < -1.01 || cosData[i] > 1.01 {
			t.Errorf("cos[%d] out of range: %v", i, cosData[i])
		}
		if sinData[i] < -1.01 || sinData[i] > 1.01 {
			t.Errorf("sin[%d] out of range: %v", i, sinData[i])
		}
	}
}
