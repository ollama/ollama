package convert

import (
	"bytes"
	"cmp"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"log/slog"
	"math"
	"slices"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
	"github.com/x448/float16"
)

type modernBertModel struct {
	ModelParameters
	NumHiddenLayers           uint32  `json:"num_hidden_layers"`
	MaxPositionEmbeddings     uint32  `json:"max_position_embeddings"`
	HiddenSize                uint32  `json:"hidden_size"`
	IntermediateSize          uint32  `json:"intermediate_size"`
	NumAttentionHeads         uint32  `json:"num_attention_heads"`
	LayerNormEPS              float32 `json:"norm_eps"`
	GlobalAttnEveryNLayers    uint32  `json:"global_attn_every_n_layers"`
	LocalAttention            uint32  `json:"local_attention"`
	LocalRopeTheta            float32 `json:"local_rope_theta"`
	GlobalRopeTheta           float32 `json:"global_rope_theta"`
	HiddenActivation          string  `json:"hidden_activation"`
	ClassifierPooling         string  `json:"classifier_pooling"`
	normalizeEmbeddings       bool
	PoolingType               uint32
}

var (
	_ ModelConverter = (*modernBertModel)(nil)
	_ moreParser     = (*modernBertModel)(nil)
)

func (p *modernBertModel) parseMore(fsys fs.FS) error {
	// Parse sentence_transformers module config if present
	var hasPoolingModule bool
	bts, err := fs.ReadFile(fsys, "modules.json")
	if err == nil {
		// modules.json exists, parse it
		var modules []struct {
			Type string `json:"type"`
			Path string `json:"path"`
		}

		if err := json.Unmarshal(bts, &modules); err != nil {
			return err
		}

		for _, m := range modules {
			switch m.Type {
			case "sentence_transformers.models.Pooling":
				hasPoolingModule = true
			case "sentence_transformers.models.Normalize":
				p.normalizeEmbeddings = true
			}
		}
	}

	// Set pooling type based on available information
	// Priority: modules.json Pooling module > classifier_pooling config
	if hasPoolingModule {
		// ModernBERT embedding models use CLS pooling (first token)
		// The modules.json indicates this is a sentence-transformers model with a Pooling module
		slog.Debug("modernbert: detected sentence-transformers Pooling module, using CLS pooling")
		p.PoolingType = 2 // CLS pooling for embedding models
	} else {
		// No pooling module - fall back to classifier_pooling setting from config.json
		slog.Debug("modernbert pooling config", "classifier_pooling", p.ClassifierPooling)
		if p.ClassifierPooling == "mean" {
			p.PoolingType = 1 // Mean pooling
		} else if p.ClassifierPooling == "cls" {
			p.PoolingType = 2 // CLS pooling
		} else {
			// Default to CLS pooling for ModernBERT
			slog.Warn("modernbert: unknown classifier_pooling value, defaulting to CLS", "value", p.ClassifierPooling)
			p.PoolingType = 2
		}
	}

	return nil
}

func (p *modernBertModel) KV(t *Tokenizer) KV {
	slog.Info("=== ModernBERT KV() called ===")
	slog.Info("Tokenizer.Pre value from tokenizer.go", "pre", t.Pre)
	kv := p.ModelParameters.KV(t)
	slog.Info("After ModelParameters.KV(), tokenizer.ggml.pre", "value", kv["tokenizer.ggml.pre"])

	kv["general.architecture"] = "modernbert"
	kv["modernbert.attention.causal"] = false
	kv["modernbert.pooling_type"] = p.PoolingType
	kv["modernbert.normalize_embeddings"] = p.normalizeEmbeddings

	kv["modernbert.block_count"] = p.NumHiddenLayers
	kv["modernbert.context_length"] = p.MaxPositionEmbeddings
	kv["modernbert.embedding_length"] = p.HiddenSize
	kv["modernbert.feed_forward_length"] = p.IntermediateSize
	kv["modernbert.attention.head_count"] = p.NumAttentionHeads
	kv["modernbert.attention.layer_norm_epsilon"] = p.LayerNormEPS

	// ModernBERT-specific parameters for alternating attention
	kv["modernbert.attention.global_attn_every_n_layers"] = cmp.Or(p.GlobalAttnEveryNLayers, 3)
	kv["modernbert.attention.local_attn_window"] = cmp.Or(p.LocalAttention, 128)
	kv["modernbert.rope.freq_base_local"] = cmp.Or(p.LocalRopeTheta, 10000.0)
	kv["modernbert.rope.freq_base_global"] = cmp.Or(p.GlobalRopeTheta, 80000.0)

	// Set general rope.freq_base to the global value (used as default by llama.cpp)
	kv["general.rope.freq_base"] = cmp.Or(p.GlobalRopeTheta, 80000.0)

	// ModernBERT uses GPT2/BPE tokenizer (like RoBERTa), not BERT WordPiece
	kv["tokenizer.ggml.model"] = "gpt2"
	// CRITICAL: Must use "gpt-2" pre-tokenizer, not "default"!
	// "default" has extra regex patterns that split numbers incorrectly
	slog.Info("Setting tokenizer.ggml.pre to 'gpt-2' for ModernBERT")
	kv["tokenizer.ggml.pre"] = "gpt-2"
	kv["tokenizer.ggml.token_type_count"] = uint32(2)
	slog.Info("ModernBERT tokenizer configured", "model", kv["tokenizer.ggml.model"], "pre", kv["tokenizer.ggml.pre"])

	// Tokens are already set by ModelParameters.KV(t) - don't overwrite

	// BERT-like models need CLS (as BOS) and SEP (as EOS) tokens added automatically
	// llama.cpp uses add_bos_token and add_eos_token with bos_token_id and eos_token_id
	kv["tokenizer.ggml.bos_token_id"] = uint32(50281) // CLS token
	kv["tokenizer.ggml.eos_token_id"] = uint32(50282) // SEP token
	kv["tokenizer.ggml.add_bos_token"] = true
	kv["tokenizer.ggml.add_eos_token"] = true

	slog.Info("=== Final KV check before return ===")
	slog.Info("Returning kv map", "tokenizer.ggml.pre", kv["tokenizer.ggml.pre"], "tokenizer.ggml.model", kv["tokenizer.ggml.model"])

	return kv
}

func (p *modernBertModel) Tensors(ts []Tensor) []*ggml.Tensor {
	slog.Info("TENSORS_DEBUG: Tensors() called", "count", len(ts))
	var out []*ggml.Tensor

	for i, t := range ts {
		if i < 5 || strings.Contains(t.Name(), "Wqkv") {
			slog.Info("TENSORS_DEBUG: Processing tensor", "index", i, "name", t.Name(), "shape", t.Shape())
		}
		// Skip pooler layers and position IDs (we do pooling in the runtime)
		if slices.Contains([]string{
			"embeddings.position_ids",
			"pooler.dense.weight",
			"pooler.dense.bias",
		}, t.Name()) {
			continue
		}

		name := t.Name()

		// Skip attention projection bias tensors for global attention layers
		// Full attention layers don't have attention biases while local attention layers do
		// Don't skip normalization biases (attn_output_norm.bias)
		if strings.Contains(name, ".bias") && !strings.Contains(name, "norm") && (strings.Contains(name, "attn") || strings.Contains(name, "attention")) {
			// Apply layer prefix replacements to parse the layer number correctly
			layerName := name
			layerName = strings.Replace(layerName, "encoder.layer.", "blk.", 1)
			layerName = strings.Replace(layerName, "encoder.layers.", "blk.", 1)
			layerName = strings.Replace(layerName, "layers.", "blk.", 1)

			var layer int
			if _, err := fmt.Sscanf(layerName, "blk.%d.", &layer); err == nil {
				globalAttnEveryN := cmp.Or(p.GlobalAttnEveryNLayers, 3)
				// Skip if it's a global layer (multiple of N) - this includes layer 0
				if layer%int(globalAttnEveryN) == 0 {
					continue
				}
			}
		}


		// ModernBERT uses GeGLU (Gated GELU) - the mlp.Wi tensor contains both gate and up weights fused
		// We need to split it into two separate tensors
		if strings.Contains(name, "mlp.Wi") {
			shape := t.Shape()
			if len(shape) != 2 {
				// Unexpected shape, just pass through
				out = append(out, &ggml.Tensor{
					Name:     name,
					Kind:     t.Kind(),
					Shape:    shape,
					WriterTo: t,
				})
				continue
			}

			// PyTorch stores linear weights as [out_features, in_features]
			// For GeGLU, shape is [2*intermediate_size, hidden_size]
			// We need to split along dim 0 into two tensors of [intermediate_size, hidden_size]
			dim0 := shape[0]
			dim1 := shape[1]
			halfDim0 := dim0 / 2

			// Create ffn_gate tensor (first half of rows)
			// ModernBERT's mlp.Wi is organized as [gate; up] (concatenated along dim 0)
			gateName := strings.Replace(name, "mlp.Wi", "ffn_gate", 1)
			out = append(out, &ggml.Tensor{
				Name:     gateName,
				Kind:     t.Kind(),
				Shape:    []uint64{halfDim0, dim1},
				WriterTo: &splitTensorRows{source: t, offset: 0, rows: halfDim0},
			})

			// Create ffn_up tensor (second half of rows)
			upName := strings.Replace(name, "mlp.Wi", "ffn_up", 1)
			out = append(out, &ggml.Tensor{
				Name:     upName,
				Kind:     t.Kind(),
				Shape:    []uint64{halfDim0, dim1},
				WriterTo: &splitTensorRows{source: t, offset: halfDim0, rows: halfDim0},
			})
		} else {
			out = append(out, &ggml.Tensor{
				Name:     name,
				Kind:     t.Kind(),
				Shape:    t.Shape(),
				WriterTo: t,
			})
		}
	}

	// PRE-NORM Architecture: Layer 0 does NOT have attn_norm in source model (it's Identity/no-op in HuggingFace)
	// However, llama.cpp graph builder may need a tensor to exist. Create an identity norm (all 1s) for layer 0
	hasLayer0AttnNorm := false
	var hiddenSize uint64

	for _, t := range out {
		if t.Name == "blk.0.attn_output_norm.weight" {
			hasLayer0AttnNorm = true
		}
		// Get hidden size from any layer's attn_output_norm (they all have same size)
		if strings.Contains(t.Name, "attn_output_norm.weight") {
			if len(t.Shape) > 0 {
				hiddenSize = t.Shape[0]
			}
		}
	}

	if !hasLayer0AttnNorm && hiddenSize > 0 {
		// Create identity LayerNorm for layer 0 (weight=1.0, bias=0.0)
		out = append(out, &ggml.Tensor{
			Name:     "blk.0.attn_output_norm.weight",
			Kind:     tensorKindFP32,
			Shape:    []uint64{hiddenSize},
			WriterTo: &identityTensor{size: hiddenSize, value: 1.0},
		})
	}

	return out
}

// identityTensor creates a tensor filled with a constant value
type identityTensor struct {
	size  uint64
	value float32
}

func (it *identityTensor) WriteTo(w io.Writer) (n int64, err error) {
	buf := make([]byte, 4)
	for i := uint64(0); i < it.size; i++ {
		binary.LittleEndian.PutUint32(buf, math.Float32bits(it.value))
		written, err := w.Write(buf)
		n += int64(written)
		if err != nil {
			return n, err
		}
	}
	return n, nil
}

// getTensorKindSize returns the size in bytes for each element type
func getTensorKindSize(kind uint32) uint64 {
	switch kind {
	case tensorKindFP32: // 0
		return 4
	case tensorKindFP16: // 1
		return 2
	case tensorKindBF16: // 30
		return 2
	default:
		// Unknown kind, assume FP32
		return 4
	}
}

// tensorF32Wrapper wraps a tensor and forces it to be written as F32
type tensorF32Wrapper struct {
	source Tensor
}

func (t *tensorF32Wrapper) Name() string {
	return t.source.Name()
}

func (t *tensorF32Wrapper) Kind() uint32 {
	return tensorKindFP32 // Always return F32
}

func (t *tensorF32Wrapper) Shape() []uint64 {
	return t.source.Shape()
}

func (t *tensorF32Wrapper) SetRepacker(fn Repacker) {
	t.source.SetRepacker(fn)
}

func (t *tensorF32Wrapper) WriteTo(w io.Writer) (int64, error) {
	// Write source to buffer first
	var buf bytes.Buffer
	if _, err := t.source.WriteTo(&buf); err != nil {
		return 0, err
	}

	// If source is already F32, just write the buffer
	if t.source.Kind() == tensorKindFP32 {
		nn, err := w.Write(buf.Bytes())
		return int64(nn), err
	}

	// Convert F16 to F32
	data := buf.Bytes()
	if t.source.Kind() == tensorKindFP16 {
		// Read as F16
		numElements := len(data) / 2
		f32s := make([]float32, numElements)

		for i := 0; i < numElements; i++ {
			u16 := uint16(data[i*2]) | (uint16(data[i*2+1]) << 8)
			f32s[i] = float16.Frombits(u16).Float32()
		}

		// Write as F32
		return int64(len(f32s) * 4), binary.Write(w, binary.LittleEndian, f32s)
	}

	// Convert BF16 to F32
	if t.source.Kind() == tensorKindBF16 {
		// Read as BF16
		numElements := len(data) / 2
		f32s := make([]float32, numElements)

		for i := 0; i < numElements; i++ {
			// BF16 is just the top 16 bits of F32
			// To convert: shift BF16 value left by 16 bits
			u16 := uint16(data[i*2]) | (uint16(data[i*2+1]) << 8)
			u32 := uint32(u16) << 16
			f32s[i] = math.Float32frombits(u32)
		}

		// Write as F32
		return int64(len(f32s) * 4), binary.Write(w, binary.LittleEndian, f32s)
	}

	// For other types, just pass through
	nn, err := w.Write(data)
	return int64(nn), err
}

func (t *tensorF32Wrapper) Clone() Tensor {
	return &tensorF32Wrapper{source: t.source.Clone()}
}

// splitTensorRows handles splitting a fused tensor along dimension 0 (rows)
type splitTensorRows struct {
	source Tensor
	offset uint64 // starting row
	rows   uint64 // number of rows to extract
}

func (st *splitTensorRows) Name() string {
	return st.source.Name()
}

func (st *splitTensorRows) Kind() uint32 {
	return st.source.Kind()
}

func (st *splitTensorRows) Shape() []uint64 {
	shape := st.source.Shape()
	if len(shape) == 2 {
		return []uint64{st.rows, shape[1]}
	}
	return shape
}

func (st *splitTensorRows) SetRepacker(fn Repacker) {
	st.source.SetRepacker(fn)
}

func (st *splitTensorRows) Clone() Tensor {
	return &splitTensorRows{
		source: st.source.Clone(),
		offset: st.offset,
		rows:   st.rows,
	}
}

func (st *splitTensorRows) WriteTo(w io.Writer) (n int64, err error) {
	shape := st.source.Shape()
	if len(shape) != 2 {
		return 0, fmt.Errorf("splitTensorRows only works with 2D tensors")
	}

	dim1 := shape[1]  // columns
	elemSize := getTensorKindSize(st.source.Kind())

	// Read the entire source tensor
	var buf bytes.Buffer
	if _, err := st.source.WriteTo(&buf); err != nil {
		return 0, err
	}
	data := buf.Bytes()

	// Calculate byte offsets
	// Each row is dim1 elements
	rowSizeBytes := dim1 * elemSize
	startByte := st.offset * rowSizeBytes
	endByte := (st.offset + st.rows) * rowSizeBytes

	// Write the contiguous block of rows
	nn, err := w.Write(data[startByte:endByte])
	return int64(nn), err
}

func (modernBertModel) Replacements() []string {
	return []string{
		// ModernBERT uses "layers.N" not "encoder.layers.N"
		"layers", "blk",
		"encoder.layer", "blk",
		"encoder.layers", "blk",
		"embeddings.tok_embeddings", "token_embd",
		"embeddings.word_embeddings", "token_embd",
		"embeddings.norm", "token_embd_norm",
		"embeddings.LayerNorm", "token_embd_norm",
		"final_norm", "output_norm",
		"attn.Wqkv", "attn_qkv",
		"attn.Wo", "attn_output",
		"attention.self.query", "attn_q",
		"attention.self.key", "attn_k",
		"attention.self.value", "attn_v",
		"attention.output.dense", "attn_output",
		"attention.output.LayerNorm", "attn_output_norm",
		"attn_norm", "attn_output_norm",
		// ModernBERT uses gated FFN - tensors are split in Tensors() method
		"mlp.Wgate", "ffn_gate",
		"mlp.Wup", "ffn_up",
		"mlp.Wo", "ffn_down",
		"intermediate.dense", "ffn_up",
		"output.dense", "ffn_down",
		"output.LayerNorm", "layer_output_norm",
		"mlp_norm", "layer_output_norm",
	}
}

func (modernBertModel) specialTokenTypes() []string {
	return []string{
		"bos", "eos", "unk", "sep", "pad", "cls", "mask",
	}
}
