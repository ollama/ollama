# Implementation Plan: IBM Granite Embedding English R2 (ModernBERT)

## Executive Summary

Implement support for `ibm-granite/granite-embedding-english-r2` model in Ollama by extending the existing NomicBERT architecture with ModernBERT-specific features, particularly the alternating global/local attention pattern.

## Architecture Analysis

### Model Specifications
- **Base Architecture**: ModernBERT (evolution of NomicBERT)
- **Parameters**: 149M
- **Embedding Dimension**: 768
- **Layers**: 22
- **Attention Heads**: 12
- **Intermediate Size**: 1152
- **Context Length**: 8192 tokens
- **Activation**: GELU
- **Pooling**: CLS pooling (cls_token_id: 50281)

### Key Architectural Features

1. **Alternating Attention Pattern** (Critical difference from NomicBERT)
   - Global attention every 3 layers (layers 0, 3, 6, 9, 12, 15, 18, 21)
   - Local sliding window (128 tokens) on other layers
   - Different RoPE theta values:
     - Global layers: θ = 80000.0
     - Local layers: θ = 10000.0

2. **Shared with NomicBERT**
   - Rotary Position Embeddings (RoPE)
   - GELU activation
   - No bias terms (attention_bias: false, mlp_bias: false, norm_bias: false)
   - Non-causal attention (bidirectional)
   - Flash Attention 2 support

3. **Tokenizer**
   - Vocabulary size: 50,368
   - Special tokens: BOS (50281), EOS (50282), PAD (50283), SEP (50282)

## Implementation Strategy: Hybrid Approach

### Phase 1: C++ Backend Changes (llama.cpp)

#### 1.1 Add ModernBERT Architecture Support

**File**: `llama/llama.cpp/src/llama-arch.h`
- Add `LLM_ARCH_MODERNBERT` enum entry after NOMIC_BERT variants

**File**: `llama/llama.cpp/src/llama-arch.cpp`
- Register architecture name: `{ LLM_ARCH_MODERNBERT, "modernbert" }`
- Add tensor mapping (based on BERT/NomicBERT pattern)

#### 1.2 Extend llama_hparams Structure

**File**: `llama/llama.cpp/src/llama-hparams.h`
- Add new fields:
  ```cpp
  uint32_t global_attn_every_n_layers = 0;  // Pattern for global attention
  uint32_t local_attn_window = 0;            // Sliding window size for local attention
  float    rope_freq_base_local = 10000.0f; // RoPE theta for local layers
  float    rope_freq_base_global = 10000.0f;// RoPE theta for global layers
  ```

#### 1.3 Model Loading

**File**: `llama/llama.cpp/src/llama-model.cpp`
- Add ModernBERT config parsing (case LLM_ARCH_MODERNBERT)
- Read `global_attn_every_n_layers`, `local_attention` from config
- Set up hyperparameters including dual RoPE theta values
- Handle pooling type (CLS pooling)

#### 1.4 Modify BERT Graph Builder

**File**: `llama/llama.cpp/src/models/bert.cpp`
- Extend `llm_build_bert` to handle ModernBERT
- Add logic to check if layer uses global vs local attention:
  ```cpp
  bool is_global_layer = (hparams.global_attn_every_n_layers > 0) &&
                         (il % hparams.global_attn_every_n_layers == 0);
  ```
- Apply different RoPE theta based on layer type:
  ```cpp
  float rope_theta = is_global_layer ? hparams.rope_freq_base_global
                                     : hparams.rope_freq_base_local;
  ```
- Implement local attention masking for non-global layers:
  ```cpp
  if (!is_global_layer && hparams.local_attn_window > 0) {
      // Apply sliding window mask
      inp_attn = build_attn_inp_with_window(hparams.local_attn_window);
  }
  ```

### Phase 2: Go Conversion Layer (Ollama)

#### 2.1 Create ModernBERT Converter

**File**: `convert/convert_modernbert.go`

```go
package convert

import (
	"cmp"
	"encoding/json"
	"io/fs"
	"path/filepath"
	"slices"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

type modernBertModel struct {
	ModelParameters
	NumHiddenLayers           uint32  `json:"num_hidden_layers"`
	MaxPositionEmbeddings     uint32  `json:"max_position_embeddings"`
	HiddenSize                uint32  `json:"hidden_size"`
	IntermediateSize          uint32  `json:"intermediate_size"`
	NumAttentionHeads         uint32  `json:"num_attention_heads"`
	LayerNormEPS              float32 `json:"layer_norm_eps"`
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
	bts, err := fs.ReadFile(fsys, "modules.json")
	if err != nil {
		// Not all models have this, return nil if missing
		return nil
	}

	var modules []struct {
		Type string `json:"type"`
		Path string `json:"path"`
	}

	if err := json.Unmarshal(bts, &modules); err != nil {
		return err
	}

	for _, m := range modules {
		if m.Type == "sentence_transformers.models.Normalize" {
			p.normalizeEmbeddings = true
		}
	}

	// ModernBERT uses CLS pooling by default
	if p.ClassifierPooling == "cls" || p.ClassifierPooling == "mean" {
		if p.ClassifierPooling == "mean" {
			p.PoolingType = 1 // Mean pooling
		} else {
			p.PoolingType = 2 // CLS pooling
		}
	} else {
		p.PoolingType = 2 // Default to CLS
	}

	return nil
}

func (p *modernBertModel) KV(t *Tokenizer) ggml.KV {
	kv := p.ModelParameters.KV(t)

	kv["general.architecture"] = "modernbert"
	kv["bert.attention.causal"] = false
	kv["bert.pooling_type"] = p.PoolingType
	kv["bert.normalize_embeddings"] = p.normalizeEmbeddings

	kv["bert.block_count"] = p.NumHiddenLayers
	kv["bert.context_length"] = p.MaxPositionEmbeddings
	kv["bert.embedding_length"] = p.HiddenSize
	kv["bert.feed_forward_length"] = p.IntermediateSize
	kv["bert.attention.head_count"] = p.NumAttentionHeads
	kv["bert.attention.layer_norm_epsilon"] = p.LayerNormEPS

	// ModernBERT specific parameters
	kv["modernbert.global_attn_every_n_layers"] = p.GlobalAttnEveryNLayers
	kv["modernbert.local_attn_window"] = p.LocalAttention
	kv["modernbert.rope_freq_base_local"] = cmp.Or(p.LocalRopeTheta, 10000.0)
	kv["modernbert.rope_freq_base_global"] = cmp.Or(p.GlobalRopeTheta, 80000.0)

	kv["tokenizer.ggml.model"] = "bert"
	kv["tokenizer.ggml.token_type_count"] = uint32(2)

	// Convert to phantom space tokens (like BERT)
	for i, e := range t.Tokens {
		if strings.HasPrefix(e, "[") && strings.HasSuffix(e, "]") {
			// Keep special tokens as-is
		} else if strings.HasPrefix(e, "##") {
			t.Tokens[i] = e[2:]
		} else {
			t.Tokens[i] = "\u2581" + e
		}
	}

	kv["tokenizer.ggml.tokens"] = t.Tokens

	return kv
}

func (p *modernBertModel) Tensors(ts []Tensor) []*ggml.Tensor {
	var out []*ggml.Tensor
	for _, t := range ts {
		// Skip pooler layers (we do pooling in the runtime)
		if slices.Contains([]string{
			"embeddings.position_ids",
			"pooler.dense.weight",
			"pooler.dense.bias",
		}, t.Name()) {
			continue
		}

		out = append(out, &ggml.Tensor{
			Name:     t.Name(),
			Kind:     t.Kind(),
			Shape:    t.Shape(),
			WriterTo: t,
		})
	}

	return out
}

func (modernBertModel) Replacements() []string {
	return []string{
		"encoder.layer", "blk",
		"encoder.layers", "blk",
		"embeddings.word_embeddings", "token_embd",
		"embeddings.token_type_embeddings", "token_types",
		"embeddings.LayerNorm", "token_embd_norm",
		"embeddings.position_embeddings", "position_embd",
		"attention.self.query", "attn_q",
		"attention.self.key", "attn_k",
		"attention.self.value", "attn_v",
		"attention.output.dense", "attn_output",
		"attention.output.LayerNorm", "attn_output_norm",
		"intermediate.dense", "ffn_up",
		"output.dense", "ffn_down",
		"output.LayerNorm", "layer_output_norm",
	}
}

func (modernBertModel) specialTokenTypes() []string {
	return []string{
		"bos", "eos", "unk", "sep", "pad", "cls", "mask",
	}
}
```

#### 2.2 Register ModernBERT in Converter

**File**: `convert/convert.go` (around line 200)

Add to the switch statement in `ConvertModel()`:
```go
case "ModernBertModel", "ModernBertForMaskedLM":
	conv = &modernBertModel{}
```

### Phase 3: Testing & Validation

#### 3.1 Download and Convert Model

```bash
# Download from HuggingFace
git clone https://huggingface.co/ibm-granite/granite-embedding-english-r2

# Create Modelfile
cat > Modelfile <<EOF
FROM ./granite-embedding-english-r2
TEMPLATE ""
PARAMETER stop "<|endoftext|>"
EOF

# Convert using Ollama
ollama create granite-embedding-r2 -f Modelfile
```

#### 3.2 Test Embedding Generation

```bash
# Test basic embedding
ollama run granite-embedding-r2 "This is a test sentence"

# Test programmatically
curl http://localhost:11434/api/embeddings -d '{
  "model": "granite-embedding-r2",
  "prompt": "The quick brown fox jumps over the lazy dog"
}'
```

#### 3.3 Validation Checklist

- [ ] Model loads without errors
- [ ] Embedding dimension is 768
- [ ] Context length supports up to 8192 tokens
- [ ] CLS pooling is applied correctly
- [ ] Embeddings are normalized (if enabled)
- [ ] Performance is reasonable (~3x faster than BERT as claimed)
- [ ] Compare embeddings with HuggingFace reference implementation

### Phase 4: Optional Enhancements

#### 4.1 Add Integration Test

**File**: `integration/embed_test.go`

Add granite-embedding-r2 to test suite for embedding models.

#### 4.2 Update Documentation

**File**: `CLAUDE.md`

Add ModernBERT to the supported architectures section.

## Technical Challenges & Solutions

### Challenge 1: Alternating Attention Implementation

**Problem**: ModernBERT uses global attention every 3 layers, local sliding window on others.

**Solution**:
- Use existing `moe_every_n_layers` pattern as reference
- Add conditional logic in attention builder to select between:
  - Full attention matrix (global layers)
  - Sliding window mask (local layers)
- Apply different RoPE theta based on layer type

### Challenge 2: Sliding Window Attention

**Problem**: llama.cpp has sliding window support but needs to be integrated with BERT graph builder.

**Solution**:
- Check `llama/llama.cpp/src/llama-graph.cpp` for sliding window implementations
- Adapt for non-causal bidirectional attention (different from decoder-only models)
- Create local attention mask: position i can attend to [i-64, i+64] window

### Challenge 3: Dual RoPE Theta Values

**Problem**: Different RoPE frequency bases for global vs local layers.

**Solution**:
- Extend RoPE application to accept per-layer theta
- Pass correct theta value based on `is_global_layer` flag
- May require per-layer RoPE parameter storage

## Risk Assessment

### High Risk
- **Sliding window implementation complexity**: Medium effort, requires careful testing
- **Performance validation**: Need to ensure parity with HuggingFace implementation

### Medium Risk
- **RoPE dual-theta support**: Minor llama.cpp changes needed
- **Tokenizer compatibility**: Should work with existing BERT tokenizer support

### Low Risk
- **Go converter implementation**: Straightforward extension of BERT converter
- **Pooling mechanism**: Already supported (CLS and mean pooling)

## Timeline Estimate

| Phase | Estimated Effort | Description |
|-------|-----------------|-------------|
| 1.1-1.2 | 2-3 hours | Add architecture enum, update hparams |
| 1.3 | 2-3 hours | Model loading and config parsing |
| 1.4 | 4-6 hours | Implement alternating attention in graph builder |
| 2.1-2.2 | 2-3 hours | Go converter implementation |
| 3.1-3.3 | 2-3 hours | Testing and validation |
| **Total** | **12-18 hours** | Full implementation |

## Alternative Approaches Considered

### 1. Quick Workaround (Not Chosen)
- Convert as NomicBERT, ignore alternating attention
- **Pros**: Fast (1-2 hours)
- **Cons**: Incorrect architecture, suboptimal quality

### 2. Full ModernBERT from Scratch (Not Chosen)
- Implement entirely new architecture type
- **Pros**: Clean separation
- **Cons**: Duplicates BERT code, longer development time

### 3. Hybrid Approach (CHOSEN)
- Extend NOMIC_BERT with ModernBERT features
- **Pros**: Reuses existing code, proper implementation
- **Cons**: Moderate complexity

## Success Criteria

1. ✅ Model converts without errors
2. ✅ Generates embeddings with correct dimension (768)
3. ✅ Supports full 8192 token context
4. ✅ Embedding quality matches HuggingFace reference (cosine similarity > 0.99)
5. ✅ Inference performance is reasonable (target: < 100ms for 512 tokens on GPU)

## References

- [ModernBERT Paper](https://arxiv.org/html/2502.19587v1)
- [Granite Embedding R2 Models](https://arxiv.org/html/2508.21085v1)
- [HuggingFace Model Card](https://huggingface.co/ibm-granite/granite-embedding-english-r2)
- [llama.cpp Issue #11282](https://github.com/ggml-org/llama.cpp/issues/11282)
- [ModernBERT vs BERT Comparison](https://huggingface.co/blog/modernbert)
- [IBM Granite Embedding Models GitHub](https://github.com/ibm-granite/granite-embedding-models)
