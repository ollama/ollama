package gguf

import "log/slog"

// This file computes the size of a model's KV cache from its metadata alone,
// for PredictServerVRAM in llm/llama_server.go — the pre-load estimate the
// scheduler leans on before llama-server exists to measure anything.
//
// Everything here is derived from what llama.cpp itself does at load time, and
// checked against llama.cpp's own buffer reporting to the MiB. See the comments
// on the individual functions for the verified figures.

// uintArray reads a metadata key that may be either a scalar or a per-block
// array, and always returns it as an array. A scalar becomes a one-element
// array; a missing key becomes a one-element array holding defaultValue.
//
// Both unsigned and signed arrays occur in the wild for the same keys, so the
// signed form is converted. A negative element would be meaningless for every
// caller here, so it is logged rather than silently wrapped.
func (m *Metadata) uintArray(key string, defaultValue uint64) []uint64 {
	value := m.KeyValue(key).Value
	if u64s := value.Uints(); len(u64s) > 0 {
		return u64s
	}
	if i64s := value.Ints(); len(i64s) > 0 {
		out := make([]uint64, len(i64s))
		for i, v := range i64s {
			if v < 0 {
				slog.Warn("array values are unexpectedly negative", "key", key, "i", i, "v", v)
			}
			out[i] = uint64(v)
		}
		return out
	}
	if n, ok := m.UintOK(key); ok {
		return []uint64{n}
	}
	return []uint64{defaultValue}
}

// HeadCountKV returns the KV head count per block, one element per block.
//
// A scalar in the metadata applies to every block. A shorter array is padded
// with that scalar's role: the sole element if there was one, otherwise 1.
func (m *Metadata) HeadCountKV() []uint64 {
	headCountKV := m.uintArray("attention.head_count_kv", 1)

	fallback := uint64(1)
	if len(headCountKV) == 1 {
		fallback = headCountKV[0]
	}

	nLayers := int(m.BlockCount())
	if len(headCountKV) > nLayers {
		slog.Warn("got more elements of attention.head_count_kv than layers", "elements", len(headCountKV), "layers", nLayers)
	}

	out := make([]uint64, nLayers)
	for i := range out {
		if i < len(headCountKV) {
			out[i] = headCountKV[i]
		} else {
			out[i] = fallback
		}
	}
	return out
}

// HeadCountKVMax returns the largest KV head count over all blocks.
func (m *Metadata) HeadCountKVMax() uint64 {
	_, maximum := m.uintRange("attention.head_count_kv", 1)
	return maximum
}

// EmbeddingHeadCountMax returns the per-head embedding width implied by the
// embedding length and the smallest head count.
func (m *Metadata) EmbeddingHeadCountMax() uint64 {
	minimum, _ := m.uintRange("attention.head_count", 1)
	if minimum > 0 {
		return m.EmbeddingLength() / minimum
	}
	return 0
}

// EmbeddingHeadCountK returns the width of one K head.
//
// attention.key_length is authoritative where present: models that scale K and
// V independently of the embedding length (deepseek2, qwen3 and relatives) are
// mis-sized by embedding_length/head_count.
func (m *Metadata) EmbeddingHeadCountK() uint64 {
	return m.Uint("attention.key_length", m.EmbeddingHeadCountMax())
}

// EmbeddingHeadCountV returns the width of one V head. See EmbeddingHeadCountK.
func (m *Metadata) EmbeddingHeadCountV() uint64 {
	return m.Uint("attention.value_length", m.EmbeddingHeadCountMax())
}

// AttentionLayers reports which of the model's blocks hold a KV cache.
//
// Hybrid models mix full-attention and recurrent (linear attention / SSM)
// blocks, and only the former allocate a KV cache. Getting this wrong is not a
// rounding error: qwen35 has 16 attention blocks out of 65, so assuming all of
// them inflates the KV estimate four-fold.
//
// The three sources are checked in llama.cpp's own order of precedence (see
// src/models/qwen35.cpp and the LLM_KV_ATTENTION_RECURRENT_LAYERS handling):
//
//  1. attention.recurrent_layers — an explicit per-block flag, most authoritative
//  2. full_attention_interval — block i is full attention iff (i+1)%interval == 0
//  3. attention.head_count_kv as a per-block array — zero means recurrent
//
// With none of them present every block is a full-attention block, which is
// correct for ordinary transformers.
func (m *Metadata) AttentionLayers() []bool {
	out := make([]bool, m.BlockCount())

	if recurrent := m.uintArray("attention.recurrent_layers", 0); len(recurrent) > 1 {
		for i := range out {
			out[i] = i >= len(recurrent) || recurrent[i] == 0
		}
		return out
	}

	if interval := int(m.Uint("full_attention_interval")); interval > 0 {
		for i := range out {
			out[i] = (i+1)%interval == 0
		}
		return out
	}

	headsKV := m.HeadCountKV()
	for i := range out {
		out[i] = i < len(headsKV) && headsKV[i] > 0
	}
	return out
}

// swaInterleave describes how an architecture alternates sliding-window and
// full-attention blocks, and where its window width comes from.
//
// llama.cpp hardcodes this per architecture in src/models/*.cpp — the GGUF
// carries attention.sliding_window, but the interleave pattern is only in the
// file for the handful of models that write attention.sliding_window_pattern.
// Reproducing the defaults is the only way to size the cache before load.
type swaInterleave struct {
	// pattern is llama.cpp's n_pattern argument to load_swa_pattern.
	pattern uint64

	// denseFirst mirrors the dense_first argument: with it, block i is SWA iff
	// i%pattern != 0; without it, iff i%pattern < pattern-1.
	denseFirst bool

	// defaultWindow is the window used when the GGUF omits
	// attention.sliding_window. Zero means the architecture has no SWA without
	// that key, which is the common case.
	defaultWindow uint64

	// forcedWindow, when non-zero, overrides whatever the GGUF declares —
	// llama4 and smallthinker both pin the window in code.
	forcedWindow uint64

	// requiredBlocks, when non-zero, restricts SWA to models with exactly that
	// many blocks. exaone4 enables it only for the 64-block 32B.
	requiredBlocks uint64

	// allBlocks marks the architectures that call set_swa_pattern(0), where
	// every block is sliding-window — including the MTP blocks, which these
	// explicitly flag rather than leaving dense as every other architecture
	// does.
	allBlocks bool

	// patternRequiresPositiveKey, when set, withholds the pattern default
	// unless that GGUF key is present and non-zero. dflash is two models under
	// one architecture string: with hyper_connection.count it is a DSV4 draft
	// backbone where every block is sliding-window, and without it a model that
	// takes its interleave solely from an explicit per-block array — llama.cpp
	// leaves the flags zeroed there, so no default may stand in.
	patternRequiresPositiveKey string
}

// swaArchitectures mirrors the set_swa_pattern calls in llama.cpp's
// src/models/*.cpp — the 18 that go through load_swa_pattern, plus the two that
// call set_swa_pattern(0) directly — keyed by the GGUF architecture string from
// src/llama-arch.cpp.
//
// An architecture missing here is treated as having no sliding-window
// attention, which leaves its estimate exactly as it was: full context for
// every block. That is the safe direction for an interleaved model, where it
// merely forgoes the correction, but it overstates an all-SWA architecture
// substantially — which is why the two below are listed rather than left out.
//
// phi3 deliberately has no entry: it reads attention.sliding_window and then
// sets swa_type = NONE regardless, because Phi SWA is disabled upstream
// (ggml-org/llama.cpp#13676). Sizing it at full context is correct.
//
// Checked against llama.cpp b11351, the tag in LLAMA_CPP_VERSION at the time.
// This is duplicated knowledge and can drift when that tag moves, so the
// recheck is one line:
//
//	grep -rn "swa_pattern" src/models/*.cpp
//
// Every call wants an entry here, with the architecture string taken from
// src/llama-arch.cpp rather than from the file name — they differ
// (openai-moe.cpp is "gpt-oss"). A call under a condition needs that condition
// too, which is what allBlocks, forcedWindow, defaultWindow,
// patternRequiresPositiveKey and requiredBlocks are for.
var swaArchitectures = map[string]swaInterleave{
	"gemma2":          {pattern: 2, defaultWindow: 4096},
	"gemma3":          {pattern: 6},
	"gemma3n":         {pattern: 5},
	"gemma-embedding": {pattern: 6},
	"cohere2":         {pattern: 4},
	"cohere2moe":      {pattern: 4, denseFirst: true},
	"afmoe":           {pattern: 4},
	"olmo2":           {pattern: 4},
	"mellum":          {pattern: 4},
	"plamo3":          {pattern: 8},
	"exaone4":         {pattern: 4, defaultWindow: 4096, requiredBlocks: 64},
	"exaone-moe":      {pattern: 4, defaultWindow: 128},
	"gpt-oss":         {pattern: 2},
	"muse-glimmer":    {pattern: 4},
	"llama4":          {pattern: 4, defaultWindow: 8192, forcedWindow: 8192},
	"smallthinker":    {pattern: 4, denseFirst: true, forcedWindow: 4096},
	"modern-bert":     {pattern: 3, denseFirst: true},
	"laguna":          {pattern: 4, denseFirst: true},

	// set_swa_pattern(0) rather than load_swa_pattern: every block, MTP
	// included. Both require attention.sliding_window in the file — deepseek4
	// reads it unconditionally and dflash asserts it is positive — so neither
	// carries a defaultWindow.
	"deepseek4": {pattern: 0, allBlocks: true},
	"dflash":    {pattern: 0, allBlocks: true, patternRequiresPositiveKey: "hyper_connection.count"},
}

// attentionBlockCount returns the number of blocks that take part in attention,
// which is llama.cpp's n_layer(): the block count less the MTP prediction
// blocks appended after them.
//
// set_swa_pattern leaves those trailing blocks dense for every architecture
// that calls it, so they hold a full-context cache — except in the two that
// flag them sliding-window by hand afterwards, which swaInterleave.allBlocks
// marks.
func (m *Metadata) attentionBlockCount() uint64 {
	blocks := m.BlockCount()
	if nextn := m.Uint("nextn_predict_layers"); nextn < blocks {
		return blocks - nextn
	}
	return blocks
}

// SlidingWindow returns the effective sliding-window width, or zero if the
// model has no sliding-window attention.
//
// The precedence follows llama.cpp: an explicit attention.sliding_window of
// zero disables SWA outright, a forced width in the architecture table wins
// over the metadata, and a missing key falls back to the architecture's default
// — which is itself zero for the architectures that require the key.
func (m *Metadata) SlidingWindow() uint64 {
	arch, ok := swaArchitectures[m.Architecture()]
	if !ok {
		return 0
	}
	if arch.requiredBlocks != 0 && m.attentionBlockCount() != arch.requiredBlocks {
		return 0
	}

	window, declared := m.UintOK("attention.sliding_window")
	if declared && window == 0 {
		return 0
	}
	if !declared {
		return arch.defaultWindow
	}
	if arch.forcedWindow != 0 {
		return arch.forcedWindow
	}
	return window
}

// SWALayers reports which blocks use sliding-window attention, and so hold the
// smaller, capped KV cache rather than a full-context one.
//
// Getting this wrong is not a rounding error in the other direction either:
// gemma3 caps 40 of its 48 blocks at 1536 cells where the context is 4096, so
// treating them as full-context overstates the cache 2.09-fold at a 4096
// context and 5.67-fold at 131072.
//
// Sources in llama.cpp's order of precedence (src/llama-model.cpp,
// load_swa_pattern):
//
//  1. attention.sliding_window_pattern as a per-block flag array
//  2. attention.sliding_window_pattern as a scalar interleave
//  3. the architecture's hardcoded default, from swaArchitectures
func (m *Metadata) SWALayers() []bool {
	out := make([]bool, m.BlockCount())
	if m.SlidingWindow() == 0 {
		return out
	}

	if explicit := m.uintArray("attention.sliding_window_pattern", 0); len(explicit) > 1 {
		for i := range out {
			out[i] = i < len(explicit) && explicit[i] != 0
		}
		return out
	}

	arch := swaArchitectures[m.Architecture()]

	// dflash's non-draft branch reads only the per-block array, never a scalar
	// and never a default, so llama.cpp's zeroed flags leave every block dense
	// even though swa_type is STANDARD. Without the array handled above, that
	// is where we stop.
	if key := arch.patternRequiresPositiveKey; key != "" && m.Uint(key) == 0 {
		return out
	}

	// Blocks from n_layer() on are MTP prediction blocks, left out of the
	// pattern except where the architecture flags them by hand.
	attention := m.attentionBlockCount()
	if arch.allBlocks {
		attention = m.BlockCount()
	}

	pattern := m.Uint("attention.sliding_window_pattern", arch.pattern)
	if pattern == 0 {
		// llama.cpp treats a zero pattern as every block being SWA.
		for i := range out {
			out[i] = uint64(i) < attention
		}
		return out
	}

	for i := range out {
		switch {
		case uint64(i) >= attention:
			out[i] = false
		case arch.denseFirst:
			out[i] = uint64(i)%pattern != 0
		default:
			out[i] = uint64(i)%pattern < pattern-1
		}
	}
	return out
}

// swaCells returns the per-stream cell count of the sliding-window cache,
// mirroring llama_kv_cache_iswa: the window plus one micro-batch, never more
// than the context, padded to 256.
//
// The padding and the micro-batch headroom are both real allocations, not slack
// — llama.cpp pads the SWA cache to 256 for performance (ggml-org/llama.cpp
// issue 17037) and needs a micro-batch of room beyond the window to write a
// batch before evicting from it.
//
// The window itself does not scale with the slot count: llama.cpp widens it by
// n_seq_max only for a unified cache, and ollama never asks for one. Each slot
// instead gets its own stream of this many cells.
func swaCells(contextPerSeq, window, numBatch uint64) uint64 {
	cells := window + numBatch
	if cells > contextPerSeq {
		cells = contextPerSeq
	}
	return pad256(cells)
}

// pad256 rounds up to a multiple of 256, as GGML_PAD does.
func pad256(n uint64) uint64 {
	return (n + 255) & ^uint64(255)
}

// KVCacheSize returns the bytes the KV cache occupies for the given context
// size, counting only the blocks that actually hold one, capping the blocks
// that use sliding-window attention, and using the real per-element size of the
// configured cache type.
//
// context is the per-sequence context; numParallel slots multiply it. The two
// cannot be collapsed into one figure, because a sliding-window cache is capped
// per stream and only then multiplied by the slot count.
//
// numBatch is llama-server's micro-batch (-ub), which adds headroom to every
// sliding-window cache. Zero falls back to llama.cpp's own default of 512.
//
// Verified against llama.cpp's own reporting, to the MiB:
//
//	qwen35 (65 blocks, interval 4 -> 16), 131072 ctx, q8_0 -> 4352 MiB
//	qwen35 (65 blocks, interval 4 -> 16),   8192 ctx, q8_0 ->  272 MiB
//	gemma3 (48 blocks, 8 full + 40 SWA),     4096 ctx, q8_0 ->  391 MiB
//	       (136.00 MiB over 4096 cells + 255.00 MiB over 1536)
//
// Models with MTP speculative decoding build a second, draft context whose KV
// cache is a separate allocation on top of this one; see DraftKVCacheSize.
// Compute, recurrent-state and output buffers are deliberately not included —
// they depend on batch size and graph topology and cannot be derived honestly
// from metadata alone.
func (m *Metadata) KVCacheSize(context uint64, numParallel int, numBatch uint64, kvCacheType string) uint64 {
	if numBatch == 0 {
		numBatch = 512 // llama.cpp's default n_ubatch
	}

	headsKV := m.HeadCountKV()
	embeddingHeadsK := m.EmbeddingHeadCountK()
	embeddingHeadsV := m.EmbeddingHeadCountV()
	bytesPerElement := kvCacheBytesPerElement(kvCacheType)

	// llama.cpp pads the per-sequence context to 256 before sizing anything
	// (llama-context.cpp, the non-unified n_ctx_seq branch).
	slots := uint64(max(numParallel, 1))
	context = pad256(context)

	// A full-attention block holds the whole per-sequence context; a
	// sliding-window block holds only its capped window. Each slot gets its own
	// stream of either, since ollama never passes --kv-unified.
	fullCells := context * slots
	windowCells := fullCells
	if window := m.SlidingWindow(); window > 0 {
		windowCells = swaCells(context, window, numBatch) * slots
	}

	swaLayers := m.SWALayers()

	var total uint64
	for i, isAttention := range m.AttentionLayers() {
		if !isAttention || i >= len(headsKV) {
			continue
		}
		cells := fullCells
		if i < len(swaLayers) && swaLayers[i] {
			cells = windowCells
		}
		total += uint64(float64(cells*(embeddingHeadsK+embeddingHeadsV)*headsKV[i]) * bytesPerElement)
	}
	return total
}

// DraftKVCacheSize returns the bytes used by the KV cache of the MTP draft
// context, or zero for models without one.
//
// Models carrying next-token-prediction blocks (qwen35 and relatives) construct
// a second llama_context for speculative decoding. Its blocks are dense
// attention and always cached at f16 regardless of the configured cache type.
//
// Verified: qwen35 with nextn_predict_layers=1 at 131072 context reports
// exactly 512 MiB for the draft context.
func (m *Metadata) DraftKVCacheSize(context uint64, numParallel int) uint64 {
	nextn := m.Uint("nextn_predict_layers")
	if nextn == 0 {
		return 0
	}
	if numParallel > 1 {
		context *= uint64(numParallel)
	}

	embeddingHeads := m.EmbeddingHeadCountK() + m.EmbeddingHeadCountV()
	return nextn * context * embeddingHeads * m.HeadCountKVMax() * 2
}

// kvCacheBytesPerElement returns the number of bytes per element for a given KV
// cache type, derived from the ggml block layout rather than rounded to a power
// of two. A q8_0 block is 34 bytes (one f16 scale plus 32 int8) for 32 elements
// — 1.0625 per element, not 1. The 6% matters: at 131072 context it is the
// difference between 4096 and 4352 MiB, and llama.cpp reports the latter.
//
// OLLAMA_KV_CACHE_TYPE is passed through to llama-server unvalidated, so any
// ggml quantization type can arrive here, not just the documented f16/q8_0/q4_0.
func kvCacheBytesPerElement(cacheType string) float64 {
	switch cacheType {
	case "q8_0":
		return 34.0 / 32.0 // f16 scale + 32 int8
	case "q4_0":
		return 18.0 / 32.0 // f16 scale + 32 nibbles
	case "q4_1":
		return 20.0 / 32.0 // f16 scale + f16 min + 32 nibbles
	case "q5_0":
		return 22.0 / 32.0 // f16 scale + 32 5-bit
	case "q5_1":
		return 24.0 / 32.0 // f16 scale + f16 min + 32 5-bit
	case "iq4_nl":
		return 18.0 / 32.0 // f16 scale + 32 nibbles into a lookup table
	case "f32":
		return 4 // f32 (default for recurrent)
	default:
		return 2 // f16 (default)
	}
}
