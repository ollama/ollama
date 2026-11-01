package gguf_test

import (
	"testing"

	"github.com/ollama/ollama/fs/gguf"
	gguftest "github.com/ollama/ollama/internal/testutil/gguf"
)

const MiB = 1024 * 1024

// readKV writes kv to a GGUF fixture and returns its metadata.
func readKV(t *testing.T, kv gguftest.KV) *gguf.Metadata {
	t.Helper()
	metadata, err := gguf.ReadFileMetadata(writeMetadataFixture(t, kv, nil), -1)
	if err != nil {
		t.Fatal(err)
	}
	return metadata
}

// qwen35KV mirrors the metadata of smtek/Swift-Qwen3.8-27B as reported by
// llama_model_loader: 65 blocks, of which every fourth holds a KV cache.
func qwen35KV() gguftest.KV {
	return gguftest.KV{
		"general.architecture":           "qwen35",
		"qwen35.block_count":             uint32(65),
		"qwen35.full_attention_interval": uint32(4),
		"qwen35.attention.head_count":    uint32(24),
		"qwen35.attention.head_count_kv": uint32(4),
		"qwen35.attention.key_length":    uint32(256),
		"qwen35.attention.value_length":  uint32(256),
		"qwen35.embedding_length":        uint32(5120),
	}
}

// granitehybridKV mirrors a granitehybrid model: 32 blocks, of which only the
// four with a non-zero head_count_kv hold a KV cache.
func granitehybridKV() gguftest.KV {
	heads := make([]int32, 32)
	for _, i := range []int{10, 17, 24, 31} {
		heads[i] = 4
	}
	return gguftest.KV{
		"general.architecture":                  "granitehybrid",
		"granitehybrid.block_count":             uint32(32),
		"granitehybrid.attention.head_count":    uint32(12),
		"granitehybrid.attention.head_count_kv": heads,
		"granitehybrid.embedding_length":        uint32(768),
	}
}

// gemma3KV mirrors the metadata of gemma3:12b as read from the real blob: 48
// blocks, a 1024-wide sliding window, and no sliding_window_pattern — so
// llama.cpp falls back to its hardcoded interleave of 6, giving 8 full-attention
// blocks and 40 sliding-window ones.
func gemma3KV() gguftest.KV {
	return gguftest.KV{
		"general.architecture":            "gemma3",
		"gemma3.block_count":              uint32(48),
		"gemma3.attention.head_count":     uint32(16),
		"gemma3.attention.head_count_kv":  uint32(8),
		"gemma3.attention.key_length":     uint32(256),
		"gemma3.attention.value_length":   uint32(256),
		"gemma3.attention.sliding_window": uint32(1024),
		"gemma3.embedding_length":         uint32(3840),
	}
}

func TestSlidingWindow(t *testing.T) {
	cases := []struct {
		name string
		kv   gguftest.KV
		want uint64
	}{
		{"gemma3 takes the declared window", gemma3KV(), 1024},
		{
			// An architecture with no sliding-window attention at all.
			name: "qwen35 has no window",
			kv:   qwen35KV(),
			want: 0,
		},
		{
			// llama4 pins the window in code regardless of the metadata.
			name: "llama4 forces its own window",
			kv: gguftest.KV{
				"general.architecture":            "llama4",
				"llama4.block_count":              uint32(48),
				"llama4.attention.sliding_window": uint32(1024),
			},
			want: 8192,
		},
		{
			// An explicit zero disables SWA, which is how llama4 ships its
			// no-SWA variants.
			name: "explicit zero disables the window",
			kv: gguftest.KV{
				"general.architecture":            "llama4",
				"llama4.block_count":              uint32(48),
				"llama4.attention.sliding_window": uint32(0),
			},
			want: 0,
		},
		{
			// gemma2 has no key in the file but a default in llama.cpp.
			name: "gemma2 falls back to its default",
			kv: gguftest.KV{
				"general.architecture": "gemma2",
				"gemma2.block_count":   uint32(42),
			},
			want: 4096,
		},
		{
			// exaone4 enables SWA only for the 64-block 32B.
			name: "exaone4 at 30 blocks has no window",
			kv: gguftest.KV{
				"general.architecture": "exaone4",
				"exaone4.block_count":  uint32(30),
			},
			want: 0,
		},
		{
			name: "exaone4 at 64 blocks does",
			kv: gguftest.KV{
				"general.architecture": "exaone4",
				"exaone4.block_count":  uint32(64),
			},
			want: 4096,
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			if got := readKV(t, tt.kv).SlidingWindow(); got != tt.want {
				t.Errorf("sliding window: got=%d want=%d", got, tt.want)
			}
		})
	}
}

func TestSWALayers(t *testing.T) {
	cases := []struct {
		name string
		kv   gguftest.KV
		want int
	}{
		{
			// 48 blocks at interval 6 without dense_first: SWA where i%6 < 5,
			// so every sixth block is full attention. llama.cpp reports exactly
			// "4096 cells, 8 layers" and "1536 cells, 40 layers" for this model.
			name: "gemma3 interleave of 6 over 48 blocks",
			kv:   gemma3KV(),
			want: 40,
		},
		{
			// dense_first inverts which end of the group is full attention:
			// SWA where i%4 != 0, so 3 of every 4.
			name: "smallthinker dense_first inverts the group",
			kv: gguftest.KV{
				"general.architecture":                  "smallthinker",
				"smallthinker.block_count":              uint32(32),
				"smallthinker.attention.sliding_window": uint32(4096),
			},
			want: 24,
		},
		{
			// An explicit per-block array outranks the hardcoded interleave.
			name: "explicit pattern array wins",
			kv: gguftest.KV{
				"general.architecture":                    "gemma3",
				"gemma3.block_count":                      uint32(4),
				"gemma3.attention.sliding_window":         uint32(1024),
				"gemma3.attention.sliding_window_pattern": []int32{1, 0, 1, 1},
			},
			want: 3,
		},
		{
			// A model without sliding-window attention caps nothing.
			name: "qwen35 has no sliding-window blocks",
			kv:   qwen35KV(),
			want: 0,
		},
		{
			// set_swa_pattern(0) makes every block sliding-window, and deepseek4
			// then flags its MTP blocks by hand — so all 8, not 7.
			name: "deepseek4 caps every block including MTP",
			kv: gguftest.KV{
				"general.architecture":               "deepseek4",
				"deepseek4.block_count":              uint32(8),
				"deepseek4.nextn_predict_layers":     uint32(1),
				"deepseek4.attention.head_count_kv":  uint32(4),
				"deepseek4.attention.sliding_window": uint32(2048),
			},
			want: 8,
		},
		{
			// The DSV4 draft backbone, which hyper_connection.count selects:
			// every block sliding-window, MTP included.
			name: "dflash draft backbone caps every block",
			kv: gguftest.KV{
				"general.architecture":            "dflash",
				"dflash.block_count":              uint32(8),
				"dflash.nextn_predict_layers":     uint32(1),
				"dflash.hyper_connection.count":   uint32(2),
				"dflash.attention.head_count_kv":  uint32(4),
				"dflash.attention.sliding_window": uint32(2048),
			},
			want: 8,
		},
		{
			// Without that key dflash takes its interleave solely from the
			// per-block array. Absent the array llama.cpp leaves every flag
			// zeroed despite swa_type being STANDARD, so nothing is capped — a
			// default here would understate the cache.
			name: "dflash without the draft key caps nothing",
			kv: gguftest.KV{
				"general.architecture":            "dflash",
				"dflash.block_count":              uint32(8),
				"dflash.attention.head_count_kv":  uint32(4),
				"dflash.attention.sliding_window": uint32(2048),
			},
			want: 0,
		},
		{
			// The array still applies on that branch, being read before the gate.
			name: "dflash without the draft key honours an explicit array",
			kv: gguftest.KV{
				"general.architecture":                    "dflash",
				"dflash.block_count":                      uint32(4),
				"dflash.attention.head_count_kv":          uint32(4),
				"dflash.attention.sliding_window":         uint32(2048),
				"dflash.attention.sliding_window_pattern": []int32{1, 1, 0, 1},
			},
			want: 3,
		},
		{
			// phi3 declares a window and then disables SWA outright
			// (ggml-org/llama.cpp#13676), so it is absent from the table on
			// purpose and sized at full context.
			name: "phi3 declares a window but has SWA disabled",
			kv: gguftest.KV{
				"general.architecture":          "phi3",
				"phi3.block_count":              uint32(32),
				"phi3.attention.head_count_kv":  uint32(8),
				"phi3.attention.sliding_window": uint32(2047),
			},
			want: 0,
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			got := 0
			for _, isSWA := range readKV(t, tt.kv).SWALayers() {
				if isSWA {
					got++
				}
			}
			if got != tt.want {
				t.Errorf("sliding-window layers: got=%d want=%d", got, tt.want)
			}
		})
	}
}

func TestAttentionLayers(t *testing.T) {
	cases := []struct {
		name string
		kv   gguftest.KV
		want int
	}{
		{"qwen35 interval of 4 over 65 blocks", qwen35KV(), 16},
		{"granitehybrid per-layer head_count_kv", granitehybridKV(), 4},
		{
			// An ordinary transformer declares none of the hybrid keys and
			// caches every block.
			name: "dense transformer caches every block",
			kv: gguftest.KV{
				"general.architecture":          "llama",
				"llama.block_count":             uint32(32),
				"llama.attention.head_count_kv": uint32(8),
			},
			want: 32,
		},
		{
			// An explicit recurrent_layers array outranks everything else.
			name: "explicit recurrent_layers wins over interval",
			kv: gguftest.KV{
				"general.architecture":              "qwen35",
				"qwen35.block_count":                uint32(4),
				"qwen35.full_attention_interval":    uint32(4),
				"qwen35.attention.recurrent_layers": []int32{1, 0, 1, 0},
			},
			want: 2,
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			got := 0
			for _, isAttention := range readKV(t, tt.kv).AttentionLayers() {
				if isAttention {
					got++
				}
			}
			if got != tt.want {
				t.Errorf("attention layers: got=%d want=%d", got, tt.want)
			}
		})
	}
}

// TestKVCacheSize checks the computed cache size against the figures llama.cpp
// prints for the same models, so a drift in either direction shows up here
// rather than as a misplaced layer at load time.
func TestKVCacheSize(t *testing.T) {
	cases := []struct {
		name        string
		kv          gguftest.KV
		context     uint64
		numParallel int
		numBatch    uint64
		cacheType   string
		want        uint64 // as llama.cpp reports it
	}{
		{
			// Measured on the real model, where llama.cpp logs both caches:
			//   llama_kv_cache: size = 136.00 MiB (4096 cells,  8 layers), q8_0
			//   llama_kv_cache: size = 255.00 MiB (1536 cells, 40 layers), q8_0
			// The SWA cache is GGML_PAD(min(4096, 1024+512), 256) = 1536 cells.
			name:      "gemma3 at 4096 with q8_0 caps its sliding-window blocks",
			kv:        gemma3KV(),
			context:   4096,
			numBatch:  512,
			cacheType: "q8_0",
			want:      391 * MiB,
		},
		{
			// Above the window the full blocks keep growing while the
			// sliding-window ones stay put, so the ratio widens with context.
			// Counting all 48 blocks at full context would give 26112 MiB.
			name:      "gemma3 at 131072 keeps the window capped",
			kv:        gemma3KV(),
			context:   131072,
			numBatch:  512,
			cacheType: "q8_0",
			want:      4607 * MiB,
		},
		{
			// Below the window there is nothing to cap: min() picks the
			// context, and the model sizes exactly as a dense one would.
			name:      "gemma3 at 1024 is not capped at all",
			kv:        gemma3KV(),
			context:   1024,
			numBatch:  512,
			cacheType: "q8_0",
			want:      204 * MiB,
		},
		{
			// A larger micro-batch widens every sliding-window cache, because
			// llama.cpp reserves a batch of room beyond the window:
			// GGML_PAD(min(4096, 1024+2048), 256) = 3072 cells.
			name:      "gemma3 sliding-window cache follows the micro-batch",
			kv:        gemma3KV(),
			context:   4096,
			numBatch:  2048,
			cacheType: "q8_0",
			want:      646 * MiB,
		},
		{
			// Slots do not widen the window — llama.cpp scales it by n_seq_max
			// only for a unified cache, which ollama never requests. Each slot
			// gets its own stream of the same 1536 cells, so this is exactly
			// twice the single-slot figure rather than the 1122 MiB a widened
			// window would give.
			name:        "gemma3 with two slots replicates rather than widens",
			kv:          gemma3KV(),
			context:     4096,
			numParallel: 2,
			numBatch:    512,
			cacheType:   "q8_0",
			want:        782 * MiB,
		},
		{
			// llama_kv_cache: size = 4352.00 MiB (131072 cells, 16 layers), K/V q8_0
			name:      "qwen35 at 131072 with q8_0",
			kv:        qwen35KV(),
			context:   131072,
			cacheType: "q8_0",
			want:      4352 * MiB,
		},
		{
			// llama_kv_cache: size = 272.00 MiB (8192 cells, 16 layers), K/V q8_0
			name:      "qwen35 at 8192 with q8_0",
			kv:        qwen35KV(),
			context:   8192,
			cacheType: "q8_0",
			want:      272 * MiB,
		},
		{
			// llama_kv_cache: size = 8.50 MiB (4096 cells, 4 layers), K/V q8_0
			name:      "granitehybrid at 4096 with q8_0",
			kv:        granitehybridKV(),
			context:   4096,
			cacheType: "q8_0",
			want:      8.5 * MiB,
		},
		{
			// f16 is twice q8_0's 1.0625 bytes per element, not twice 1.
			name:      "qwen35 at 8192 with f16",
			kv:        qwen35KV(),
			context:   8192,
			cacheType: "f16",
			want:      512 * MiB,
		},
		{
			// Parallel slots each get their own cells.
			name:        "qwen35 at 8192 with two parallel slots",
			kv:          qwen35KV(),
			context:     8192,
			numParallel: 2,
			cacheType:   "q8_0",
			want:        544 * MiB,
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			got := readKV(t, tt.kv).KVCacheSize(tt.context, tt.numParallel, tt.numBatch, tt.cacheType)
			if got != tt.want {
				t.Errorf("KV cache size: got=%.2f MiB want=%.2f MiB",
					float64(got)/MiB, float64(tt.want)/MiB)
			}
		})
	}
}

// TestDraftKVCacheSize covers the second context that MTP models build. Its
// buffers are a separate allocation, which is exactly what the old accounting
// overwrote.
func TestDraftKVCacheSize(t *testing.T) {
	withNextN := qwen35KV()
	withNextN["qwen35.nextn_predict_layers"] = uint32(1)

	// llama_kv_cache: size = 512.00 MiB (131072 cells, 1 layers), K/V f16
	if got, want := readKV(t, withNextN).DraftKVCacheSize(131072, 1), uint64(512*MiB); got != want {
		t.Errorf("draft KV cache: got=%.2f MiB want=%.2f MiB",
			float64(got)/MiB, float64(want)/MiB)
	}

	// A model without MTP blocks builds no draft context at all.
	if got := readKV(t, qwen35KV()).DraftKVCacheSize(131072, 1); got != 0 {
		t.Errorf("draft KV cache without nextn blocks: got=%d want=0", got)
	}
}

// TestKVCacheType covers every cache type OLLAMA_KV_CACHE_TYPE can select,
// which the old code replaced with a flat `* 2` for f16.
//
// The per-element sizes are the ggml block layouts rather than powers of two: a
// quantized block stores its scale alongside the values, so q8_0 costs 1.0625
// bytes per element and not 1. qwen35 at 8192 is the yardstick because two of
// these rows are anchored against llama.cpp's own output in TestKVCacheSize —
// 272 MiB at q8_0 and 512 at f16 — so the rest are that same cache scaled by
// the layout, 256 MiB per byte per element.
func TestKVCacheType(t *testing.T) {
	cases := []struct {
		cacheType string
		want      uint64
		layout    string
	}{
		{"f16", 512 * MiB, "2 bytes: the default, and what the old code assumed for every type"},
		{"f32", 1024 * MiB, "4 bytes: the default for recurrent caches"},
		{"q8_0", 272 * MiB, "34/32 bytes: f16 scale + 32 int8"},
		{"q4_0", 144 * MiB, "18/32 bytes: f16 scale + 32 nibbles"},
		{"q4_1", 160 * MiB, "20/32 bytes: f16 scale + f16 min + 32 nibbles"},
		{"q5_0", 176 * MiB, "22/32 bytes: f16 scale + 32 5-bit"},
		{"q5_1", 192 * MiB, "24/32 bytes: f16 scale + f16 min + 32 5-bit"},
		{"iq4_nl", 144 * MiB, "18/32 bytes: f16 scale + 32 nibbles into a lookup table"},

		// OLLAMA_KV_CACHE_TYPE reaches llama-server unvalidated, so an
		// unrecognized value has to land somewhere. f16 is llama.cpp's default.
		{"", 512 * MiB, "unset falls back to f16"},
		{"nonsense", 512 * MiB, "an unknown type falls back to f16"},
	}

	for _, tt := range cases {
		name := tt.cacheType
		if name == "" {
			name = "unset"
		}
		t.Run(name, func(t *testing.T) {
			got := readKV(t, qwen35KV()).KVCacheSize(8192, 1, 0, tt.cacheType)
			if got != tt.want {
				t.Errorf("%s (%s): got=%.2f MiB want=%.2f MiB",
					tt.cacheType, tt.layout, float64(got)/MiB, float64(tt.want)/MiB)
			}
		})
	}
}
