package mlxrunner

import (
	"testing"

	"github.com/ollama/ollama/manifest"
)

func testManifest(tensorSizes ...int64) *manifest.Manifest {
	m := &manifest.Manifest{}
	for _, size := range tensorSizes {
		m.Layers = append(m.Layers, manifest.Layer{MediaType: manifest.MediaTypeImageTensor, Size: size})
	}
	m.Layers = append(m.Layers, manifest.Layer{MediaType: "application/vnd.ollama.image.json", Name: "config.json", Size: 1})
	return m
}

// qwen3_5 27B: 64 layers, every fourth one full attention, the rest gated
// delta-net, config nested under text_config as the multimodal checkpoint
// ships it.
const qwen3_5Config = `{
	"architectures": ["Qwen3_5ForConditionalGeneration"],
	"model_type": "qwen3_5",
	"text_config": {
		"dtype": "bfloat16",
		"hidden_size": 5120,
		"num_hidden_layers": 64,
		"num_attention_heads": 24,
		"num_key_value_heads": 4,
		"head_dim": 256,
		"max_position_embeddings": 262144,
		"full_attention_interval": 4,
		"linear_num_value_heads": 48,
		"linear_num_key_heads": 16,
		"linear_key_head_dim": 128,
		"linear_value_head_dim": 128,
		"linear_conv_kernel_dim": 4
	},
	"vision_config": {"num_hidden_layers": 27, "hidden_size": 1152}
}`

func TestPredictMemoryHybrid(t *testing.T) {
	m := testManifest(10<<30, 6<<30)
	got, err := predictMemory(m, []byte(qwen3_5Config), 262144)
	if err != nil {
		t.Fatal(err)
	}

	// 16 full-attention layers, each 2 (K and V) * 4 heads * 256 * 2 bytes
	// = 4 KiB per token, so 64 KiB per token over 262144 tokens.
	const kv = 16 * 2 * 4 * 256 * 2 * 262144

	// 48 recurrent layers, each a bf16 conv tail of 3 x (2*16*128 +
	// 48*128) plus a float32 state of 48 x 128 x 128.
	const recurrent = 48 * (3*(2*16*128+48*128)*2 + 48*128*128*4)

	want := MemoryEstimate{
		Weights:     16 << 30,
		KVCache:     kv,
		Recurrent:   recurrent,
		PrefixCache: 8 << 30,
		NumCtx:      262144,
	}
	if got != want {
		t.Errorf("predictMemory = %+v, want %+v", got, want)
	}
	if got.Total() != want.Weights+kv+recurrent+8<<30 {
		t.Errorf("Total = %d", got.Total())
	}
}

func TestPredictMemoryLayerTypesWin(t *testing.T) {
	// An explicit layer_types list overrides the interval, here making
	// two of four layers full attention instead of one.
	config := `{
		"num_hidden_layers": 4,
		"num_attention_heads": 8,
		"num_key_value_heads": 2,
		"head_dim": 64,
		"full_attention_interval": 4,
		"layer_types": ["linear_attention", "full_attention", "linear_attention", "full_attention"],
		"linear_conv_kernel_dim": 4,
		"linear_num_key_heads": 1,
		"linear_key_head_dim": 8,
		"linear_num_value_heads": 1,
		"linear_value_head_dim": 8
	}`
	got, err := predictMemory(testManifest(1), []byte(config), 1024)
	if err != nil {
		t.Fatal(err)
	}
	if want := uint64(2 * 2 * 2 * 64 * 2 * 1024); got.KVCache != want {
		t.Errorf("KVCache = %d, want %d", got.KVCache, want)
	}
	if want := uint64(2 * (3*(2*8+8)*2 + 8*8*4)); got.Recurrent != want {
		t.Errorf("Recurrent = %d, want %d", got.Recurrent, want)
	}
}

func TestPredictMemoryContextCappedAtModel(t *testing.T) {
	config := `{
		"num_hidden_layers": 2,
		"num_attention_heads": 4,
		"num_key_value_heads": 4,
		"hidden_size": 256,
		"max_position_embeddings": 4096
	}`
	for _, tt := range []struct {
		name   string
		numCtx int
		want   int
	}{
		{"below the model's length", 1024, 1024},
		{"above the model's length", 262144, 4096},
		{"unset", 0, 4096},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, err := predictMemory(testManifest(1), []byte(config), tt.numCtx)
			if err != nil {
				t.Fatal(err)
			}
			if got.NumCtx != tt.want {
				t.Errorf("NumCtx = %d, want %d", got.NumCtx, tt.want)
			}
			// head_dim defaults to hidden_size / heads = 64.
			if want := uint64(2 * 2 * 4 * 64 * 2 * tt.want); got.KVCache != want {
				t.Errorf("KVCache = %d, want %d", got.KVCache, want)
			}
		})
	}
}

func TestPredictMemorySlidingWindowAndSharedKV(t *testing.T) {
	// Gemma 4 style: sliding layers hold the window, full layers use
	// global_head_dim, and the last num_kv_shared_layers allocate nothing.
	config := `{
		"text_config": {
			"num_hidden_layers": 6,
			"num_attention_heads": 8,
			"num_key_value_heads": 2,
			"head_dim": 256,
			"global_head_dim": 512,
			"sliding_window": 512,
			"num_kv_shared_layers": 2,
			"layer_types": ["sliding_attention", "sliding_attention", "full_attention",
				"sliding_attention", "sliding_attention", "full_attention"]
		}
	}`
	got, err := predictMemory(testManifest(1), []byte(config), 8192)
	if err != nil {
		t.Fatal(err)
	}
	sliding := uint64(2 * 2 * 256 * 2 * 512)
	full := uint64(2 * 2 * 512 * 2 * 8192)
	if want := 3*sliding + full; got.KVCache != want {
		t.Errorf("KVCache = %d, want %d", got.KVCache, want)
	}
}

func TestPredictMemorySlidingWindowPattern(t *testing.T) {
	// Without layer_types, Gemma 4 makes every fifth layer full attention.
	config := `{
		"num_hidden_layers": 5,
		"num_attention_heads": 8,
		"num_key_value_heads": 1,
		"head_dim": 256,
		"sliding_window": 1024,
		"sliding_window_pattern": 5
	}`
	got, err := predictMemory(testManifest(1), []byte(config), 4096)
	if err != nil {
		t.Fatal(err)
	}
	sliding := uint64(2 * 1 * 256 * 2 * 1024)
	full := uint64(2 * 1 * 256 * 2 * 4096)
	if want := 4*sliding + full; got.KVCache != want {
		t.Errorf("KVCache = %d, want %d", got.KVCache, want)
	}
}

func TestPredictMemoryHybridOverridePattern(t *testing.T) {
	// Nemotron-H: M is Mamba, * is attention, - is an MLP block.
	config := `{
		"llm_config": {
			"num_hidden_layers": 6,
			"hybrid_override_pattern": "M*M-M*",
			"num_attention_heads": 8,
			"num_key_value_heads": 2,
			"head_dim": 128,
			"conv_kernel": 4,
			"ssm_state_size": 128,
			"mamba_num_heads": 64,
			"mamba_head_dim": 64,
			"n_groups": 8
		}
	}`
	got, err := predictMemory(testManifest(1), []byte(config), 2048)
	if err != nil {
		t.Fatal(err)
	}
	if want := uint64(2 * 2 * 2 * 128 * 2 * 2048); got.KVCache != want {
		t.Errorf("KVCache = %d, want %d", got.KVCache, want)
	}
	convDim := 64*64 + 2*8*128
	if want := uint64(3 * (3*convDim*2 + 64*64*128*4)); got.Recurrent != want {
		t.Errorf("Recurrent = %d, want %d", got.Recurrent, want)
	}
}

func TestPredictMemoryMLA(t *testing.T) {
	config := `{
		"num_hidden_layers": 2,
		"num_attention_heads": 16,
		"kv_lora_rank": 512,
		"qk_rope_head_dim": 64,
		"torch_dtype": "bfloat16"
	}`
	got, err := predictMemory(testManifest(1), []byte(config), 1000)
	if err != nil {
		t.Fatal(err)
	}
	if want := uint64(2 * (512 + 64) * 2 * 1000); got.KVCache != want {
		t.Errorf("KVCache = %d, want %d", got.KVCache, want)
	}
}

func TestPredictMemoryFloat32(t *testing.T) {
	config := `{
		"num_hidden_layers": 1,
		"num_attention_heads": 1,
		"head_dim": 8,
		"torch_dtype": "float32"
	}`
	got, err := predictMemory(testManifest(1), []byte(config), 10)
	if err != nil {
		t.Fatal(err)
	}
	if want := uint64(2 * 1 * 8 * 4 * 10); got.KVCache != want {
		t.Errorf("KVCache = %d, want %d", got.KVCache, want)
	}
}

func TestPredictMemoryWithoutCaches(t *testing.T) {
	// A config without layers describes no cache, so there is nothing to
	// page out either: the estimate is the weights.
	got, err := predictMemory(testManifest(100, 200), []byte(`{"model_type": "other"}`), 4096)
	if err != nil {
		t.Fatal(err)
	}
	if want := (MemoryEstimate{Weights: 300, NumCtx: 4096}); got != want {
		t.Errorf("predictMemory = %+v, want %+v", got, want)
	}
}

func TestPredictMemoryBadConfig(t *testing.T) {
	if _, err := predictMemory(testManifest(1), []byte(`{"num_hidden_layers": "many"}`), 10); err == nil {
		t.Error("expected an error for a malformed config")
	}
}
