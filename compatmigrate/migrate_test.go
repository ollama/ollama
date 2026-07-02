package compatmigrate

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"io"
	"math"
	"os"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/x448/float16"

	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/fs/gguf"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

func TestEnsureLocalCompatibilityMigrationAppendsToExistingManifestList(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	registerTestCompatMigrator(t)

	source := model.ParseName("registry.ollama.ai/library/testcompat:latest")
	writeSourceManifest(t, source, sourceManifestInput{
		config: model.ConfigV2{
			ModelFormat:   "gguf",
			ModelFamily:   "testcompat",
			ModelFamilies: []string{"testcompat"},
			Architecture:  "amd64",
			OS:            "linux",
			RootFS:        model.RootFS{Type: "layers"},
		},
		modelKV: ggml.KV{
			"general.architecture":  "testcompat",
			"tokenizer.ggml.tokens": []string{"x"},
		},
		modelTensors: []*ggml.Tensor{
			fixtureTensor("token_embd.weight", ggml.TensorTypeF16, []uint64{1, 8}),
		},
	})
	wrapSourceManifestAsList(t, source)

	migrated, err := EnsureLocalCompatibilityMigration(source)
	if err != nil {
		t.Fatalf("EnsureLocalCompatibilityMigration() error = %v", err)
	}
	if !migrated {
		t.Fatal("expected migration to append a llamacpp child")
	}

	raw, err := manifest.ReadManifestData(source)
	if err != nil {
		t.Fatalf("ReadManifestData(source) error = %v", err)
	}
	var parent manifest.Manifest
	if err := json.Unmarshal(raw, &parent); err != nil {
		t.Fatalf("unmarshal parent manifest: %v", err)
	}
	if parent.MediaType != manifest.MediaTypeManifestList {
		t.Fatalf("expected manifest list, got %q", parent.MediaType)
	}
	if len(parent.Manifests) != 2 {
		t.Fatalf("expected two child manifests, got %d", len(parent.Manifests))
	}
	if _, err := manifest.ParseNamedManifestForRunner(source, manifest.RunnerGGML); err != nil {
		t.Fatalf("expected ggml child to resolve: %v", err)
	}
	if _, err := manifest.ParseNamedManifestForRunner(source, manifest.RunnerLlamaCPP); err != nil {
		t.Fatalf("expected llamacpp child to resolve: %v", err)
	}

	migratedAgain, err := EnsureLocalCompatibilityMigration(source)
	if err != nil {
		t.Fatalf("EnsureLocalCompatibilityMigration(second) error = %v", err)
	}
	if !migratedAgain {
		t.Fatal("expected existing llamacpp child to satisfy migration")
	}
	rawAgain, err := manifest.ReadManifestData(source)
	if err != nil {
		t.Fatalf("ReadManifestData(second) error = %v", err)
	}
	if !bytes.Equal(raw, rawAgain) {
		t.Fatal("expected second migration attempt to leave manifest list unchanged")
	}
}

func TestEnsureLocalCompatibilityMigrationUnsupportedFamilyNoop(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	source := model.ParseName("registry.ollama.ai/library/notcompat:latest")
	writeSourceManifest(t, source, sourceManifestInput{
		config: model.ConfigV2{
			ModelFormat:  "gguf",
			ModelFamily:  "notcompat",
			Architecture: "amd64",
			OS:           "linux",
			RootFS:       model.RootFS{Type: "layers"},
		},
		modelKV: ggml.KV{
			"general.architecture":  "notcompat",
			"tokenizer.ggml.tokens": []string{"x"},
		},
		modelTensors: []*ggml.Tensor{
			fixtureTensor("token_embd.weight", ggml.TensorTypeF16, []uint64{1, 8}),
		},
	})

	migrated, err := EnsureLocalCompatibilityMigration(source)
	if err != nil {
		t.Fatalf("EnsureLocalCompatibilityMigration() error = %v", err)
	}
	if migrated {
		t.Fatal("expected unsupported family to skip migration")
	}

	raw, err := manifest.ReadManifestData(source)
	if err != nil {
		t.Fatalf("ReadManifestData(source) error = %v", err)
	}
	var stored manifest.Manifest
	if err := json.Unmarshal(raw, &stored); err != nil {
		t.Fatalf("unmarshal stored manifest: %v", err)
	}
	if stored.MediaType == manifest.MediaTypeManifestList {
		t.Fatal("expected unsupported source manifest to remain simple")
	}
}

func TestEnsureLocalCompatibilityMigrationPreservesPromptMetadata(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	registerTestCompatMigrator(t)

	source := model.ParseName("registry.ollama.ai/library/testcompat:latest")
	template := "TEMPLATE keep"
	writeSourceManifest(t, source, sourceManifestInput{
		config: model.ConfigV2{
			ModelFormat:   "gguf",
			ModelFamily:   "testcompat",
			ModelFamilies: []string{"testcompat"},
			Renderer:      "keep-renderer",
			Parser:        "keep-parser",
			Architecture:  "amd64",
			OS:            "linux",
			RootFS:        model.RootFS{Type: "layers"},
		},
		modelKV: ggml.KV{
			"general.architecture": "testcompat",
		},
		modelTensors: []*ggml.Tensor{
			fixtureTensor("token_embd.weight", ggml.TensorTypeF32, []uint64{2, 2}),
		},
		template: template,
	})

	migrated, err := EnsureLocalCompatibilityMigration(source)
	if err != nil {
		t.Fatalf("EnsureLocalCompatibilityMigration() error = %v", err)
	}
	if !migrated {
		t.Fatal("expected migration to create target manifest")
	}

	mf, err := manifest.ParseNamedManifest(source)
	if err != nil {
		t.Fatalf("ParseNamedManifest(target) error = %v", err)
	}
	if len(mf.Layers) != 2 {
		t.Fatalf("expected model + template layers, got %d", len(mf.Layers))
	}

	configPath, err := manifest.BlobsPath(mf.Config.Digest)
	if err != nil {
		t.Fatalf("BlobsPath(config) error = %v", err)
	}
	configFile, err := os.Open(configPath)
	if err != nil {
		t.Fatalf("Open(config) error = %v", err)
	}
	defer configFile.Close()
	var config model.ConfigV2
	if err := json.NewDecoder(configFile).Decode(&config); err != nil {
		t.Fatalf("Decode(config) error = %v", err)
	}
	if config.Renderer != "keep-renderer" || config.Parser != "keep-parser" {
		t.Fatalf("expected renderer/parser to be preserved, got %q/%q", config.Renderer, config.Parser)
	}

	templatePath, err := manifest.BlobsPath(mf.Layers[1].Digest)
	if err != nil {
		t.Fatalf("BlobsPath(template) error = %v", err)
	}
	if got, err := os.ReadFile(templatePath); err != nil {
		t.Fatalf("ReadFile(template) error = %v", err)
	} else if string(got) != template {
		t.Fatalf("expected template layer to be preserved, got %q", got)
	}
}

func TestEnsureLocalCompatibilityMigrationGemma4(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	source := model.ParseName("registry.ollama.ai/dhiltgen/gemma4:e4b")
	writeSourceManifest(t, source, sourceManifestInput{
		config: model.ConfigV2{
			ModelFormat:   "gguf",
			ModelFamily:   "gemma4",
			ModelFamilies: []string{"gemma4"},
			ModelType:     "4.3B",
			FileType:      "Q4_K_M",
			Architecture:  "amd64",
			OS:            "linux",
			RootFS:        model.RootFS{Type: "layers"},
		},
		modelKV: ggml.KV{
			"general.architecture":                       "gemma4",
			"gemma4.block_count":                         uint32(2),
			"gemma4.embedding_length":                    uint32(32),
			"gemma4.attention.head_count":                uint32(8),
			"gemma4.attention.head_count_kv":             uint32(2),
			"gemma4.attention.key_length":                uint32(64),
			"gemma4.attention.value_length":              uint32(64),
			"gemma4.attention.key_length_swa":            uint32(32),
			"gemma4.attention.value_length_swa":          uint32(32),
			"gemma4.attention.layer_norm_rms_epsilon":    float32(1e-6),
			"gemma4.attention.sliding_window":            uint32(512),
			"gemma4.attention.sliding_window_pattern":    []bool{true, false},
			"gemma4.attention.shared_kv_layers":          uint32(1),
			"gemma4.embedding_length_per_layer_input":    uint32(8),
			"gemma4.rope.dimension_count":                uint32(64),
			"gemma4.rope.dimension_count_swa":            uint32(32),
			"gemma4.rope.freq_base":                      float32(1e6),
			"gemma4.rope.freq_base_swa":                  float32(1e4),
			"gemma4.vision.block_count":                  uint32(4),
			"gemma4.vision.embedding_length":             uint32(24),
			"gemma4.vision.feed_forward_length":          uint32(96),
			"gemma4.vision.attention.head_count":         uint32(6),
			"gemma4.vision.attention.layer_norm_epsilon": float32(1e-6),
			"gemma4.vision.patch_size":                   uint32(16),
			"gemma4.vision.projector.scale_factor":       uint32(3),
			"gemma4.audio.block_count":                   uint32(3),
			"gemma4.audio.embedding_length":              uint32(16),
			"gemma4.audio.attention.head_count":          uint32(4),
			"tokenizer.ggml.model":                       "llama",
			"tokenizer.ggml.tokens":                      []string{"<bos>", "hello"},
		},
		modelTensors: []*ggml.Tensor{
			fixtureTensor("token_embd.weight", ggml.TensorTypeQ4_K, []uint64{2, 32}),
			fixtureTensor("blk.0.attn_q.weight", ggml.TensorTypeQ4_K, []uint64{32, 32}),
			fixtureTensor("blk.0.ffn_gate_exps.weight", ggml.TensorTypeF16, []uint64{32, 16, 2}),
			fixtureTensor("blk.0.ffn_up_exps.weight", ggml.TensorTypeF16, []uint64{32, 16, 2}),
			fixtureTensor("blk.0.ffn_gate_inp.per_expert_scale", ggml.TensorTypeF32, []uint64{2}),
			fixtureTensor("v.patch_embd.weight", ggml.TensorTypeF16, []uint64{16, 16, 3, 24}),
			fixtureTensor("mm.input_projection.weight", ggml.TensorTypeF16, []uint64{24, 32}),
			fixtureTensor("a.blk.0.attn_q.weight", ggml.TensorTypeBF16, []uint64{16, 16}),
			fixtureTensor("a.blk.0.linear_pos.weight", ggml.TensorTypeBF16, []uint64{16, 16}),
			fixtureTensor("a.blk.0.ln1.weight", ggml.TensorTypeF32, []uint64{16}),
			fixtureTensor("a.blk.0.ln2.weight", ggml.TensorTypeF32, []uint64{16}),
			fixtureTensor("a.blk.0.layer_pre_norm.weight", ggml.TensorTypeF32, []uint64{16}),
			fixtureTensor("a.pre_encode.out.weight", ggml.TensorTypeF16, []uint64{16, 16}),
			fixtureTensor("mm.a.input_projection.weight", ggml.TensorTypeF16, []uint64{16, 32}),
			fixtureTensor("mm.a.fc.weight", ggml.TensorTypeF16, []uint64{16, 32}),
		},
		template: "TEMPLATE gemma4",
	})

	migrated, err := EnsureLocalCompatibilityMigration(source)
	if err != nil {
		t.Fatalf("EnsureLocalCompatibilityMigration() error = %v", err)
	}
	if !migrated {
		t.Fatal("expected migration to create target manifest")
	}

	mf, err := manifest.ParseNamedManifest(source)
	if err != nil {
		t.Fatalf("ParseNamedManifest(target) error = %v", err)
	}
	if len(mf.Layers) != 3 {
		t.Fatalf("expected model + projector + template layers, got %d", len(mf.Layers))
	}

	config := readConfigLayer(t, mf.Config.Digest)
	if config.Renderer != "gemma4" || config.Parser != "gemma4" {
		t.Fatalf("expected gemma4 renderer/parser, got %q/%q", config.Renderer, config.Parser)
	}

	modelGGUF := openGGUFLayer(t, mf.Layers[0].Digest)
	defer modelGGUF.Close()
	if got := modelGGUF.KeyValue("general.architecture").String(); got != "gemma4" {
		t.Fatalf("expected model architecture gemma4, got %q", got)
	}
	if got := modelGGUF.KeyValue("tokenizer.ggml.model").String(); got != "gemma4" {
		t.Fatalf("expected migrated tokenizer.ggml.model gemma4, got %q", got)
	}
	if got := modelGGUF.TensorInfo("a.blk.0.attn_q.weight"); got.Valid() {
		t.Fatal("expected audio tensor to be moved out of migrated text model")
	}
	if got := modelGGUF.TensorInfo("v.patch_embd.weight"); got.Valid() {
		t.Fatal("expected vision tensor to be moved out of migrated text model")
	}
	if got := modelGGUF.TensorInfo("blk.0.ffn_gate_up_exps.weight"); !got.Valid() || !slices.Equal(got.Shape, []uint64{32, 32, 2}) {
		t.Fatalf("expected fused MoE gate/up tensor shape [32 32 2], got valid=%v shape=%v", got.Valid(), got.Shape)
	}
	if got := modelGGUF.TensorInfo("blk.0.ffn_up_exps.weight"); got.Valid() {
		t.Fatal("expected split MoE up tensor to be fused")
	}
	if got := modelGGUF.TensorInfo("blk.0.ffn_down_exps.scale"); !got.Valid() {
		t.Fatal("expected per-expert scale to move to ffn_down_exps.scale")
	}

	projectorGGUF := openGGUFLayer(t, mf.Layers[1].Digest)
	defer projectorGGUF.Close()
	if got := projectorGGUF.KeyValue("general.architecture").String(); got != "clip" {
		t.Fatalf("expected projector architecture clip, got %q", got)
	}
	if got := projectorGGUF.KeyValue("projector_type"); got.Valid() {
		t.Fatalf("mixed Gemma4 projector must not set generic projector type, got %q", got.String())
	}
	if got := projectorGGUF.KeyValue("vision.projector_type").String(); got != "gemma4v" {
		t.Fatalf("expected vision projector type gemma4v, got %q", got)
	}
	if got := projectorGGUF.KeyValue("has_audio_encoder").Bool(); !got {
		t.Fatal("expected projector to advertise audio encoder")
	}
	if got := projectorGGUF.KeyValue("audio.projector_type").String(); got != "gemma4a" {
		t.Fatalf("expected audio projector type gemma4a, got %q", got)
	}
	for _, name := range []string{
		"v.patch_embd.weight",
		"mm.input_projection.weight",
		"mm.a.input_projection.weight",
		"a.input_projection.weight",
		"a.pre_encode.out.weight",
		"a.blk.0.attn_q.weight",
		"a.blk.0.attn_k_rel.weight",
		"a.blk.0.attn_pre_norm.weight",
		"a.blk.0.attn_post_norm.weight",
		"a.blk.0.ln2.weight",
	} {
		if got := projectorGGUF.TensorInfo(name); !got.Valid() {
			t.Fatalf("expected migrated Gemma4 projector tensor %s", name)
		}
	}
}

func TestEnsureLocalCompatibilityMigrationGemma4CompatibleCopyNoop(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	source := model.ParseName("registry.ollama.ai/dhiltgen/gemma4:e4b-llamacpp")
	writeSourceManifest(t, source, sourceManifestInput{
		config: model.ConfigV2{
			ModelFormat:   "gguf",
			ModelFamily:   "gemma4",
			ModelFamilies: []string{"gemma4"},
			Architecture:  "amd64",
			OS:            "linux",
			RootFS:        model.RootFS{Type: "layers"},
		},
		modelKV: ggml.KV{
			"general.architecture":  "gemma4",
			"gemma4.block_count":    uint32(1),
			"tokenizer.ggml.model":  "gemma4",
			"tokenizer.ggml.tokens": []string{"<bos>", "hello"},
		},
		modelTensors: []*ggml.Tensor{
			fixtureTensor("token_embd.weight", ggml.TensorTypeQ4_K, []uint64{2, 32}),
			fixtureTensor("blk.0.attn_q.weight", ggml.TensorTypeQ4_K, []uint64{32, 32}),
		},
	})

	migrated, err := EnsureLocalCompatibilityMigration(source)
	if err != nil {
		t.Fatalf("EnsureLocalCompatibilityMigration() error = %v", err)
	}
	if migrated {
		t.Fatal("expected compatible copied Gemma4 model to skip migration")
	}

	raw, err := manifest.ReadManifestData(source)
	if err != nil {
		t.Fatalf("ReadManifestData(source) error = %v", err)
	}
	var stored manifest.Manifest
	if err := json.Unmarshal(raw, &stored); err != nil {
		t.Fatalf("unmarshal stored manifest: %v", err)
	}
	if stored.MediaType == manifest.MediaTypeManifestList {
		t.Fatal("expected compatible source manifest to remain simple")
	}
}

func TestGemma4ProjectorTensorName(t *testing.T) {
	tests := map[string]string{
		"a.blk.0.linear_pos.weight":     "a.blk.0.attn_k_rel.weight",
		"a.blk.0.ln1.weight":            "a.blk.0.attn_pre_norm.weight",
		"a.blk.0.ln2.weight":            "a.blk.0.attn_post_norm.weight",
		"a.blk.0.layer_pre_norm.weight": "a.blk.0.ln2.weight",
		"mm.a.fc.weight":                "a.pre_encode.out.weight",
		"a.pre_encode.out.weight":       "a.input_projection.weight",
		"v.patch_embd.weight":           "v.patch_embd.weight",
		"mm.input_projection.weight":    "mm.input_projection.weight",
	}

	for in, want := range tests {
		if got := gemma4ProjectorTensorName(in, true); got != want {
			t.Fatalf("gemma4ProjectorTensorName(%q) = %q, want %q", in, got, want)
		}
	}
	if got := gemma4ProjectorTensorName("a.pre_encode.out.weight", false); got != "a.pre_encode.out.weight" {
		t.Fatalf("expected non-legacy audio name to be preserved, got %q", got)
	}
}

func TestEnsureLocalCompatibilityMigrationSerializesConcurrentCalls(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	source := model.ParseName("registry.ollama.ai/library/testcompat:latest")
	writeSourceManifest(t, source, sourceManifestInput{
		config: model.ConfigV2{
			ModelFormat:   "gguf",
			ModelFamily:   "testcompat",
			ModelFamilies: []string{"testcompat"},
			Architecture:  "amd64",
			OS:            "linux",
			RootFS:        model.RootFS{Type: "layers"},
		},
		modelKV: ggml.KV{
			"general.architecture":  "testcompat",
			"tokenizer.ggml.tokens": []string{"x"},
		},
		modelTensors: []*ggml.Tensor{
			fixtureTensor("token_embd.weight", ggml.TensorTypeF16, []uint64{1, 8}),
		},
	})

	var calls atomic.Int32
	registerCountingCompatMigrator(t, &calls)

	const workers = 8
	var wg sync.WaitGroup
	errs := make(chan error, workers)
	start := make(chan struct{})
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			migrated, err := EnsureLocalCompatibilityMigration(source)
			if err != nil {
				errs <- err
				return
			}
			if !migrated {
				errs <- errors.New("expected migration to succeed")
				return
			}
		}()
	}
	close(start)
	wg.Wait()
	close(errs)

	for err := range errs {
		if err != nil {
			t.Fatalf("concurrent migration failed: %v", err)
		}
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("expected one migration under concurrent calls, got %d", got)
	}
}

func TestEnsureLocalCompatibilityMigrationSkipsWhenDiskIsTooFull(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	registerTestCompatMigrator(t)

	source := model.ParseName("registry.ollama.ai/library/testcompat:latest")
	writeSourceManifest(t, source, sourceManifestInput{
		config: model.ConfigV2{
			ModelFormat:  "gguf",
			ModelFamily:  "testcompat",
			Architecture: "amd64",
			OS:           "linux",
			RootFS:       model.RootFS{Type: "layers"},
		},
		modelKV: ggml.KV{
			"general.architecture":  "testcompat",
			"tokenizer.ggml.tokens": []string{"x"},
		},
		modelTensors: []*ggml.Tensor{
			fixtureTensor("token_embd.weight", ggml.TensorTypeF16, []uint64{1, 8}),
		},
	})

	old := availableSpaceForPath
	availableSpaceForPath = func(string) (uint64, error) { return 0, nil }
	defer func() { availableSpaceForPath = old }()

	migrated, err := EnsureLocalCompatibilityMigration(source)
	if err != nil {
		t.Fatalf("EnsureLocalCompatibilityMigration() error = %v", err)
	}
	if migrated {
		t.Fatal("expected migration to skip when disk headroom is insufficient")
	}
	data, err := manifest.ReadManifestData(source)
	if err != nil {
		t.Fatalf("ReadManifestData(source) error = %v", err)
	}
	var stored manifest.Manifest
	if err := json.Unmarshal(data, &stored); err != nil {
		t.Fatalf("unmarshal stored manifest: %v", err)
	}
	if stored.MediaType == manifest.MediaTypeManifestList {
		t.Fatal("expected source manifest to remain simple when disk headroom is insufficient")
	}
}

func TestQwen3VLSplitLegacyPatchTensorOrder(t *testing.T) {
	shape := []uint64{2, 2, 2, 6}
	raw := make([]byte, tensorBytes(ggml.TensorTypeF16, shape))
	for i := range len(raw) / 2 {
		putF16(raw[i*2:], float32(i))
	}

	source := &sourceTensor{
		readerAt: bytes.NewReader(raw),
		info: gguf.TensorInfo{
			Name:  "v.patch_embed.weight",
			Shape: shape,
			Type:  gguf.TensorTypeF16,
		},
		name:  "v.patch_embed.weight",
		shape: shape,
	}

	got, err := qwen3VLSplitLegacyPatchTensor(source, 3)
	if err != nil {
		t.Fatalf("qwen3VLSplitLegacyPatchTensor() error = %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("expected two split tensors, got %d", len(got))
	}
	if !slices.Equal(got[0].Shape, []uint64{2, 2, 3, 2}) || !slices.Equal(got[1].Shape, []uint64{2, 2, 3, 2}) {
		t.Fatalf("unexpected split shapes: %v %v", got[0].Shape, got[1].Shape)
	}

	first := writeTensorF32(t, got[0])
	second := writeTensorF32(t, got[1])
	wantFirst := []float32{0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27, 32, 33, 34, 35, 40, 41, 42, 43}
	wantSecond := []float32{4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31, 36, 37, 38, 39, 44, 45, 46, 47}
	if !slices.Equal(first, wantFirst) {
		t.Fatalf("unexpected first temporal patch split:\n got %v\nwant %v", first, wantFirst)
	}
	if !slices.Equal(second, wantSecond) {
		t.Fatalf("unexpected second temporal patch split:\n got %v\nwant %v", second, wantSecond)
	}
}

func TestQwen3VLConcatQKVWeightsOrder(t *testing.T) {
	q := sourceTensorF16("v.blk.0.attn_q.weight", []uint64{2, 2}, []float32{0, 1, 2, 3})
	k := sourceTensorF16("v.blk.0.attn_k.weight", []uint64{2, 2}, []float32{4, 5, 6, 7})
	v := sourceTensorF16("v.blk.0.attn_v.weight", []uint64{2, 2}, []float32{8, 9, 10, 11})

	got, err := qwen3VLConcatQKVWeights("v.blk.0.attn_qkv.weight", q, k, v)
	if err != nil {
		t.Fatalf("qwen3VLConcatQKVWeights() error = %v", err)
	}
	if !slices.Equal(got.Shape, []uint64{2, 6}) {
		t.Fatalf("unexpected qkv shape: %v", got.Shape)
	}

	if values := writeTensorF16(t, got); !slices.Equal(values, []float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}) {
		t.Fatalf("unexpected qkv raw order: %v", values)
	}
}

func TestMistralPixtralVisionQKRepack(t *testing.T) {
	data := []float32{
		0, 1, 2, 3, 4, 5, 6, 7,
		8, 9, 10, 11, 12, 13, 14, 15,
		16, 17, 18, 19, 20, 21, 22, 23,
		24, 25, 26, 27, 28, 29, 30, 31,
	}
	got, err := mistralPixtralVisionQKRepack(data, []uint64{8, 4}, 2)
	if err != nil {
		t.Fatalf("mistralPixtralVisionQKRepack() error = %v", err)
	}

	want := []float32{
		0, 1, 2, 3, 8, 9, 10, 11,
		4, 5, 6, 7, 12, 13, 14, 15,
		16, 17, 18, 19, 24, 25, 26, 27,
		20, 21, 22, 23, 28, 29, 30, 31,
	}
	if !slices.Equal(got, want) {
		t.Fatalf("unexpected mistral/pixtral qk repack:\n got %v\nwant %v", got, want)
	}
}

func TestDecodeQuantizedTensorRows(t *testing.T) {
	q8 := make([]byte, q8_0BlkSize)
	putF16(q8, 1)
	q8[2] = 3
	q8[3] = 0xfe
	gotQ8, err := decodeQ8_0Row(q8, qk8_0)
	if err != nil {
		t.Fatalf("decodeQ8_0Row() error = %v", err)
	}
	if gotQ8[0] != 3 || gotQ8[1] != -2 {
		t.Fatalf("unexpected q8_0 decode prefix: %v", gotQ8[:2])
	}

	q5 := make([]byte, q5_0BlkSize)
	putF16(q5, 1)
	binary.LittleEndian.PutUint32(q5[2:], (1<<0)|(1<<16))
	q5[6] = 0x21
	gotQ5, err := decodeQ5_0Row(q5, qk5_0)
	if err != nil {
		t.Fatalf("decodeQ5_0Row() error = %v", err)
	}
	if gotQ5[0] != 1 || gotQ5[16] != 2 {
		t.Fatalf("unexpected q5_0 decode prefix: %v %v", gotQ5[0], gotQ5[16])
	}

	q4 := make([]byte, q4KBlkSize)
	putF16(q4, 1)
	putF16(q4[2:], 0)
	for i := 4; i < 16; i++ {
		q4[i] = 1
	}
	for i := 16; i < q4KBlkSize; i++ {
		q4[i] = 0x21
	}
	gotQ4, err := decodeQ4KRow(q4, qkK)
	if err != nil {
		t.Fatalf("decodeQ4KRow() error = %v", err)
	}
	if gotQ4[0] != 1 || gotQ4[31] != 1 || gotQ4[32] != 2 || gotQ4[63] != 2 {
		t.Fatalf("unexpected q4_k decode prefix: %v %v %v %v", gotQ4[0], gotQ4[31], gotQ4[32], gotQ4[63])
	}

	q6 := make([]byte, q6KBlkSize)
	for i := 192; i < 208; i++ {
		q6[i] = 1
	}
	putF16(q6[208:], 1)
	gotQ6, err := decodeQ6KRow(q6, qkK)
	if err != nil {
		t.Fatalf("decodeQ6KRow() error = %v", err)
	}
	if gotQ6[0] != -32 || gotQ6[127] != -32 {
		t.Fatalf("unexpected q6_k decode prefix: %v %v", gotQ6[0], gotQ6[127])
	}
}

func readConfigLayer(t *testing.T, digest string) model.ConfigV2 {
	t.Helper()

	configPath, err := manifest.BlobsPath(digest)
	if err != nil {
		t.Fatalf("BlobsPath(config) error = %v", err)
	}
	configFile, err := os.Open(configPath)
	if err != nil {
		t.Fatalf("Open(config) error = %v", err)
	}
	defer configFile.Close()

	var config model.ConfigV2
	if err := json.NewDecoder(configFile).Decode(&config); err != nil {
		t.Fatalf("Decode(config) error = %v", err)
	}
	return config
}

func openGGUFLayer(t *testing.T, digest string) *gguf.File {
	t.Helper()

	path, err := manifest.BlobsPath(digest)
	if err != nil {
		t.Fatalf("BlobsPath(%s) error = %v", digest, err)
	}
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("gguf.Open(%s) error = %v", path, err)
	}
	return f
}

type testCompatMigrator struct{}

func (testCompatMigrator) NeedsMigration(*SourceModel) bool {
	return true
}

func (testCompatMigrator) Migrate(src *SourceModel) (*Result, error) {
	tensors, err := readAllSourceTensors(src)
	if err != nil {
		return nil, err
	}

	kv := ggml.KV{}
	for _, keyValue := range src.GGUF.KeyValues() {
		if !keyValue.Valid() {
			continue
		}
		kv[keyValue.Key] = normalizeGGUFValue(rawGGUFValue(keyValue.Value))
	}
	if kv.String("general.architecture") == "" {
		kv["general.architecture"] = "testcompat"
	}

	out := make([]*ggml.Tensor, 0, len(tensors))
	for _, tensor := range tensors {
		out = append(out, copyTensor(tensor.name, tensor))
	}

	return &Result{
		ModelKV:      kv,
		ModelTensors: out,
	}, nil
}

type countingCompatMigrator struct {
	calls *atomic.Int32
}

func (countingCompatMigrator) NeedsMigration(*SourceModel) bool {
	return true
}

func (m countingCompatMigrator) Migrate(src *SourceModel) (*Result, error) {
	m.calls.Add(1)
	time.Sleep(50 * time.Millisecond)
	return testCompatMigrator{}.Migrate(src)
}

func registerTestCompatMigrator(t *testing.T) {
	t.Helper()

	const key = "testcompat"
	old, ok := migratorsByArchitecture[key]
	migratorsByArchitecture[key] = []Migrator{testCompatMigrator{}}
	t.Cleanup(func() {
		if ok {
			migratorsByArchitecture[key] = old
		} else {
			delete(migratorsByArchitecture, key)
		}
	})
}

func registerCountingCompatMigrator(t *testing.T, calls *atomic.Int32) {
	t.Helper()

	const key = "testcompat"
	old, ok := migratorsByArchitecture[key]
	migratorsByArchitecture[key] = []Migrator{countingCompatMigrator{calls: calls}}
	t.Cleanup(func() {
		if ok {
			migratorsByArchitecture[key] = old
		} else {
			delete(migratorsByArchitecture, key)
		}
	})
}

func wrapSourceManifestAsList(t *testing.T, name model.Name) {
	t.Helper()

	data, err := manifest.ReadManifestData(name)
	if err != nil {
		t.Fatalf("ReadManifestData(source) error = %v", err)
	}
	var child manifest.Manifest
	if err := json.Unmarshal(data, &child); err != nil {
		t.Fatalf("unmarshal child manifest: %v", err)
	}
	if err := fillManifestMetadata(&child); err != nil {
		t.Fatalf("fillManifestMetadata(child) error = %v", err)
	}
	ref, err := manifestReferenceForChild(&child)
	if err != nil {
		t.Fatalf("manifestReferenceForChild() error = %v", err)
	}
	parent := manifest.Manifest{
		SchemaVersion: 2,
		MediaType:     manifest.MediaTypeManifestList,
		Manifests:     []manifest.Manifest{ref},
	}
	parentData, err := json.Marshal(parent)
	if err != nil {
		t.Fatalf("marshal parent manifest: %v", err)
	}
	if err := manifest.WriteManifestData(name, parentData); err != nil {
		t.Fatalf("WriteManifestData(parent) error = %v", err)
	}
}

type sourceManifestInput struct {
	config           model.ConfigV2
	modelKV          ggml.KV
	modelTensors     []*ggml.Tensor
	projectorKV      ggml.KV
	projectorTensors []*ggml.Tensor
	template         string
}

func writeSourceManifest(t *testing.T, name model.Name, input sourceManifestInput) {
	t.Helper()

	modelLayer := writeFixtureGGUFLayer(t, input.modelKV, input.modelTensors)

	layers := []manifest.Layer{modelLayer}
	if len(input.projectorTensors) > 0 {
		projectorLayer := writeFixtureGGUFLayer(t, input.projectorKV, input.projectorTensors)
		projectorLayer.MediaType = "application/vnd.ollama.image.projector"
		layers = append(layers, projectorLayer)
	}
	if input.template != "" {
		layer, err := manifest.NewLayer(strings.NewReader(input.template), "application/vnd.ollama.image.template")
		if err != nil {
			t.Fatalf("manifest.NewLayer(template) error = %v", err)
		}
		layers = append(layers, layer)
	}

	configLayer, err := createConfigLayer(layers, input.config)
	if err != nil {
		t.Fatalf("createConfigLayer() error = %v", err)
	}
	if err := manifest.WriteManifest(name, *configLayer, layers); err != nil {
		t.Fatalf("WriteManifest() error = %v", err)
	}
}

func writeFixtureGGUFLayer(t *testing.T, kv ggml.KV, tensors []*ggml.Tensor) manifest.Layer {
	t.Helper()

	f, err := os.CreateTemp(t.TempDir(), "fixture-*.gguf")
	if err != nil {
		t.Fatalf("CreateTemp() error = %v", err)
	}
	defer os.Remove(f.Name())
	defer f.Close()

	if err := ggml.WriteGGUF(f, kv, tensors); err != nil {
		t.Fatalf("WriteGGUF() error = %v", err)
	}
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		t.Fatalf("Seek() error = %v", err)
	}

	layer, err := manifest.NewLayer(f, "application/vnd.ollama.image.model")
	if err != nil {
		t.Fatalf("manifest.NewLayer(model) error = %v", err)
	}
	return layer
}

func fixtureTensor(name string, kind ggml.TensorType, shape []uint64) *ggml.Tensor {
	return &ggml.Tensor{
		Name:     name,
		Kind:     uint32(kind),
		Shape:    shape,
		WriterTo: bytes.NewReader(make([]byte, tensorBytes(kind, shape))),
	}
}

func tensorBytes(kind ggml.TensorType, shape []uint64) int {
	var values uint64 = 1
	for _, dim := range shape {
		values *= dim
	}
	return int(values * kind.TypeSize() / kind.BlockSize())
}

func writeTensorF32(t *testing.T, tensor *ggml.Tensor) []float32 {
	t.Helper()

	var b bytes.Buffer
	if _, err := tensor.WriterTo.WriteTo(&b); err != nil {
		t.Fatalf("WriteTo(%s) error = %v", tensor.Name, err)
	}
	if b.Len()%4 != 0 {
		t.Fatalf("WriteTo(%s) produced %d bytes, not f32 aligned", tensor.Name, b.Len())
	}

	out := make([]float32, b.Len()/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b.Bytes()[i*4:]))
	}
	return out
}

func writeTensorF16(t *testing.T, tensor *ggml.Tensor) []float32 {
	t.Helper()

	var b bytes.Buffer
	if _, err := tensor.WriterTo.WriteTo(&b); err != nil {
		t.Fatalf("WriteTo(%s) error = %v", tensor.Name, err)
	}
	if b.Len()%2 != 0 {
		t.Fatalf("WriteTo(%s) produced %d bytes, not f16 aligned", tensor.Name, b.Len())
	}

	out := make([]float32, b.Len()/2)
	for i := range out {
		out[i] = float16.Frombits(binary.LittleEndian.Uint16(b.Bytes()[i*2:])).Float32()
	}
	return out
}

func sourceTensorF16(name string, shape []uint64, values []float32) *sourceTensor {
	raw := make([]byte, len(values)*2)
	for i, value := range values {
		putF16(raw[i*2:], value)
	}

	return &sourceTensor{
		readerAt: bytes.NewReader(raw),
		info: gguf.TensorInfo{
			Name:  name,
			Shape: shape,
			Type:  gguf.TensorTypeF16,
		},
		name:  name,
		shape: shape,
	}
}

func putF16(b []byte, v float32) {
	binary.LittleEndian.PutUint16(b, float16.Fromfloat32(v).Bits())
}
