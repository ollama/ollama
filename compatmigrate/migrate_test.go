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
			ModelFormat: "gguf",
			ModelFamily: "notcompat",
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

func TestEnsureLocalCompatibilityMigrationSkipsAdapterModels(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	registerTestCompatMigrator(t)

	source := model.ParseName("registry.ollama.ai/library/testcompat:adapter")
	writeSourceManifest(t, source, sourceManifestInput{
		config: model.ConfigV2{
			ModelFormat: "gguf",
			ModelFamily: "testcompat",
		},
		modelKV: ggml.KV{
			"general.architecture": "testcompat",
		},
		modelTensors: []*ggml.Tensor{
			fixtureTensor("token_embd.weight", ggml.TensorTypeF32, []uint64{2, 2}),
		},
		adapter: "fake lora adapter payload",
	})

	migrated, err := EnsureLocalCompatibilityMigration(source)
	if err != nil {
		t.Fatalf("EnsureLocalCompatibilityMigration() error = %v", err)
	}
	if migrated {
		t.Fatal("expected adapter-bearing source to skip migration; the converted child would drop the adapter")
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
		t.Fatal("expected adapter-bearing source manifest to remain simple")
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

	mf, err := manifest.ParseNamedManifestForRunner(source, manifest.RunnerLlamaCPP)
	if err != nil {
		t.Fatalf("ParseNamedManifestForRunner(target, llamacpp) error = %v", err)
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

	source := model.ParseName("registry.ollama.ai/library/gemma4:e4b")
	writeSourceManifest(t, source, sourceManifestInput{
		config: model.ConfigV2{
			ModelFormat:   "gguf",
			ModelFamily:   "gemma4",
			ModelFamilies: []string{"gemma4"},
			ModelType:     "4.3B",
			FileType:      "Q4_K_M",
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
			fixtureTensor("token_embd.weight", ggml.TensorTypeF16, []uint64{2, 32}),
			fixtureTensor("blk.0.attn_q.weight", ggml.TensorTypeF16, []uint64{32, 32}),
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

	mf, err := manifest.ParseNamedManifestForRunner(source, manifest.RunnerLlamaCPP)
	if err != nil {
		t.Fatalf("ParseNamedManifestForRunner(target, llamacpp) error = %v", err)
	}
	if len(mf.Layers) != 3 {
		var mediaTypes []string
		for _, layer := range mf.Layers {
			mediaTypes = append(mediaTypes, layer.MediaType)
		}
		t.Fatalf("expected model + projector + template layers, got %d: %v runner=%q format=%q", len(mf.Layers), mediaTypes, mf.Runner, mf.Format)
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

	source := model.ParseName("registry.ollama.ai/library/gemma4:e4b-llamacpp")
	writeSourceManifest(t, source, sourceManifestInput{
		config: model.ConfigV2{
			ModelFormat:   "gguf",
			ModelFamily:   "gemma4",
			ModelFamilies: []string{"gemma4"},
		},
		modelKV: ggml.KV{
			"general.architecture":  "gemma4",
			"gemma4.block_count":    uint32(1),
			"tokenizer.ggml.model":  "gemma4",
			"tokenizer.ggml.tokens": []string{"<bos>", "hello"},
		},
		modelTensors: []*ggml.Tensor{
			fixtureTensor("token_embd.weight", ggml.TensorTypeF16, []uint64{2, 32}),
			fixtureTensor("blk.0.attn_q.weight", ggml.TensorTypeF16, []uint64{32, 32}),
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

func TestDeepseekOCRProjectorTensorName(t *testing.T) {
	const (
		publishedViewSeparator = "mm.view_seperator" //nolint:misspell // published DeepSeek OCR tensor spelling
		llamaCPPViewSeparator  = "v.view_seperator"  //nolint:misspell // published DeepSeek OCR tensor spelling
	)
	tests := map[string]string{
		publishedViewSeparator: llamaCPPViewSeparator,
		"mm.view_separator":    "v.view_separator",
	}

	for in, want := range tests {
		if got := deepseekOCRProjectorTensorName(in); got != want {
			t.Fatalf("deepseekOCRProjectorTensorName(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestEnsureLocalCompatibilityMigrationLaguna(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	source := model.ParseName("registry.ollama.ai/library/laguna-xs.2:q4_K_M")
	writeSourceManifest(t, source, sourceManifestInput{
		config: model.ConfigV2{
			ModelFormat:   "gguf",
			ModelFamily:   "laguna",
			ModelFamilies: []string{"laguna"},
			Renderer:      "laguna",
			Parser:        "laguna",
		},
		modelKV: ggml.KV{
			"general.architecture":                    "laguna",
			"laguna.block_count":                      uint32(2),
			"laguna.embedding_length":                 uint32(32),
			"laguna.attention.head_count":             uint32(8),
			"laguna.attention.head_count_kv":          uint32(2),
			"laguna.attention.layer_norm_rms_epsilon": float32(1e-6),
			"laguna.attention.sliding_window":         uint32(512),
			"laguna.attention.layer_types":            []int32{1, 0},
			"laguna.rope.dimension_count":             uint32(64),
			"laguna.rope.freq_base":                   float32(1e6),
			"laguna.rope.swa.dimension_count":         uint32(32),
			"laguna.rope.swa.freq_base":               float32(1e4),
			"tokenizer.ggml.model":                    "gpt2",
			"tokenizer.ggml.tokens":                   []string{"<bos>", "hello"},
		},
		modelTensors: []*ggml.Tensor{
			fixtureTensor("token_embd.weight", ggml.TensorTypeF16, []uint64{2, 32}),
			fixtureTensor("blk.0.attn_q.weight", ggml.TensorTypeF16, []uint64{32, 32}),
			fixtureTensor("blk.0.attn_g.weight", ggml.TensorTypeF16, []uint64{32, 32}),
			fixtureTensor("blk.1.attn_g.weight", ggml.TensorTypeF16, []uint64{32, 32}),
		},
		template: "TEMPLATE laguna",
	})

	migrated, err := EnsureLocalCompatibilityMigration(source)
	if err != nil {
		t.Fatalf("EnsureLocalCompatibilityMigration() error = %v", err)
	}
	if !migrated {
		t.Fatal("expected migration to create target manifest")
	}

	mf, err := manifest.ParseNamedManifestForRunner(source, manifest.RunnerLlamaCPP)
	if err != nil {
		t.Fatalf("ParseNamedManifestForRunner(target, llamacpp) error = %v", err)
	}
	if len(mf.Layers) != 2 {
		t.Fatalf("expected model + template layers, got %d", len(mf.Layers))
	}

	config := readConfigLayer(t, mf.Config.Digest)
	if config.Renderer != "laguna" || config.Parser != "laguna" {
		t.Fatalf("expected laguna renderer/parser, got %q/%q", config.Renderer, config.Parser)
	}

	modelGGUF := openGGUFLayer(t, mf.Layers[0].Digest)
	defer modelGGUF.Close()
	if got := modelGGUF.KeyValue("general.architecture").String(); got != "laguna" {
		t.Fatalf("expected model architecture laguna, got %q", got)
	}
	if got := modelGGUF.KeyValue("rope.dimension_count_swa"); !got.Valid() || got.Uint() != 32 {
		t.Fatalf("expected migrated rope.dimension_count_swa 32, got valid=%v value=%d", got.Valid(), got.Uint())
	}
	if got := modelGGUF.KeyValue("rope.freq_base_swa"); !got.Valid() || got.Float() != 1e4 {
		t.Fatalf("expected migrated rope.freq_base_swa 1e4, got valid=%v value=%f", got.Valid(), got.Float())
	}
	if got := modelGGUF.KeyValue("rope.swa.dimension_count"); got.Valid() {
		t.Fatal("expected legacy rope.swa.dimension_count to be dropped")
	}
	if got := modelGGUF.KeyValue("rope.swa.freq_base"); got.Valid() {
		t.Fatal("expected legacy rope.swa.freq_base to be dropped")
	}
	for _, name := range []string{"blk.0.attn_gate.weight", "blk.1.attn_gate.weight", "blk.0.attn_q.weight", "token_embd.weight"} {
		if got := modelGGUF.TensorInfo(name); !got.Valid() {
			t.Fatalf("expected migrated laguna tensor %s", name)
		}
	}
	for _, name := range []string{"blk.0.attn_g.weight", "blk.1.attn_g.weight"} {
		if got := modelGGUF.TensorInfo(name); got.Valid() {
			t.Fatalf("expected legacy laguna tensor %s to be renamed", name)
		}
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
			ModelFormat: "gguf",
			ModelFamily: "testcompat",
		},
		modelKV: ggml.KV{
			"general.architecture":  "testcompat",
			"tokenizer.ggml.tokens": []string{"x"},
		},
		modelTensors: []*ggml.Tensor{
			fixtureTensor("token_embd.weight", ggml.TensorTypeF16, []uint64{1, 8}),
		},
	})

	overrideAvailableSpace(t, func(string) (uint64, error) { return 0, nil })

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
	t.Run("q8_0", func(t *testing.T) {
		q8 := make([]byte, q8_0BlkSize)
		putF16(q8, 1) // block scale d = 1.0, so decoded values equal the raw int8 quants
		q8[2] = 3
		q8[3] = 0xfe // int8(0xfe) = -2
		gotQ8, err := decodeQ8_0Row(q8, qk8_0)
		if err != nil {
			t.Fatalf("decodeQ8_0Row() error = %v", err)
		}
		if gotQ8[0] != 3 || gotQ8[1] != -2 {
			t.Fatalf("unexpected q8_0 decode prefix: %v", gotQ8[:2])
		}
	})

	t.Run("q5_0", func(t *testing.T) {
		q5 := make([]byte, q5_0BlkSize)
		putF16(q5, 1) // block scale d = 1.0
		// qh bits 0 and 16 set the fifth (high) bit for elements 0 and 16.
		binary.LittleEndian.PutUint32(q5[2:], (1<<0)|(1<<16))
		// 0x21 packs low nibble 1 (element 0) and high nibble 2 (element 16);
		// with the high bits above they decode as (1|0x10)-16=1 and (2|0x10)-16=2.
		q5[6] = 0x21
		gotQ5, err := decodeQ5_0Row(q5, qk5_0)
		if err != nil {
			t.Fatalf("decodeQ5_0Row() error = %v", err)
		}
		if gotQ5[0] != 1 || gotQ5[16] != 2 {
			t.Fatalf("unexpected q5_0 decode prefix: %v %v", gotQ5[0], gotQ5[16])
		}
	})

	t.Run("q4_K", func(t *testing.T) {
		q4 := make([]byte, q4KBlkSize)
		putF16(q4, 1)     // super-block scale d = 1.0
		putF16(q4[2:], 0) // super-block min = 0, so per-sub-block mins drop out
		for i := 4; i < 16; i++ {
			q4[i] = 1 // packed 6-bit sub-block scales/mins all decode to 1
		}
		for i := 16; i < q4KBlkSize; i++ {
			q4[i] = 0x21 // packs 4-bit quants: low nibble 1 (first 32 values), high nibble 2 (next 32)
		}
		gotQ4, err := decodeQ4KRow(q4, qkK)
		if err != nil {
			t.Fatalf("decodeQ4KRow() error = %v", err)
		}
		if gotQ4[0] != 1 || gotQ4[31] != 1 || gotQ4[32] != 2 || gotQ4[63] != 2 {
			t.Fatalf("unexpected q4_k decode prefix: %v %v %v %v", gotQ4[0], gotQ4[31], gotQ4[32], gotQ4[63])
		}
	})

	t.Run("q6_K", func(t *testing.T) {
		q6 := make([]byte, q6KBlkSize)
		for i := 192; i < 208; i++ {
			q6[i] = 1 // int8 sub-block scales = 1
		}
		putF16(q6[208:], 1) // super-block scale d = 1.0
		// All-zero 6-bit quants decode as 0-32 = -32 (q6_k centers quants on 32).
		gotQ6, err := decodeQ6KRow(q6, qkK)
		if err != nil {
			t.Fatalf("decodeQ6KRow() error = %v", err)
		}
		if gotQ6[0] != -32 || gotQ6[127] != -32 {
			t.Fatalf("unexpected q6_k decode prefix: %v %v", gotQ6[0], gotQ6[127])
		}
	})
}

// TestNeedsMigrationAlreadyConverted verifies that files already in
// llama.cpp-compatible form are not re-detected as needing migration, with one
// positive control per family to keep the detectors honest.
func TestNeedsMigrationAlreadyConverted(t *testing.T) {
	// llama3NeedsMetadataFix requires the real Llama 3 marker tokens at their ids.
	llama3Tokens := make([]string, 128010)
	llama3Tokens[128006] = "<|start_header_id|>"
	llama3Tokens[128009] = "<|eot_id|>"

	cases := []struct {
		name             string
		migrator         Migrator
		kv               ggml.KV
		tensors          []*ggml.Tensor
		projectorKV      ggml.KV
		projectorTensors []*ggml.Tensor
		want             bool
	}{
		{
			name:     "embeddinggemma already converted dense sidecar only",
			migrator: embeddingGemmaMigrator{},
			kv: ggml.KV{
				"general.architecture": "gemma3",
			},
			tensors: []*ggml.Tensor{
				fixtureTensor("dense.0.bias", ggml.TensorTypeF16, []uint64{8}),
			},
			want: false,
		},
		{
			name:     "embeddinggemma legacy dense weight",
			migrator: embeddingGemmaMigrator{},
			kv: ggml.KV{
				"general.architecture": "gemma3",
			},
			tensors: []*ggml.Tensor{
				fixtureTensor("dense.0.weight", ggml.TensorTypeF16, []uint64{8, 8}),
			},
			want: true,
		},
		{
			name:     "gemma3 already converted",
			migrator: gemma3Migrator{},
			kv: ggml.KV{
				"general.architecture":                    "gemma3",
				"gemma3.rope.freq_base":                   float32(1e6),
				"gemma3.rope.freq_base_swa":               float32(1e4),
				"gemma3.attention.layer_norm_rms_epsilon": float32(1e-6),
			},
			tensors: []*ggml.Tensor{
				fixtureTensor("token_embd.weight", ggml.TensorTypeF16, []uint64{2, 8}),
			},
			want: false,
		},
		{
			name:     "gemma3 legacy rope keys",
			migrator: gemma3Migrator{},
			kv: ggml.KV{
				"general.architecture":         "gemma3",
				"gemma3.rope.global.freq_base": float32(1e6),
			},
			tensors: []*ggml.Tensor{
				fixtureTensor("token_embd.weight", ggml.TensorTypeF16, []uint64{2, 8}),
			},
			want: true,
		},
		{
			name:     "laguna already converted",
			migrator: lagunaMigrator{},
			kv: ggml.KV{
				"general.architecture":            "laguna",
				"laguna.rope.dimension_count_swa": uint32(32),
				"laguna.rope.freq_base_swa":       float32(1e4),
			},
			tensors: []*ggml.Tensor{
				fixtureTensor("blk.0.attn_gate.weight", ggml.TensorTypeF16, []uint64{8, 8}),
			},
			want: false,
		},
		{
			name:     "laguna legacy swa keys and gate tensor",
			migrator: lagunaMigrator{},
			kv: ggml.KV{
				"general.architecture":            "laguna",
				"laguna.rope.swa.dimension_count": uint32(32),
				"laguna.rope.swa.freq_base":       float32(1e4),
			},
			tensors: []*ggml.Tensor{
				fixtureTensor("blk.0.attn_g.weight", ggml.TensorTypeF16, []uint64{8, 8}),
			},
			want: true,
		},
		{
			name:     "laguna legacy swa keys without gate tensor",
			migrator: lagunaMigrator{},
			kv: ggml.KV{
				"general.architecture":            "laguna",
				"laguna.rope.swa.dimension_count": uint32(32),
				"laguna.rope.swa.freq_base":       float32(1e4),
			},
			tensors: []*ggml.Tensor{
				fixtureTensor("blk.0.attn_gate.weight", ggml.TensorTypeF16, []uint64{8, 8}),
			},
			want: true,
		},
		{
			name:     "qwen3next already converted",
			migrator: qwen3NextMigrator{},
			kv: ggml.KV{
				"general.architecture": "qwen3next",
			},
			tensors: []*ggml.Tensor{
				fixtureTensor("blk.0.ssm_dt.bias", ggml.TensorTypeF32, []uint64{8}),
			},
			want: false,
		},
		{
			name:     "qwen3next legacy ssm_dt tensor",
			migrator: qwen3NextMigrator{},
			kv: ggml.KV{
				"general.architecture": "qwen3next",
			},
			tensors: []*ggml.Tensor{
				fixtureTensor("blk.0.ssm_dt", ggml.TensorTypeF32, []uint64{8}),
			},
			want: true,
		},
		{
			name:     "llama3 already converted",
			migrator: llama3Migrator{},
			kv: ggml.KV{
				"general.architecture":        "llama",
				"tokenizer.ggml.pre":          "llama-bpe",
				"tokenizer.ggml.eos_token_id": uint32(128009),
				"tokenizer.ggml.tokens":       llama3Tokens,
			},
			tensors: []*ggml.Tensor{
				fixtureTensor("token_embd.weight", ggml.TensorTypeF16, []uint64{2, 8}),
			},
			want: false,
		},
		{
			name:     "llama3 legacy metadata gap",
			migrator: llama3Migrator{},
			kv: ggml.KV{
				"general.architecture":        "llama",
				"tokenizer.ggml.pre":          "default",
				"tokenizer.ggml.eos_token_id": uint32(128001),
				"tokenizer.ggml.tokens":       llama3Tokens,
			},
			tensors: []*ggml.Tensor{
				fixtureTensor("token_embd.weight", ggml.TensorTypeF16, []uint64{2, 8}),
			},
			want: true,
		},
		{
			name:     "llama3 missing eos with explicit pre",
			migrator: llama3Migrator{},
			kv: ggml.KV{
				"general.architecture":  "llama",
				"tokenizer.ggml.pre":    "llama-bpe",
				"tokenizer.ggml.tokens": llama3Tokens,
			},
			tensors: []*ggml.Tensor{
				fixtureTensor("token_embd.weight", ggml.TensorTypeF16, []uint64{2, 8}),
			},
			want: false,
		},
		{
			name:     "bakllava legacy clip projector",
			migrator: bakllavaMigrator{},
			kv: ggml.KV{
				"general.architecture": "llama",
			},
			tensors: []*ggml.Tensor{
				fixtureTensor("token_embd.weight", ggml.TensorTypeF16, []uint64{2, 8}),
			},
			projectorKV: ggml.KV{
				"general.architecture":    "clip",
				"clip.has_vision_encoder": true,
			},
			projectorTensors: []*ggml.Tensor{
				fixtureTensor("mm.0.weight", ggml.TensorTypeF16, []uint64{2, 2}),
			},
			want: true,
		},
		{
			name:     "bakllava clip projector without vision encoder",
			migrator: bakllavaMigrator{},
			kv: ggml.KV{
				"general.architecture": "llama",
			},
			tensors: []*ggml.Tensor{
				fixtureTensor("token_embd.weight", ggml.TensorTypeF16, []uint64{2, 8}),
			},
			projectorKV: ggml.KV{
				"general.architecture": "clip",
			},
			projectorTensors: []*ggml.Tensor{
				fixtureTensor("mm.0.weight", ggml.TensorTypeF16, []uint64{2, 2}),
			},
			want: false,
		},
		{
			name:     "bakllava non clip projector",
			migrator: bakllavaMigrator{},
			kv: ggml.KV{
				"general.architecture": "llama",
			},
			tensors: []*ggml.Tensor{
				fixtureTensor("token_embd.weight", ggml.TensorTypeF16, []uint64{2, 8}),
			},
			projectorKV: ggml.KV{
				"general.architecture": "bert",
				"bert.pooling_type":    uint32(1),
			},
			projectorTensors: []*ggml.Tensor{
				fixtureTensor("token_embd.weight", ggml.TensorTypeF16, []uint64{2, 8}),
			},
			want: false,
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			src := fixtureSourceModel(t, tt.kv, tt.tensors)
			if tt.projectorKV != nil {
				projector := fixtureSourceModel(t, tt.projectorKV, tt.projectorTensors)
				src.ProjectorGGUF = projector.GGUF
			}
			if got := tt.migrator.NeedsMigration(src); got != tt.want {
				t.Fatalf("NeedsMigration() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGPTOSSExpertFeedForwardLengthFromTensor(t *testing.T) {
	src := fixtureSourceModel(t, ggml.KV{
		"general.architecture":       "gptoss",
		"gptoss.feed_forward_length": uint32(4096),
	}, []*ggml.Tensor{
		fixtureTensor("blk.0.ffn_gate_exps.weight", ggml.TensorTypeF16, []uint64{32, 16, 2}),
	})

	result, err := (gptossMigrator{}).Migrate(src)
	if err != nil {
		t.Fatalf("Migrate() error = %v", err)
	}
	if got := result.ModelKV["gpt-oss.expert_feed_forward_length"]; got != uint32(16) {
		t.Fatalf("expert_feed_forward_length = %v, want 16", got)
	}
}

func fixtureSourceModel(t *testing.T, kv ggml.KV, tensors []*ggml.Tensor) *SourceModel {
	t.Helper()

	f, err := os.CreateTemp(t.TempDir(), "source-*.gguf")
	if err != nil {
		t.Fatalf("CreateTemp() error = %v", err)
	}
	defer f.Close()

	if err := ggml.WriteGGUF(f, kv, tensors); err != nil {
		t.Fatalf("WriteGGUF() error = %v", err)
	}

	g, err := gguf.Open(f.Name())
	if err != nil {
		t.Fatalf("gguf.Open(%s) error = %v", f.Name(), err)
	}
	t.Cleanup(func() { g.Close() })

	return &SourceModel{
		GGUFPath:       f.Name(),
		GGUF:           g,
		GGUFData:       g.ReaderAt(),
		GGUFDataOffset: g.TensorDataOffset(),
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
		kv[keyValue.Key] = normalizeGGUFValue(keyValue.Any())
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

func overrideAvailableSpace(t *testing.T, fn func(string) (uint64, error)) {
	t.Helper()

	old := availableSpaceForPath
	availableSpaceForPath = fn
	t.Cleanup(func() { availableSpaceForPath = old })
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
	if err := manifest.FillMetadata(&child); err != nil {
		t.Fatalf("FillMetadata(child) error = %v", err)
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
	adapter          string
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
	if input.adapter != "" {
		layer, err := manifest.NewLayer(strings.NewReader(input.adapter), "application/vnd.ollama.image.adapter")
		if err != nil {
			t.Fatalf("manifest.NewLayer(adapter) error = %v", err)
		}
		layers = append(layers, layer)
	}

	configLayer, err := newConfigLayer(input.config)
	if err != nil {
		t.Fatalf("newConfigLayer() error = %v", err)
	}
	if err := manifest.WriteManifest(name, configLayer, layers); err != nil {
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
