package convert

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/ollama/ollama/fs/ggml"
)

func TestConvertEmbeddingGemmaSentenceTransformers(t *testing.T) {
	tempDir := t.TempDir()

	writeJSONFile(t, filepath.Join(tempDir, "config.json"), map[string]any{
		"architectures":               []string{"Gemma3TextModel"},
		"vocab_size":                  uint32(4),
		"max_position_embeddings":     uint32(2048),
		"hidden_size":                 uint32(8),
		"num_hidden_layers":           uint32(1),
		"intermediate_size":           uint32(12),
		"num_attention_heads":         uint32(1),
		"num_key_value_heads":         uint32(1),
		"head_dim":                    uint32(8),
		"rms_norm_eps":                float32(1e-6),
		"rope_theta":                  float32(1000000),
		"rope_local_base_freq":        float32(10000),
		"sliding_window":              uint32(512),
		"use_bidirectional_attention": true,
	})
	writeJSONFile(t, filepath.Join(tempDir, "tokenizer.json"), map[string]any{
		"model": map[string]any{
			"vocab": map[string]int{
				"<pad>": 0,
				"<eos>": 1,
				"<bos>": 2,
				"<unk>": 3,
			},
		},
		"added_tokens": []map[string]any{
			{"id": 4, "content": "<image_soft_token>", "special": true},
		},
	})
	writeJSONFile(t, filepath.Join(tempDir, "modules.json"), []map[string]string{
		{"type": "sentence_transformers.models.Transformer", "path": ""},
		{"type": "sentence_transformers.models.Pooling", "path": "1_Pooling"},
		{"type": "sentence_transformers.models.Dense", "path": "2_Dense"},
		{"type": "sentence_transformers.models.Dense", "path": "3_Dense"},
		{"type": "sentence_transformers.models.Normalize", "path": "4_Normalize"},
	})
	writeJSONFile(t, filepath.Join(tempDir, "1_Pooling", "config.json"), map[string]any{
		"pooling_mode_mean_tokens": true,
	})
	writeJSONFile(t, filepath.Join(tempDir, "2_Dense", "config.json"), map[string]any{
		"in_features":  uint32(8),
		"out_features": uint32(16),
		"bias":         false,
	})
	writeJSONFile(t, filepath.Join(tempDir, "3_Dense", "config.json"), map[string]any{
		"in_features":  uint32(16),
		"out_features": uint32(8),
		"bias":         false,
	})

	writeSafetensorsFile(t, filepath.Join(tempDir, "model.safetensors"), []safetensorFixtureTensor{
		{name: "embed_tokens.weight", shape: []int{4, 8}},
		{name: "norm.weight", shape: []int{8}},
		{name: "layers.0.input_layernorm.weight", shape: []int{8}},
		{name: "layers.0.self_attn.q_proj.weight", shape: []int{8, 8}},
	})
	writeSafetensorsFile(t, filepath.Join(tempDir, "2_Dense", "model.safetensors"), []safetensorFixtureTensor{
		{name: "linear.weight", shape: []int{16, 8}},
	})
	writeSafetensorsFile(t, filepath.Join(tempDir, "3_Dense", "model.safetensors"), []safetensorFixtureTensor{
		{name: "linear.weight", shape: []int{8, 16}},
	})

	f, kv, tensors := convertFull(t, os.DirFS(tempDir))
	defer f.Close()

	if got := kv.Architecture(); got != "gemma-embedding" {
		t.Fatalf("architecture = %q, want gemma-embedding", got)
	}

	for key, want := range map[string]uint32{
		"dense_2_feat_in":          8,
		"dense_2_feat_out":         16,
		"dense_3_feat_in":          16,
		"dense_3_feat_out":         8,
		"pooling_type":             1,
		"attention.sliding_window": 512,
	} {
		if got := kv.Uint(key); got != want {
			t.Errorf("%s = %d, want %d", key, got, want)
		}
	}

	if got := kv.Float("rope.freq_base_swa"); got != 10000 {
		t.Errorf("rope.freq_base_swa = %v, want 10000", got)
	}
	if got := kv.Strings("tokenizer.ggml.tokens"); len(got) != 4 {
		t.Errorf("token count = %d, want 4", len(got))
	}

	names := tensorNames(tensors)
	for _, name := range []string{
		"token_embd.weight",
		"output_norm.weight",
		"blk.0.attn_norm.weight",
		"blk.0.attn_q.weight",
		"dense_2.weight",
		"dense_3.weight",
	} {
		if !slices.Contains(names, name) {
			t.Errorf("missing tensor %s", name)
		}
	}

	assertF32TensorValues(t, f, tensors, "output_norm.weight", 1)
	assertF32TensorValues(t, f, tensors, "blk.0.attn_norm.weight", 1)
}

type safetensorFixtureTensor struct {
	name  string
	shape []int
}

func writeJSONFile(t *testing.T, path string, value any) {
	t.Helper()

	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}

	bts, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}

	if err := os.WriteFile(path, bts, 0o644); err != nil {
		t.Fatal(err)
	}
}

func writeSafetensorsFile(t *testing.T, path string, tensors []safetensorFixtureTensor) {
	t.Helper()

	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}

	offset := 0
	metadata := map[string]*tensorData{}
	for _, tensor := range tensors {
		size := 4
		for _, dim := range tensor.shape {
			size *= dim
		}

		metadata[tensor.name] = &tensorData{
			Offsets: []int{offset, offset + size},
			Type:    "F32",
			Shape:   tensor.shape,
		}
		offset += size
	}

	header, err := json.Marshal(metadata)
	if err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	if err := binary.Write(&buf, binary.LittleEndian, int64(len(header))); err != nil {
		t.Fatal(err)
	}
	if _, err := buf.Write(header); err != nil {
		t.Fatal(err)
	}
	if _, err := buf.Write(make([]byte, offset)); err != nil {
		t.Fatal(err)
	}

	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}

func tensorNames(tensors ggml.Tensors) []string {
	names := make([]string, 0, len(tensors.Items()))
	for _, tensor := range tensors.Items() {
		names = append(names, tensor.Name)
	}
	return names
}

func assertF32TensorValues(t *testing.T, f *os.File, tensors ggml.Tensors, name string, want float32) {
	t.Helper()

	var tensor *ggml.Tensor
	for _, item := range tensors.Items() {
		if item.Name == name {
			tensor = item
			break
		}
	}
	if tensor == nil {
		t.Fatalf("missing tensor %s", name)
	}
	if tensor.Kind != uint32(ggml.TensorTypeF32) {
		t.Fatalf("%s kind = %d, want F32", name, tensor.Kind)
	}

	bts := make([]byte, tensor.Size())
	reader := io.NewSectionReader(f, int64(tensors.Offset+tensor.Offset), int64(tensor.Size()))
	if _, err := io.ReadFull(reader, bts); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < len(bts); i += 4 {
		if got := math.Float32frombits(binary.LittleEndian.Uint32(bts[i:])); got != want {
			t.Fatalf("%s[%d] = %v, want %v", name, i/4, got, want)
		}
	}
}
