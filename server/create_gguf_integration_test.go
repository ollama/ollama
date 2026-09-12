package server

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

func TestCreateModelGGUFValidationWithLlamaQuantize(t *testing.T) {
	// Helper binaries are built relative to the repository root.
	t.Chdir("..")
	if _, err := findLlamaQuantize(); err != nil {
		t.Skipf("requires a built llama-quantize: %v", err)
	}
	oldRun := runLlamaQuantize
	runLlamaQuantize = runLlamaQuantizeCommand
	t.Cleanup(func() { runLlamaQuantize = oldRun })

	for _, tt := range []struct {
		name    string
		value   float32
		wantErr bool
	}{
		{name: "valid", value: 1},
		{name: "nan", value: float32(math.NaN()), wantErr: true},
		{name: "inf", value: float32(math.Inf(1)), wantErr: true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("OLLAMA_MODELS", t.TempDir())
			data := make([]byte, 16)
			for i := 0; i < len(data); i += 4 {
				binary.LittleEndian.PutUint32(data[i:], math.Float32bits(tt.value))
			}
			_, digest := createBinFile(t, ggml.KV{
				"general.architecture":                   "llama",
				"general.file_type":                      uint32(ggml.FileTypeF32),
				"llama.context_length":                   uint32(32),
				"llama.embedding_length":                 uint32(2),
				"llama.block_count":                      uint32(1),
				"llama.feed_forward_length":              uint32(2),
				"llama.attention.head_count":             uint32(1),
				"llama.attention.layer_norm_rms_epsilon": float32(1e-5),
			}, []*ggml.Tensor{{
				Name:     "blk.0.attn_q.weight",
				Kind:     uint32(ggml.TensorTypeF32),
				Shape:    []uint64{2, 2},
				WriterTo: bytes.NewReader(data),
			}})
			layers, err := ggufLayers(digest, "test.gguf", func(api.ProgressResponse) {})
			if err != nil {
				t.Fatal(err)
			}
			blob, err := manifest.BlobsPath(digest)
			if err != nil {
				t.Fatal(err)
			}
			before, err := os.ReadFile(blob)
			if err != nil {
				t.Fatal(err)
			}
			name := model.ParseName("test-create-native:latest")
			err = createModel(api.CreateRequest{Model: name.String()}, name, layers, &model.ConfigV2{}, func(api.ProgressResponse) {})
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected invalid tensor data to be rejected")
				}
				if _, err := manifest.ParseNamedManifest(name); !os.IsNotExist(err) {
					t.Fatalf("expected no manifest for invalid model, got %v", err)
				}
			} else {
				if err != nil {
					t.Fatal(err)
				}
				mf, err := manifest.ParseNamedManifest(name)
				if err != nil {
					t.Fatal(err)
				}
				if len(mf.Layers) != 1 || mf.Layers[0].Digest != digest {
					t.Fatalf("model layers = %+v, want original digest %q", mf.Layers, digest)
				}
			}
			after, err := os.ReadFile(blob)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(before, after) {
				t.Fatal("original GGUF contents changed")
			}
		})
	}
}
