package server

import (
	"bytes"
	"crypto/sha256"
	"fmt"
	"os"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

const maxCreateGGUFFuzzBytes = 1 << 16

func FuzzConvertModelFromFiles(f *testing.F) {
	valid := ggufBytesForFuzz(f, ggml.KV{
		"general.architecture": "llama",
		"general.file_type":    uint32(ggml.FileTypeF32),
	}, "blk.0.attn_q.weight")
	split0 := ggufBytesForFuzz(f, ggml.KV{
		"general.architecture":      "llama",
		"general.file_type":         uint32(ggml.FileTypeF32),
		"llama.split.no":            uint32(0),
		"llama.split.count":         uint32(2),
		"llama.split.tensors.count": int32(2),
	}, "blk.0.attn_q.weight")
	split1 := ggufBytesForFuzz(f, ggml.KV{
		"general.architecture":      "llama",
		"general.file_type":         uint32(ggml.FileTypeF32),
		"llama.split.no":            uint32(1),
		"llama.split.count":         uint32(2),
		"llama.split.tensors.count": int32(2),
	}, "blk.1.attn_q.weight")

	f.Add(uint8(0), valid, []byte{})
	f.Add(uint8(1), []byte("GGUF"), []byte{})
	f.Add(uint8(2), []byte{}, []byte{})
	f.Add(uint8(3), valid, []byte("not a safetensors file"))
	f.Add(uint8(4), split0, split1)
	f.Add(uint8(5), split0, split0)

	f.Fuzz(func(t *testing.T, mode uint8, first, second []byte) {
		if len(first) > maxCreateGGUFFuzzBytes || len(second) > maxCreateGGUFFuzzBytes {
			t.Skip("bounded fuzz input")
		}
		t.Setenv("OLLAMA_MODELS", t.TempDir())

		files := fuzzCreateFiles(t, mode, first, second)
		if err := validateCreateFiles(files); err != nil {
			return
		}

		layers, err := convertModelFromFiles(files, nil, false, func(api.ProgressResponse) {})
		if err != nil {
			return
		}
		if len(layers) > len(files) {
			t.Fatalf("convertModelFromFiles returned %d layers for %d files", len(layers), len(files))
		}

		name := model.ParseName("fuzz-create-gguf:latest")
		config := &model.ConfigV2{
			OS:           "linux",
			Architecture: "amd64",
		}
		_ = createModel(api.CreateRequest{Model: name.String()}, name, layers, config, func(api.ProgressResponse) {})
	})
}

func fuzzCreateFiles(t *testing.T, mode uint8, first, second []byte) map[string]string {
	t.Helper()

	firstDigest := writeFuzzBlob(t, first)
	secondDigest := writeFuzzBlob(t, second)

	switch mode % 6 {
	case 0:
		return map[string]string{"model.gguf": firstDigest}
	case 1:
		return map[string]string{"model": firstDigest}
	case 2:
		return map[string]string{"model.safetensors": firstDigest}
	case 3:
		return map[string]string{
			"model.gguf":        firstDigest,
			"model.safetensors": secondDigest,
		}
	case 4:
		return map[string]string{
			"model-00001-of-00002.gguf": firstDigest,
			"model-00002-of-00002.gguf": secondDigest,
		}
	default:
		return map[string]string{
			"model.gguf":   firstDigest,
			"projector":    secondDigest,
			"mmproj.gguf":  secondDigest,
			"nested/model": firstDigest,
		}
	}
}

func ggufBytesForFuzz(tb testing.TB, kv ggml.KV, tensorName string) []byte {
	tb.Helper()

	f, err := os.CreateTemp(tb.TempDir(), "")
	if err != nil {
		tb.Fatal(err)
	}
	defer f.Close()

	if err := ggml.WriteGGUF(f, kv, []*ggml.Tensor{
		{
			Name:     tensorName,
			Kind:     uint32(ggml.TensorTypeF32),
			Shape:    []uint64{1, 1},
			WriterTo: bytes.NewReader(make([]byte, 4)),
		},
	}); err != nil {
		tb.Fatal(err)
	}

	if _, err := f.Seek(0, 0); err != nil {
		tb.Fatal(err)
	}
	data, err := os.ReadFile(f.Name())
	if err != nil {
		tb.Fatal(err)
	}
	return data
}

func writeFuzzBlob(t *testing.T, data []byte) string {
	t.Helper()

	sum := sha256.Sum256(data)
	digest := fmt.Sprintf("sha256:%x", sum)
	blobPath, err := manifest.BlobsPath(digest)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(blobPath, data, 0o644); err != nil {
		t.Fatal(err)
	}
	return digest
}
