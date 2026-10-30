package gguf_test

import (
	"bytes"
	"errors"
	"os"
	"slices"
	"testing"

	"github.com/ollama/ollama/fs/gguf"
	gguftest "github.com/ollama/ollama/internal/testutil/gguf"
)

func writeMetadataFixture(t *testing.T, kv gguftest.KV, tensors []*gguftest.Tensor) string {
	t.Helper()
	f, err := os.CreateTemp(t.TempDir(), "*.gguf")
	if err != nil {
		t.Fatal(err)
	}
	if err := gguftest.Write(f, kv, tensors); err != nil {
		f.Close()
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	return f.Name()
}

func TestReadFileMetadataSkipsLargeArrays(t *testing.T) {
	path := writeMetadataFixture(t, gguftest.KV{
		"general.architecture":          "llama",
		"general.file_type":             gguf.FileTypeQ4_K_M,
		"llama.block_count":             uint32(2),
		"llama.context_length":          uint32(4096),
		"llama.embedding_length":        uint32(128),
		"llama.attention.head_count":    []uint32{4, 8},
		"llama.attention.head_count_kv": []uint32{2, 4},
		"tokenizer.ggml.tokens":         []string{"zero", "one", "two"},
		"tokenizer.ggml.scores":         []float32{0, 1, 2},
		"tokenizer.ggml.model":          "gpt2",
	}, []*gguftest.Tensor{{
		Name:     "blk.0.attn_q.weight",
		Type:     gguf.TensorTypeF32,
		Shape:    []uint64{2, 3},
		WriterTo: bytes.NewReader(make([]byte, 24)),
	}})

	metadata, err := gguf.ReadFileMetadata(path, 2)
	if err != nil {
		t.Fatal(err)
	}
	if got := metadata.KeyValue("tokenizer.ggml.tokens").Strings(); got != nil {
		t.Fatalf("tokens = %v, want omitted", got)
	}
	if got := metadata.KeyValue("tokenizer.ggml.scores").Floats(); got != nil {
		t.Fatalf("scores = %v, want omitted", got)
	}
	if got := metadata.String("tokenizer.ggml.model"); got != "gpt2" {
		t.Fatalf("tokenizer model = %q, want gpt2", got)
	}
	if got := metadata.ContextLength(); got != 4096 {
		t.Fatalf("context length = %d, want 4096", got)
	}
	if got := metadata.HeadCountMax(); got != 8 {
		t.Fatalf("head count max = %d, want 8", got)
	}
	if got := metadata.HeadCountKVMin(); got != 2 {
		t.Fatalf("KV head count min = %d, want 2", got)
	}
	if got := metadata.ParameterCount(); got != 6 {
		t.Fatalf("parameter count = %d, want 6", got)
	}
	if got := metadata.FileType(); got != gguf.FileTypeQ4_K_M {
		t.Fatalf("file type = %v, want %v", got, gguf.FileTypeQ4_K_M)
	}
	if got := metadata.Values()["tokenizer.ggml.tokens"]; len(got.([]any)) != 0 {
		t.Fatalf("serialized tokens = %v, want empty array", got)
	}
	got := metadata.OmittedKeys()
	slices.Sort(got)
	if !slices.Equal(got, []string{"tokenizer.ggml.scores", "tokenizer.ggml.tokens"}) {
		t.Fatalf("omitted keys = %v", got)
	}
	if _, ok := metadata.Values()["general.parameter_count"]; ok {
		t.Fatal("Values contains derived parameter count")
	}
}

func TestReadFileMetadataRetainsAllArrays(t *testing.T) {
	path := writeMetadataFixture(t, gguftest.KV{
		"general.architecture":  "llama",
		"tokenizer.ggml.tokens": []string{"zero", "one", "two"},
	}, nil)

	metadata, err := gguf.ReadFileMetadata(path, -1)
	if err != nil {
		t.Fatal(err)
	}
	if got := metadata.KeyValue("tokenizer.ggml.tokens").Strings(); len(got) != 3 {
		t.Fatalf("tokens = %v, want three entries", got)
	}
}

func TestMetadataHeadCountsDefaultForEmptyArrays(t *testing.T) {
	path := writeMetadataFixture(t, gguftest.KV{
		"general.architecture":          "llama",
		"llama.attention.head_count":    []uint32{},
		"llama.attention.head_count_kv": []int32{},
	}, nil)

	metadata, err := gguf.ReadFileMetadata(path, 0)
	if err != nil {
		t.Fatal(err)
	}
	if got := metadata.HeadCountMax(); got != 1 {
		t.Fatalf("head count max = %d, want 1", got)
	}
	if got := metadata.HeadCountKVMin(); got != 1 {
		t.Fatalf("KV head count min = %d, want 1", got)
	}
}

func TestMetadataUintOKSplitKeys(t *testing.T) {
	path := writeMetadataFixture(t, gguftest.KV{
		"general.architecture": "llama",
		"split.count":          uint32(2),
		"llama.split.no":       int32(1),
	}, nil)

	metadata, err := gguf.ReadFileMetadata(path, 0)
	if err != nil {
		t.Fatal(err)
	}
	if got, ok := metadata.UintOK("split.count"); !ok || got != 2 {
		t.Fatalf("split count = %d, %v; want 2, true", got, ok)
	}
	if got, ok := metadata.UintOK("split.no"); !ok || got != 1 {
		t.Fatalf("split number = %d, %v; want 1, true", got, ok)
	}
}

func TestReadFileMetadataValidatesTensorData(t *testing.T) {
	path := writeMetadataFixture(t, gguftest.KV{
		"general.architecture": "llama",
	}, []*gguftest.Tensor{{
		Name:  "blk.0.attn_q.weight",
		Type:  gguf.TensorTypeF32,
		Shape: []uint64{8},
	}})

	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Truncate(path, info.Size()-32); err != nil {
		t.Fatal(err)
	}
	if _, err := gguf.ReadFileMetadata(path, 0); !errors.Is(err, gguf.ErrUnsupported) {
		t.Fatalf("ReadFileMetadata() error = %v, want ErrUnsupported", err)
	}
}

func TestReadFileMetadataUsesUnsignedAlignment(t *testing.T) {
	path := writeMetadataFixture(t, gguftest.KV{
		"general.architecture": "llama",
		"general.alignment":    uint32(64),
	}, []*gguftest.Tensor{{
		Name:  "blk.0.attn_q.weight",
		Type:  gguf.TensorTypeF32,
		Shape: []uint64{8},
	}})

	if _, err := gguf.ReadFileMetadata(path, 0); err != nil {
		t.Fatal(err)
	}
}
