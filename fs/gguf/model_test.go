package gguf_test

import (
	"bytes"
	"errors"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/fs/gguf"
	gguftest "github.com/ollama/ollama/internal/testutil/gguf"
)

func TestReadModelAggregatesFiles(t *testing.T) {
	primary := writeMetadataFixture(t, gguftest.KV{
		"general.architecture": "llama",
		"general.file_type":    gguf.FileTypeF16,
		"llama.context_length": uint32(4096),
	}, []*gguftest.Tensor{{
		Name:     "blk.0.attn_q.weight",
		Type:     gguf.TensorTypeF32,
		Shape:    []uint64{2},
		WriterTo: bytes.NewReader(make([]byte, 8)),
	}})
	shard := writeMetadataFixture(t, gguftest.KV{
		"general.architecture": "unknown",
		"general.file_type":    gguf.FileTypeUnknown,
	}, []*gguftest.Tensor{{
		Name:     "blk.1.attn_q.weight",
		Type:     gguf.TensorTypeF16,
		Shape:    []uint64{4},
		WriterTo: bytes.NewReader(make([]byte, 8)),
	}})

	model, err := gguf.ReadModel(primary, 0, shard)
	if err != nil {
		t.Fatal(err)
	}
	if got := model.KV().Architecture(); got != "llama" {
		t.Fatalf("architecture = %q, want llama", got)
	}
	if got := model.KV().ContextLength(); got != 4096 {
		t.Fatalf("context length = %d, want 4096", got)
	}
	if got := model.KV().ParameterCount(); got != 6 {
		t.Fatalf("parameter count = %d, want 6", got)
	}
	if got, _ := model.KV().Values()["general.parameter_count"].(uint64); got != 6 {
		t.Fatalf("serialized parameter count = %d, want 6", got)
	}
	if got := len(model.Tensors().Items()); got != 2 {
		t.Fatalf("tensor count = %d, want 2", got)
	}
	if got := len(model.Tensors().Items("blk.1.")); got != 1 {
		t.Fatalf("filtered tensor count = %d, want 1", got)
	}
	if got := model.Tensors().Size(); got != 16 {
		t.Fatalf("tensor size = %d, want 16", got)
	}
	if got := model.Tensors().Size("blk.1."); got != 8 {
		t.Fatalf("filtered tensor size = %d, want 8", got)
	}
	if got, want := model.Files(), []string{primary, shard}; !slices.Equal(got, want) {
		t.Fatalf("files = %q, want %q", got, want)
	}
	var wantFileSize uint64
	for _, path := range []string{primary, shard} {
		info, err := os.Stat(path)
		if err != nil {
			t.Fatal(err)
		}
		wantFileSize += uint64(info.Size())
	}
	if got := model.FileSize(); got != wantFileSize {
		t.Fatalf("file size = %d, want %d", got, wantFileSize)
	}
}

func TestModelTensorItemsAreCopies(t *testing.T) {
	path := writeMetadataFixture(t, gguftest.KV{
		"general.architecture": "llama",
	}, []*gguftest.Tensor{{
		Name:     "weight",
		Type:     gguf.TensorTypeF32,
		Shape:    []uint64{2},
		WriterTo: bytes.NewReader(make([]byte, 8)),
	}})

	model, err := gguf.ReadModel(path, 0)
	if err != nil {
		t.Fatal(err)
	}
	items := model.Tensors().Items()
	items[0].Name = "changed"
	items[0].Shape[0] = 99

	got := model.Tensors().Items()[0]
	if got.Name != "weight" || got.Shape[0] != 2 {
		t.Fatalf("tensor after mutating result = %#v, want original", got)
	}
}

func TestModelFilesAreCopies(t *testing.T) {
	path := writeMetadataFixture(t, gguftest.KV{"general.architecture": "llama"}, nil)
	model, err := gguf.ReadModel(path, 0)
	if err != nil {
		t.Fatal(err)
	}
	files := model.Files()
	files[0] = "changed"
	if got := model.Files()[0]; got != path {
		t.Fatalf("file after mutating result = %q, want %q", got, path)
	}
}

func TestReadModelRejectsInconsistentFiles(t *testing.T) {
	primary := writeMetadataFixture(t, gguftest.KV{
		"general.architecture": "llama",
		"general.file_type":    gguf.FileTypeF16,
	}, []*gguftest.Tensor{{
		Name:     "weight",
		Type:     gguf.TensorTypeF32,
		Shape:    []uint64{1},
		WriterTo: bytes.NewReader(make([]byte, 4)),
	}})

	tests := []struct {
		name   string
		kv     gguftest.KV
		tensor *gguftest.Tensor
		want   string
	}{
		{
			name: "architecture",
			kv: gguftest.KV{
				"general.architecture": "mistral",
				"general.file_type":    gguf.FileTypeF16,
			},
			tensor: &gguftest.Tensor{
				Name:     "other",
				Type:     gguf.TensorTypeF32,
				Shape:    []uint64{1},
				WriterTo: bytes.NewReader(make([]byte, 4)),
			},
			want: "architecture",
		},
		{
			name: "file type",
			kv: gguftest.KV{
				"general.architecture": "llama",
				"general.file_type":    gguf.FileTypeQ4_K_M,
			},
			tensor: &gguftest.Tensor{
				Name:     "other",
				Type:     gguf.TensorTypeF32,
				Shape:    []uint64{1},
				WriterTo: bytes.NewReader(make([]byte, 4)),
			},
			want: "file type",
		},
		{
			name: "duplicate tensor",
			kv: gguftest.KV{
				"general.architecture": "llama",
				"general.file_type":    gguf.FileTypeF16,
			},
			tensor: &gguftest.Tensor{
				Name:     "weight",
				Type:     gguf.TensorTypeF32,
				Shape:    []uint64{1},
				WriterTo: bytes.NewReader(make([]byte, 4)),
			},
			want: "duplicate tensor",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			shard := writeMetadataFixture(t, tt.kv, []*gguftest.Tensor{tt.tensor})
			_, err := gguf.ReadModel(primary, 0, shard)
			if !errors.Is(err, gguf.ErrUnsupported) || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("ReadModel() error = %v, want containing %q", err, tt.want)
			}
		})
	}
}
