package ggml

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	fsgguf "github.com/ollama/ollama/fs/gguf"
)

func TestDecodeQ2_0(t *testing.T) {
	for _, tc := range []struct {
		name        string
		width, rows uint64
		size        int
		wantErr     string
	}{
		{"one_block", 64, 1, 18, ""},
		{"multiple_rows", 128, 3, 108, ""},
		{"invalid_row_alignment", 32, 2, 18, "size overflow"},
		{"truncated_payload", 64, 1, 17, "exceeds file size"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			payload := make([]byte, tc.size)
			for i := range payload {
				payload[i] = byte(i)
			}
			shape := []uint64{tc.width, tc.rows}
			path := filepath.Join(t.TempDir(), "q2_0.gguf")
			w, err := os.Create(path)
			if err != nil {
				t.Fatal(err)
			}
			defer w.Close()
			if err := WriteGGUF(w, KV{"general.architecture": "llama", "general.file_type": uint32(fileTypeQ2_0)}, []*Tensor{{
				Name:     "token_embd.weight",
				Kind:     uint32(TensorTypeQ2_0),
				Shape:    shape,
				WriterTo: bytes.NewReader(payload),
			}}); err != nil {
				t.Fatal(err)
			}
			wantErr := func(t *testing.T, err error) bool {
				t.Helper()
				if tc.wantErr != "" && (err == nil || !strings.Contains(err.Error(), tc.wantErr)) {
					t.Fatalf("error = %v, want %q", err, tc.wantErr)
				}
				return tc.wantErr != ""
			}

			t.Run("ggml", func(t *testing.T) {
				if _, err := w.Seek(0, io.SeekStart); err != nil {
					t.Fatal(err)
				}
				f, err := Decode(w, -1)
				if wantErr(t, err) {
					return
				}
				if err != nil {
					t.Fatal(err)
				}
				if len(f.Tensors().Items()) != 1 {
					t.Fatalf("decoded %d tensors, want 1", len(f.Tensors().Items()))
				}
				tensor := f.Tensors().Items()[0]
				if tensor.Kind != uint32(TensorTypeQ2_0) || tensor.Size() != uint64(tc.size) || !slices.Equal(tensor.Shape, shape) {
					t.Fatalf("unexpected tensor: %+v, size %d", tensor, tensor.Size())
				}
				if got := f.KV().FileType(); got.String() != "Q2_0" || got.ToTensorType() != TensorTypeQ2_0 {
					t.Fatalf("file type = %v, tensor type = %v, want Q2_0", got, got.ToTensorType())
				}
				rd := io.NewSectionReader(w, int64(f.Tensors().Offset+tensor.Offset), int64(tensor.Size()))
				got, err := io.ReadAll(rd)
				if err != nil || !bytes.Equal(got, payload) {
					t.Fatalf("tensor payload = %v, error = %v, want %v", got, err, payload)
				}
			})

			t.Run("gguf", func(t *testing.T) {
				f, err := fsgguf.Open(path)
				if err != nil {
					t.Fatal(err)
				}
				defer f.Close()
				tensor, rd, err := f.TensorReader("token_embd.weight")
				if wantErr(t, err) {
					return
				}
				if err != nil {
					t.Fatal(err)
				}
				if tensor.Type != fsgguf.TensorTypeQ2_0 || tensor.Type.String() != "q2_0" || tensor.NumBytes() != int64(tc.size) || !slices.Equal(tensor.Shape, shape) {
					t.Fatalf("unexpected tensor: %+v, size %d", tensor, tensor.NumBytes())
				}
				got, err := io.ReadAll(rd)
				if err != nil || !bytes.Equal(got, payload) {
					t.Fatalf("tensor payload = %v, error = %v, want %v", got, err, payload)
				}
			})
		})
	}
}
