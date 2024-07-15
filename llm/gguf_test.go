package llm

import (
	"io"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestGGUFRewrite(t *testing.T) {
	tests := []string{
		"glm2.gguf",
		"nutiny.gguf",
	}

	for i := range tests {
		tt := tests[i]
		t.Run(tt, func(t *testing.T) {
			t.Parallel()
			p := filepath.Join("testdata", tt)

			if _, err := os.Stat(p); err != nil {
				t.Fatalf("%s not found", p)
			}

			ggml, err := decodeGGML(t, p)
			if err != nil {
				t.Fatal(err)
			}

			ggml2, err := rewriteGGML(t, ggml, p)
			if err != nil {
				t.Fatal(err)
			}

			if cmp.Diff(ggml, ggml2) != "" {
				t.Fatal(cmp.Diff(ggml, ggml2))
			}
		})
	}
}

func decodeGGML(t *testing.T, p string) (*GGML, error) {
	f, err := os.Open(p)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	ggml, _, err := DecodeGGML(f, math.MaxInt)
	if err != nil {
		t.Fatal(err)
	}
	return ggml, nil
}

func rewriteGGML(t *testing.T, ggml *GGML, path string) (*GGML, error) {
	var tensors Tensors
	temp, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer temp.Close()

	for _, tensor := range ggml.Tensors() {
		shape := make([]uint64, len(tensor.Shape))
		for i := range len(tensor.Shape) {
			shape[i] = tensor.Shape[len(tensor.Shape)-i-1]
		}

		tensors = append(tensors, &Tensor{
			Name:  tensor.Name,
			Kind:  tensor.Kind,
			Shape: shape,

			WriterTo: TensorWriter{
				Reader: io.NewSectionReader(temp, int64(tensor.Offset), int64(tensor.Size())),
			},
		})
	}

	reader := &GGUFWriter{
		KV: ggml.KV(),
		// Update .Tensors
		Tensors: tensors,
	}

	_, err = io.Copy(temp, reader)
	if err != nil {
		t.Fatal(err)
	}

	ggml2, _, err := DecodeGGML(temp, -1)
	if err != nil {
		t.Fatal(err)
	}

	return ggml2, nil
}
