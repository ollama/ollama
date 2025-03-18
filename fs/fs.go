package fs

import (
	"fmt"
	"io"
	"log/slog"
	"os"

	"github.com/ollama/ollama/fs/ggml"
)

type DType int

type Model struct {
	KV      Config
	Tensors map[string]TensorReader
}

func (m Model) LogValue() slog.Value {
	return slog.GroupValue(
		slog.String("architecture", m.KV.Architecture()),
	)
}

type Tensor interface {
	Name() string
	Shape() []int
	DType() DType
	Size() int
}

type TensorReader interface {
	Tensor
	io.Reader
}

type shimTensorReader struct {
	internal *ggml.Tensor
	*io.SectionReader
}

func (t *shimTensorReader) Name() string {
	return t.internal.Name
}

func (t *shimTensorReader) Shape() []int {
	shape := make([]int, len(t.internal.Shape))
	for i, s := range t.internal.Shape {
		shape[i] = int(s)
	}

	return shape
}

func (t *shimTensorReader) Size() int {
	return int(t.internal.Size())
}

func (t *shimTensorReader) DType() DType {
	return DType(t.internal.Kind)
}

func ReadFrom(f *os.File) (*Model, error) {
	bts, err := io.ReadAll(io.NewSectionReader(f, 0, 4))
	if err != nil {
		return nil, err
	}

	switch ggml.DetectContentType(bts[:4]) {
	case "gguf":
		c, _, err := ggml.Decode(f, -1)
		if err != nil {
			return nil, err
		}

		tensors := make(map[string]TensorReader, len(c.Tensors().Items()))
		for _, t := range c.Tensors().Items() {
			tensors[t.Name] = &shimTensorReader{
				internal:      t,
				SectionReader: io.NewSectionReader(f, int64(c.Tensors().Offset+t.Offset), int64(t.Size())),
			}
		}

		return &Model{KV: c.KV(), Tensors: tensors}, nil
	default:
		return nil, fmt.Errorf("unsupported file type")
	}
}
