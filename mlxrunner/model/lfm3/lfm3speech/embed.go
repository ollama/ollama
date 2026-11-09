package lfm3speech

import (
	"fmt"

	"github.com/ollama/ollama/mlx"
)

// CodeEmbedding maps a frame of codes to the backbone input that follows it:
// the sum of one row per codebook from a table shared by all codebooks.
type CodeEmbedding struct {
	Weight       *mlx.Array // [codebooks*codebookSize, hidden]
	numCodebooks int
	codebookSize int
}

// LoadCodeEmbedding reads the feedback table stored under name.
func LoadCodeEmbedding(tensors map[string]*mlx.Array, name string, cfg *Config) (*CodeEmbedding, error) {
	w := tensors[name]
	if w == nil {
		return nil, fmt.Errorf("missing %s", name)
	}
	if err := requireDense(tensors, name); err != nil {
		return nil, err
	}
	if w.NumDims() != 2 || w.Dim(0) != cfg.NumCodebooks*cfg.CodebookSize {
		return nil, fmt.Errorf("%s has shape %v, want [%d, hidden]", name, w.Dims(), cfg.NumCodebooks*cfg.CodebookSize)
	}
	return &CodeEmbedding{Weight: w, numCodebooks: cfg.NumCodebooks, codebookSize: cfg.CodebookSize}, nil
}

// Forward embeds frames, returning [len(frames), hidden] in the table dtype.
// EOS frames embed like any other: the publisher feeds them back too.
func (e *CodeEmbedding) Forward(frames ...[]int32) (*mlx.Array, error) {
	if len(frames) == 0 {
		return nil, fmt.Errorf("no frames to embed")
	}
	rows := make([]int32, 0, len(frames)*e.numCodebooks)
	for _, frame := range frames {
		if len(frame) != e.numCodebooks {
			return nil, fmt.Errorf("frame has %d codes, want %d", len(frame), e.numCodebooks)
		}
		for k, c := range frame {
			if c < 0 || int(c) >= e.codebookSize {
				return nil, fmt.Errorf("code %d out of range [0, %d)", c, e.codebookSize)
			}
			rows = append(rows, int32(k*e.codebookSize)+c)
		}
	}
	x := mlx.Take(e.Weight, mlx.FromValues(rows, len(frames), e.numCodebooks), 0)
	// One rounding after a float32 sum, as torch reduces bfloat16.
	return mlx.Sum(x.AsType(mlx.DTypeFloat32), 1, false).AsType(e.Weight.DType()), nil
}

// requireDense refuses quantized weights: these components stay unquantized.
func requireDense(tensors map[string]*mlx.Array, weight string) error {
	if tensors[weight+"_scale"] != nil {
		return fmt.Errorf("%s is quantized; LFM3 speech components must stay unquantized", weight)
	}
	return nil
}
