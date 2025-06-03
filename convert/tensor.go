package convert

import (
	"cmp"
	"iter"
	"slices"
	"strings"

	"github.com/pdevine/tensor"
	"github.com/pdevine/tensor/native"

	"github.com/ollama/ollama/fs/ggml"
)

type split struct {
	*strings.Replacer
	dim int

	// fn is an optional function to apply to the tensor after slicing
	fn func(tensor.Tensor) (tensor.Tensor, error)
}

// splitDim splits a tensor along a specified dimension into multiple tensors. The dimension
// is split evenly based on the number of replacers provided unless a specific count is given.
func splitDim(t Tensor, dim int, splits ...split) iter.Seq[*ggml.Tensor] {
	return func(yield func(*ggml.Tensor) bool) {
		var offset int
		for _, split := range splits {
			t := t.Clone()
			shape := slices.Clone(t.Shape())
			shape[dim] = cmp.Or(uint64(split.dim), shape[dim]/uint64(len(splits)))

			slice := slices.Repeat([]tensor.Slice{nil}, len(shape))
			slice[dim] = tensor.S(offset, offset+int(shape[dim]))
			offset += int(shape[dim])

			t.SetRepacker(func(_ string, data []float32, shape []uint64) ([]float32, error) {
				dims := make([]int, len(shape))
				for i := range shape {
					dims[i] = int(shape[i])
				}

				var tt tensor.Tensor = tensor.New(tensor.WithShape(dims...), tensor.WithBacking(data))
				tt, err := tt.Slice(slice...)
				if err != nil {
					return nil, err
				}

				tt = tensor.Materialize(tt)

				if split.fn != nil {
					tt, err = split.fn(tt)
					if err != nil {
						return nil, err
					}
				}

				// flatten tensor so it can be written as a vector
				if err := tt.Reshape(tt.Shape().TotalSize()); err != nil {
					return nil, err
				}

				return native.VectorF32(tt.(*tensor.Dense))
			})

			if !yield(&ggml.Tensor{
				Name:     split.Replace(t.Name()),
				Kind:     t.Kind(),
				Shape:    shape,
				WriterTo: t,
			}) {
				break
			}
		}
	}
}
