package convert

import (
	"iter"
	"slices"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
	"github.com/pdevine/tensor"
	"github.com/pdevine/tensor/native"
)

// splitDim splits a tensor along a specified dimension into multiple tensors. The dimension
// is split evenly based on the number of replacers provided.
func splitDim(t Tensor, dim int, replacers ...*strings.Replacer) iter.Seq[ggml.Tensor] {
	return func(yield func(ggml.Tensor) bool) {
		for i, replacer := range replacers {
			shape := slices.Clone(t.Shape())
			shape[dim] = shape[dim] / uint64(len(replacers))

			slice := slices.Repeat([]tensor.Slice{nil}, len(shape))
			slice[dim] = tensor.S(i*int(shape[dim]), (i+1)*int(shape[dim]))

			tt := t.Clone()
			tt.SetRepacker(func(_ string, data []float32, shape []uint64) ([]float32, error) {
				dims := make([]int, len(shape))
				for i := range shape {
					dims[i] = int(shape[i])
				}

				var t tensor.Tensor = tensor.New(tensor.WithShape(dims...), tensor.WithBacking(data))
				t, err := t.Slice(slice...)
				if err != nil {
					return nil, err
				}

				t = tensor.Materialize(t)
				// flatten tensor so it can be written as a vector
				if err := t.Reshape(t.Shape().TotalSize()); err != nil {
					return nil, err
				}

				return native.VectorF32(t.(*tensor.Dense))
			})

			if !yield(ggml.Tensor{
				Name:     replacer.Replace(t.Name()),
				Kind:     t.Kind(),
				Shape:    shape,
				WriterTo: tt,
			}) {
				break
			}
		}
	}
}
