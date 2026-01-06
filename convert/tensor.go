package convert

import (
	"cmp"
	"errors"
	"io"
	"iter"
	"path"
	"slices"
	"strconv"
	"strings"

	"github.com/pdevine/tensor"
	"github.com/pdevine/tensor/native"

	"github.com/ollama/ollama/fs/ggml"
)

type split struct {
	*strings.Replacer
	dim    int
	slices []tensor.Slice

	// afterFunc is an optional function to apply to the tensor after slicing
	afterFunc func(tensor.Tensor) (tensor.Tensor, error)
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

			slice := split.slices
			if len(slice) == 0 {
				slice = slices.Repeat([]tensor.Slice{nil}, len(shape))
				slice[dim] = tensor.S(offset, offset+int(shape[dim]))
				offset += int(shape[dim])
			}

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

				if split.afterFunc != nil {
					tt, err = split.afterFunc(tt)
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

type merge struct {
	pattern, name string
}

// mergeTensors merges tensors that match a given pattern into a single tensor.
func mergeTensors(unmatched []Tensor, merges ...merge) (out []*ggml.Tensor, _ []Tensor) {
	var matched []Tensor
	for i := range merges {
		matched, unmatched = slicesSplitFunc(unmatched, func(t Tensor) bool {
			matched, _ := path.Match(merges[i].pattern, t.Name())
			return matched
		})

		slices.SortStableFunc(matched, func(a, b Tensor) int {
			x := strings.Split(a.Name(), ".")
			y := strings.Split(b.Name(), ".")
			if len(x) != len(y) {
				return cmp.Compare(len(x), len(y))
			}

			vals := make([]int, len(x))
			for i := range x {
				vals[i] = strings.Compare(x[i], y[i])
				m, err := strconv.ParseInt(x[i], 0, 0)
				n, err2 := strconv.ParseInt(y[i], 0, 0)
				if errors.Join(err, err2) == nil {
					vals[i] = cmp.Compare(m, n)
				}
			}

			return cmp.Or(vals...)
		})

		if len(matched) > 0 {
			out = append(out, &ggml.Tensor{
				Name:     merges[i].name,
				Kind:     matched[0].Kind(),
				Shape:    append([]uint64{uint64(len(matched))}, matched[0].Shape()...),
				WriterTo: mergeGroup(matched),
			})
		}
	}

	return out, unmatched
}

// slicesSplitFunc splits a slice into two slices based on a predicate function.
func slicesSplitFunc[S ~[]E, E comparable](s S, fn func(e E) bool) (matched, unmatched S) {
	for _, e := range s {
		if fn(e) {
			matched = append(matched, e)
		} else {
			unmatched = append(unmatched, e)
		}
	}

	return matched, unmatched
}

type mergeGroup []Tensor

func (g mergeGroup) WriteTo(w io.Writer) (int64, error) {
	for _, t := range g {
		if _, err := t.WriteTo(w); err != nil {
			return 0, err
		}
	}

	return 0, nil
}
