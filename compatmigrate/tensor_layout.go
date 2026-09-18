package compatmigrate

import (
	"fmt"
	"slices"
)

// Row-major float32 helpers for the repackers; dims[0] is the outermost
// dimension, matching the tensor library they replaced.

func tensorDims(shape []uint64) []int {
	dims := make([]int, len(shape))
	for i, d := range shape {
		dims[i] = int(d)
	}
	return dims
}

func tensorSize(dims []int) int {
	n := 1
	for _, d := range dims {
		n *= d
	}
	return n
}

func checkTensorDims(data []float32, dims []int) error {
	for _, d := range dims {
		if d < 0 {
			return fmt.Errorf("invalid tensor dims %v", dims)
		}
	}
	if want := tensorSize(dims); len(data) != want {
		return fmt.Errorf("tensor has %d elements, expected %d for dims %v", len(data), want, dims)
	}
	return nil
}

// sliceTensorDim returns elements [start, end) along dim as a new contiguous
// tensor whose dims[dim] is end-start.
func sliceTensorDim(data []float32, dims []int, dim, start, end int) ([]float32, error) {
	if err := checkTensorDims(data, dims); err != nil {
		return nil, err
	}
	if dim < 0 || dim >= len(dims) {
		return nil, fmt.Errorf("slice dimension %d out of range for dims %v", dim, dims)
	}
	if start < 0 || end < start || end > dims[dim] {
		return nil, fmt.Errorf("slice [%d:%d] out of range for dimension %d of dims %v", start, end, dim, dims)
	}

	outer := tensorSize(dims[:dim])
	inner := tensorSize(dims[dim+1:])
	width := (end - start) * inner
	out := make([]float32, 0, outer*width)
	for o := range outer {
		base := (o*dims[dim] + start) * inner
		out = append(out, data[base:base+width]...)
	}
	return out, nil
}

type tensorPart struct {
	data []float32
	dims []int
}

// concatTensorDim joins parts along dim; every other dimension must match.
func concatTensorDim(dim int, parts ...tensorPart) ([]float32, []int, error) {
	if len(parts) == 0 {
		return nil, nil, fmt.Errorf("concat: no tensors")
	}
	dims := slices.Clone(parts[0].dims)
	if dim < 0 || dim >= len(dims) {
		return nil, nil, fmt.Errorf("concat dimension %d out of range for dims %v", dim, dims)
	}
	dims[dim] = 0
	for _, p := range parts {
		if err := checkTensorDims(p.data, p.dims); err != nil {
			return nil, nil, err
		}
		if len(p.dims) != len(dims) {
			return nil, nil, fmt.Errorf("concat: rank mismatch %v vs %v", parts[0].dims, p.dims)
		}
		for i := range dims {
			if i != dim && p.dims[i] != dims[i] {
				return nil, nil, fmt.Errorf("concat: incompatible dims %v vs %v", parts[0].dims, p.dims)
			}
		}
		dims[dim] += p.dims[dim]
	}

	outer := tensorSize(dims[:dim])
	inner := tensorSize(dims[dim+1:])
	out := make([]float32, 0, tensorSize(dims))
	for o := range outer {
		for _, p := range parts {
			width := p.dims[dim] * inner
			out = append(out, p.data[o*width:(o+1)*width]...)
		}
	}
	return out, dims, nil
}

// permuteTensor reorders axes so result dimension i is source dimension
// perm[i], like numpy.transpose.
func permuteTensor(data []float32, dims []int, perm ...int) ([]float32, []int, error) {
	if err := checkTensorDims(data, dims); err != nil {
		return nil, nil, err
	}
	if len(perm) != len(dims) {
		return nil, nil, fmt.Errorf("permutation %v does not match dims %v", perm, dims)
	}

	srcStrides := make([]int, len(dims))
	stride := 1
	for i := len(dims) - 1; i >= 0; i-- {
		srcStrides[i] = stride
		stride *= dims[i]
	}

	seen := make([]bool, len(dims))
	outDims := make([]int, len(dims))
	strides := make([]int, len(dims))
	for i, p := range perm {
		if p < 0 || p >= len(dims) || seen[p] {
			return nil, nil, fmt.Errorf("invalid permutation %v for dims %v", perm, dims)
		}
		seen[p] = true
		outDims[i] = dims[p]
		strides[i] = srcStrides[p]
	}

	out := make([]float32, len(data))
	idx := make([]int, len(dims))
	src := 0
	for i := range out {
		out[i] = data[src]
		for axis := len(idx) - 1; axis >= 0; axis-- {
			idx[axis]++
			src += strides[axis]
			if idx[axis] < outDims[axis] {
				break
			}
			src -= idx[axis] * strides[axis]
			idx[axis] = 0
		}
	}
	return out, outDims, nil
}
