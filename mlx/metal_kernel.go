package mlx

import (
	"maps"
	"slices"
)

// MetalKernel is a single-output custom Metal kernel defined outside this
// package. Without Metal, Run computes the output with the fallback instead.
type MetalKernel struct {
	kernel gpuKernel
}

// NewMetalKernel defines a kernel whose source reads the named inputs and
// writes out.
func NewMetalKernel(name string, inputs []string, source string, fallback func(inputs []*Array) *Array) *MetalKernel {
	return &MetalKernel{kernel: gpuKernel{
		name:     name,
		inputs:   inputs,
		outputs:  []string{"out"},
		metal:    gpuSource{source: source},
		fallback: func(launch gpuLaunch) []*Array { return []*Array{fallback(launch.inputs)} },
	}}
}

// Run launches the kernel with integer template arguments over a grid of
// threads and returns out with the given shape and dtype.
func (k *MetalKernel) Run(inputs []*Array, ints map[string]int, grid, threadGroup [3]int, shape []int32, dtype DType) *Array {
	launch := gpuLaunch{
		outputs:     []gpuOutputSpec{{"out", shape, dtype}},
		grid:        grid,
		threadGroup: threadGroup,
		inputs:      inputs,
	}
	for _, name := range slices.Sorted(maps.Keys(ints)) {
		launch.ints = append(launch.ints, gpuIntArg{name, ints[name]})
	}
	return k.kernel.run(launch)[0]
}
