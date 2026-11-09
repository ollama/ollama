package phonon

import (
	"fmt"
	"slices"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/nn"
)

// The published checkpoints sum the default speaker count (one) and zero
// duration adjustment into every generated frame. Their voice and text
// prefixes are run through the transformer separately.
func gradiumConditions(tensors map[string]*mlx.Array) (*mlx.Array, error) {
	const prefix = "flow_lm.condition_provider.conditioners."
	numEmb := tensors[prefix+"num_speakers.embed.weight"]
	numProj := tensors[prefix+"num_speakers.output_proj.weight"]
	durationProj := tensors[prefix+"duration_delta.output_proj.weight"]
	if numEmb == nil || numProj == nil || durationProj == nil ||
		!slices.Equal(numEmb.Dims(), []int{32, 16}) ||
		!slices.Equal(numProj.Dims(), []int{512, 16}) ||
		!slices.Equal(durationProj.Dims(), []int{512, 128}) {
		return nil, fmt.Errorf("missing or invalid Gradium Phonon conditioners")
	}
	index := mlx.FromValues([]int32{1}, 1)
	num := mlx.ExpandDims(mlx.Take(numEmb, index, 0), 1)
	num = nn.NewLinear(numProj, nil).Forward(num)
	// The duration delta defaults to zero. A sinusoidal embedding at zero has
	// ones in its cosine half and zeros in its sine half.
	duration := make([]float32, 128)
	for i := range 64 {
		duration[i] = 1
	}
	delta := mlx.FromValues(duration, 1, 1, len(duration)).AsType(durationProj.DType())
	delta = nn.NewLinear(durationProj, nil).Forward(delta)
	return mlx.Add(num, delta).AsType(mlx.DTypeBFloat16), nil
}
