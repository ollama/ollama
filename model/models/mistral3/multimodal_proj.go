package mistral3

import (
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
)

type MultiModalProjector struct {
	Norm       *nn.RMSNorm `gguf:"norm"`
	Projection *nn.Linear  `gguf:"projection"`

	spatialMergeSize int
	imageTokenIndex  int
	hasBias          bool
}

func (p *MultiModalProjector) Forward(ctx ml.Context, visionOutputs ml.Tensor, eps float32) ml.Tensor {
	// Apply normalization
	visionOutputs = p.Norm.Forward(ctx, visionOutputs, eps)

	// If the spatial merge size is > 1, average pool the patches
	if p.spatialMergeSize > 1 {
		// Implementation depends on how the model handles spatial merging
		// For simplicity, we'll use a spatial pooling approach
		visionOutputs = visionOutputs.AvgPool2D(ctx, p.spatialMergeSize, p.spatialMergeSize, 0)
	}

	// Project to text embedding dimension
	return p.Projection.Forward(ctx, visionOutputs)
}

func newMultiModalProjector(c ml.Config) *MultiModalProjector {
	return &MultiModalProjector{
		spatialMergeSize: int(c.Uint("spatial_merge_size", 2)),
		imageTokenIndex:  int(c.Uint("image_token_index", 10)),
		hasBias:          c.Bool("mm.projector_bias", false),
	}
}
