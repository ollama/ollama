//go:build mlx

package gemma3

import (
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/nn"
)

// MultiModalProjector projects vision features to text embedding space
type MultiModalProjector struct {
	// mm_input_projection_weight: [vision_hidden, text_hidden]
	InputProjection *mlx.Array  `weight:"mm_input_projection_weight"`
	SoftEmbNorm     *nn.RMSNorm `weight:"mm_soft_emb_norm"`

	// Precomputed (1 + weight) for Gemma-style RMSNorm
	SoftEmbNormScaled *mlx.Array `weight:"-"`
}

// Forward projects vision features to text space
// Input: [B, num_patches, vision_hidden] (e.g., [1, 4096, 1152])
// Output: [B, num_image_tokens, text_hidden] (e.g., [1, 256, 2560])
func (p *MultiModalProjector) Forward(visionFeatures *mlx.Array, eps float32) *mlx.Array {
	// Average pool 4x4: [B, 4096, 1152] -> [B, 256, 1152]
	// 4096 patches = 64x64 grid, pool to 16x16 = 256 tokens
	B := visionFeatures.Shape()[0]
	visionHidden := visionFeatures.Shape()[2]

	// Reshape to [B, 64, 64, hidden]
	gridSize := int32(64) // sqrt(4096)
	pooledSize := int32(16) // 64/4
	h := mlx.Reshape(visionFeatures, B, gridSize, gridSize, visionHidden)

	// Reshape to [B, 16, 4, 16, 4, hidden] for 4x4 pooling
	h = mlx.Reshape(h, B, pooledSize, 4, pooledSize, 4, visionHidden)

	// Average over pooling dimensions (axes 2 and 4)
	h = mlx.Mean(h, 4, false)
	h = mlx.Mean(h, 2, false)

	// h is now [B, 16, 16, hidden], reshape to [B, 256, hidden]
	numTokens := pooledSize * pooledSize
	h = mlx.Reshape(h, B, numTokens, visionHidden)

	// Apply Gemma-style RMS norm (use precomputed 1 + weight)
	h = mlx.RMSNorm(h, p.SoftEmbNormScaled, eps)

	// Project to text space: [B, 256, vision_hidden] @ [vision_hidden, text_hidden]
	return mlx.Linear(h, p.InputProjection)
}
