//go:build mlx

package gemma3

import (
	"math"

	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/nn"
)

// VisionConfig holds configuration for the SigLIP vision tower
type VisionConfig struct {
	HiddenSize        int32 `json:"hidden_size"`
	ImageSize         int32 `json:"image_size"`
	IntermediateSize  int32 `json:"intermediate_size"`
	NumAttentionHeads int32 `json:"num_attention_heads"`
	NumHiddenLayers   int32 `json:"num_hidden_layers"`
	PatchSize         int32 `json:"patch_size"`
}

// VisionTower is the SigLIP vision encoder
type VisionTower struct {
	Embeddings    *VisionEmbeddings     `weight:"vision_model.embeddings"`
	Encoder       []*VisionEncoderLayer `weight:"vision_model.encoder.layers"`
	PostLayerNorm *nn.LayerNorm         `weight:"vision_model.post_layernorm"`
	Config        *VisionConfig
}

// VisionEmbeddings handles patch and position embeddings
type VisionEmbeddings struct {
	// PatchWeight: [O, C, kH, kW] from PyTorch, transposed to [O, kH, kW, C] for MLX
	PatchWeight *mlx.Array    `weight:"patch_embedding.weight"`
	PatchBias   *mlx.Array    `weight:"patch_embedding.bias"`
	PosEmbed    *nn.Embedding `weight:"position_embedding"`
}

// VisionEncoderLayer is a single transformer encoder layer
type VisionEncoderLayer struct {
	LayerNorm1 *nn.LayerNorm     `weight:"layer_norm1"`
	Attention  *VisionAttention  `weight:"self_attn"`
	LayerNorm2 *nn.LayerNorm     `weight:"layer_norm2"`
	MLP        *VisionMLP        `weight:"mlp"`
}

// VisionAttention implements multi-head self-attention
type VisionAttention struct {
	QProj   *nn.Linear `weight:"q_proj"`
	KProj   *nn.Linear `weight:"k_proj"`
	VProj   *nn.Linear `weight:"v_proj"`
	OutProj *nn.Linear `weight:"out_proj"`
}

// VisionMLP is the feed-forward network
type VisionMLP struct {
	FC1 *nn.Linear `weight:"fc1"`
	FC2 *nn.Linear `weight:"fc2"`
}

// Forward runs the vision tower on preprocessed images
// Input: [B, H, W, C] normalized image tensor (NHWC layout for MLX)
// Output: [B, num_patches, hidden_size]
func (v *VisionTower) Forward(x *mlx.Array) *mlx.Array {
	// Patch embedding conv: input [B, H, W, C], weight [O, kH, kW, C] -> [B, grid, grid, O]
	// Weight comes as [O, C, kH, kW] from PyTorch, transpose to [O, kH, kW, C]
	weight := mlx.Transpose(v.Embeddings.PatchWeight, 0, 2, 3, 1)
	h := mlx.Conv2d(x, weight, v.Config.PatchSize, 0) // stride=patch_size, no padding

	// Add bias: [O] -> [1, 1, 1, O] for broadcasting
	bias := mlx.Reshape(v.Embeddings.PatchBias, 1, 1, 1, v.Embeddings.PatchBias.Shape()[0])
	h = mlx.Add(h, bias)

	// h is [B, grid, grid, hidden], flatten to [B, num_patches, hidden]
	B := h.Shape()[0]
	gridH, gridW := h.Shape()[1], h.Shape()[2]
	hidden := h.Shape()[3]
	numPatches := gridH * gridW
	h = mlx.Reshape(h, B, numPatches, hidden)

	// Add position embeddings
	posIds := mlx.ArangeInt(0, numPatches, 1, mlx.DtypeInt32)
	posEmbed := v.Embeddings.PosEmbed.Forward(posIds)
	h = mlx.Add(h, posEmbed)

	// Encoder layers
	headDim := float32(v.Config.HiddenSize / v.Config.NumAttentionHeads)
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	for _, layer := range v.Encoder {
		h = layer.Forward(h, v.Config, scale)
	}

	// Final layer norm
	h = v.PostLayerNorm.Forward(h)

	return h
}

// Forward runs a vision encoder layer
func (l *VisionEncoderLayer) Forward(x *mlx.Array, cfg *VisionConfig, scale float32) *mlx.Array {
	// Pre-norm attention
	h := l.LayerNorm1.Forward(x)
	h = l.Attention.Forward(h, cfg, scale)
	x = mlx.Add(x, h)

	// Pre-norm MLP
	h = l.LayerNorm2.Forward(x)
	h = l.MLP.Forward(h)
	return mlx.Add(x, h)
}

// Forward runs multi-head self-attention
func (a *VisionAttention) Forward(x *mlx.Array, cfg *VisionConfig, scale float32) *mlx.Array {
	B, L := x.Shape()[0], x.Shape()[1]
	headDim := cfg.HiddenSize / cfg.NumAttentionHeads

	q := a.QProj.Forward(x)
	k := a.KProj.Forward(x)
	v := a.VProj.Forward(x)

	// Reshape to [B, num_heads, L, head_dim]
	q = mlx.Transpose(mlx.Reshape(q, B, L, cfg.NumAttentionHeads, headDim), 0, 2, 1, 3)
	k = mlx.Transpose(mlx.Reshape(k, B, L, cfg.NumAttentionHeads, headDim), 0, 2, 1, 3)
	v = mlx.Transpose(mlx.Reshape(v, B, L, cfg.NumAttentionHeads, headDim), 0, 2, 1, 3)

	// Scaled dot-product attention (no causal mask for vision)
	out := mlx.ScaledDotProductAttention(q, k, v, scale, false)

	// Reshape back: [B, num_heads, L, head_dim] -> [B, L, hidden]
	out = mlx.Reshape(mlx.Transpose(out, 0, 2, 1, 3), B, L, cfg.HiddenSize)

	return a.OutProj.Forward(out)
}

// Forward runs the MLP with GELU activation
func (m *VisionMLP) Forward(x *mlx.Array) *mlx.Array {
	h := mlx.GELU(m.FC1.Forward(x))
	return m.FC2.Forward(h)
}
