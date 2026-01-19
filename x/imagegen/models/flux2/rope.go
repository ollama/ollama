//go:build mlx

package flux2

import (
	"math"

	"github.com/ollama/ollama/x/imagegen/mlx"
)

// RoPEConfig holds 4D RoPE configuration for Flux2
type RoPEConfig struct {
	Theta    int32   // 2000 for Klein
	AxesDims []int32 // [32, 32, 32, 32] - dimensions for T, H, W, L axes
}

// RoPECache holds precomputed RoPE cos/sin values
type RoPECache struct {
	Cos      *mlx.Array // [1, TotalSeqLen, 1, head_dim/2]
	Sin      *mlx.Array // [1, TotalSeqLen, 1, head_dim/2]
	TextLen  int32      // Length of text sequence
	ImageLen int32      // Length of image sequence
}

// PrepareTextIDs creates position IDs for text tokens.
// Text tokens use: T=0, H=0, W=0, L=0..seqLen-1
// Returns: [seqLen, 4]
func PrepareTextIDs(seqLen int32) *mlx.Array {
	ids := make([]float32, seqLen*4)
	for i := int32(0); i < seqLen; i++ {
		idx := i * 4
		ids[idx+0] = 0             // T = 0
		ids[idx+1] = 0             // H = 0
		ids[idx+2] = 0             // W = 0
		ids[idx+3] = float32(i)    // L = sequence position
	}
	return mlx.NewArray(ids, []int32{seqLen, 4})
}

// PrepareLatentIDs creates position IDs for image latent tokens.
// Latent tokens use: T=0, H=0..height-1, W=0..width-1, L=0
// The latents are in row-major order (H then W).
// Returns: [height*width, 4]
func PrepareLatentIDs(height, width int32) *mlx.Array {
	seqLen := height * width
	ids := make([]float32, seqLen*4)
	idx := 0
	for h := int32(0); h < height; h++ {
		for w := int32(0); w < width; w++ {
			ids[idx*4+0] = 0           // T = 0
			ids[idx*4+1] = float32(h)  // H = row
			ids[idx*4+2] = float32(w)  // W = column
			ids[idx*4+3] = 0           // L = 0
			idx++
		}
	}
	return mlx.NewArray(ids, []int32{seqLen, 4})
}

// PrepareImageIDs creates position IDs for reference image tokens (used in editing).
// Reference images use: T=scale*(i+1), H=0..h-1, W=0..w-1, L=0
// where i is the image index (0, 1, 2, ...) and scale separates images in T dimension.
// Returns: [total_tokens, 4]
func PrepareImageIDs(imageHeights, imageWidths []int32, scale int32) *mlx.Array {
	// Calculate total tokens
	totalTokens := int32(0)
	for i := range imageHeights {
		totalTokens += imageHeights[i] * imageWidths[i]
	}

	ids := make([]float32, totalTokens*4)
	idx := int32(0)
	for imgIdx, h := range imageHeights {
		w := imageWidths[imgIdx]
		tValue := float32(scale * int32(imgIdx+1))
		for hi := int32(0); hi < h; hi++ {
			for wi := int32(0); wi < w; wi++ {
				ids[idx*4+0] = tValue       // T = scale * (imgIdx + 1)
				ids[idx*4+1] = float32(hi)  // H = row
				ids[idx*4+2] = float32(wi)  // W = column
				ids[idx*4+3] = 0            // L = 0
				idx++
			}
		}
	}
	return mlx.NewArray(ids, []int32{totalTokens, 4})
}

// ComputeRoPE computes cos and sin for 4D rotary position embeddings.
// ids: [L, 4] with (T, H, W, L) coordinates
// axesDims: [32, 32, 32, 32] - each axis has this many dimensions (total = head_dim = 128)
// theta: base frequency (2000 for Klein)
// Returns: cos, sin each [1, L, 1, head_dim] with repeat_interleave applied
func ComputeRoPE(ids *mlx.Array, axesDims []int32, theta int32) (*mlx.Array, *mlx.Array) {
	shape := ids.Shape()
	seqLen := shape[0]

	// Compute total head dim (sum of all axes dims)
	headDim := int32(0)
	for _, d := range axesDims {
		headDim += d
	}

	// Extract each coordinate dimension
	// ids[:, 0] = T, ids[:, 1] = H, ids[:, 2] = W, ids[:, 3] = L
	posT := mlx.Slice(ids, []int32{0, 0}, []int32{seqLen, 1}) // [L, 1]
	posH := mlx.Slice(ids, []int32{0, 1}, []int32{seqLen, 2}) // [L, 1]
	posW := mlx.Slice(ids, []int32{0, 2}, []int32{seqLen, 3}) // [L, 1]
	posL := mlx.Slice(ids, []int32{0, 3}, []int32{seqLen, 4}) // [L, 1]

	// Compute frequencies for each axis
	logTheta := float32(math.Log(float64(theta)))
	cosArrs := make([]*mlx.Array, 4)
	sinArrs := make([]*mlx.Array, 4)
	positions := []*mlx.Array{posT, posH, posW, posL}

	for i, axisDim := range axesDims {
		half := axisDim / 2

		// Create frequency array for this axis: theta^(-2j/dim) for j=0..half-1
		// This matches diffusers: 1.0 / (theta ** (torch.arange(0, dim, 2) / dim))
		freqs := make([]float32, half)
		for j := int32(0); j < half; j++ {
			freqs[j] = float32(math.Exp(float64(-logTheta * float32(2*j) / float32(axisDim))))
		}
		freqArr := mlx.NewArray(freqs, []int32{1, half})

		// Compute pos * freq -> [L, half]
		posExpanded := positions[i] // [L, 1]
		args := mlx.Mul(posExpanded, freqArr) // [L, half]

		// Compute cos and sin for this axis
		cosAxis := mlx.Cos(args) // [L, half]
		sinAxis := mlx.Sin(args) // [L, half]

		// repeat_interleave(2): [c0, c1, ...] -> [c0, c0, c1, c1, ...]
		// Reshape [L, half] -> [L, half, 1], tile to [L, half, 2], reshape to [L, axisDim]
		cosAxis = mlx.ExpandDims(cosAxis, 2)                        // [L, half, 1]
		cosAxis = mlx.Tile(cosAxis, []int32{1, 1, 2})               // [L, half, 2]
		cosAxis = mlx.Reshape(cosAxis, seqLen, axisDim)             // [L, axisDim]

		sinAxis = mlx.ExpandDims(sinAxis, 2)
		sinAxis = mlx.Tile(sinAxis, []int32{1, 1, 2})
		sinAxis = mlx.Reshape(sinAxis, seqLen, axisDim)

		cosArrs[i] = cosAxis
		sinArrs[i] = sinAxis
	}

	// Concatenate all axes: [L, headDim]
	cos := mlx.Concatenate(cosArrs, 1)
	sin := mlx.Concatenate(sinArrs, 1)

	// Reshape to [1, L, 1, headDim] for broadcasting with attention
	cos = mlx.Reshape(cos, 1, seqLen, 1, headDim)
	sin = mlx.Reshape(sin, 1, seqLen, 1, headDim)

	return cos, sin
}

// ApplyRoPE4D applies 4D rotary position embeddings to queries and keys.
// x: [B, L, nheads, head_dim]
// cos, sin: [1, L, 1, head_dim] (with repeat_interleave applied)
// Returns: x with RoPE applied
// Matches diffusers apply_rotary_emb with use_real=True, use_real_unbind_dim=-1
func ApplyRoPE4D(x *mlx.Array, cos, sin *mlx.Array) *mlx.Array {
	shape := x.Shape()
	B := shape[0]
	L := shape[1]
	nheads := shape[2]
	headDim := shape[3]
	half := headDim / 2

	// Reshape x to [B, L, nheads, half, 2] and split into real/imag
	xReshaped := mlx.Reshape(x, B, L, nheads, half, 2)

	// Extract real (index 0) and imag (index 1) parts
	xReal := mlx.Slice(xReshaped, []int32{0, 0, 0, 0, 0}, []int32{B, L, nheads, half, 1})
	xImag := mlx.Slice(xReshaped, []int32{0, 0, 0, 0, 1}, []int32{B, L, nheads, half, 2})
	xReal = mlx.Squeeze(xReal, 4) // [B, L, nheads, half]
	xImag = mlx.Squeeze(xImag, 4) // [B, L, nheads, half]

	// x_rotated = stack([-x_imag, x_real], dim=-1).flatten(-2)
	// This creates [-x_imag[0], x_real[0], -x_imag[1], x_real[1], ...]
	negXImag := mlx.Neg(xImag)
	negXImag = mlx.ExpandDims(negXImag, 4) // [B, L, nheads, half, 1]
	xReal = mlx.ExpandDims(xReal, 4)       // [B, L, nheads, half, 1]
	xRotated := mlx.Concatenate([]*mlx.Array{negXImag, xReal}, 4) // [B, L, nheads, half, 2]
	xRotated = mlx.Reshape(xRotated, B, L, nheads, headDim)       // [B, L, nheads, headDim]

	// out = x * cos + x_rotated * sin
	return mlx.Add(mlx.Mul(x, cos), mlx.Mul(xRotated, sin))
}

// PrepareRoPECache creates RoPE cache for text + noise, optionally with reference images.
// textLen: number of text tokens
// noiseH, noiseW: dimensions of the noise latent in patch tokens
// axesDims: [32, 32, 32, 32]
// theta: 2000
// refHeights, refWidths: optional reference image dimensions (pass nil/empty for no images)
// scale: time coordinate offset between reference images (e.g., 10)
func PrepareRoPECache(textLen, noiseH, noiseW int32, axesDims []int32, theta int32, refHeights, refWidths []int32, scale int32) *RoPECache {
	textIDs := PrepareTextIDs(textLen)
	noiseIDs := PrepareLatentIDs(noiseH, noiseW)

	var allIDs *mlx.Array
	imageLen := noiseH * noiseW

	if len(refHeights) > 0 {
		refIDs := PrepareImageIDs(refHeights, refWidths, scale)
		allIDs = mlx.Concatenate([]*mlx.Array{textIDs, noiseIDs, refIDs}, 0)
		for i := range refHeights {
			imageLen += refHeights[i] * refWidths[i]
		}
	} else {
		allIDs = mlx.Concatenate([]*mlx.Array{textIDs, noiseIDs}, 0)
	}

	cos, sin := ComputeRoPE(allIDs, axesDims, theta)
	cos = mlx.ToBFloat16(cos)
	sin = mlx.ToBFloat16(sin)

	return &RoPECache{Cos: cos, Sin: sin, TextLen: textLen, ImageLen: imageLen}
}
