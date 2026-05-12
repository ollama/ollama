package glmocr

import (
	"log/slog"
	"math"
	"slices"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/ml/nn/rope"
)

type Grid struct {
	Height      int // Number of patches in height direction
	Width       int // Number of patches in width direction
	Temporal    int
	ImageHeight int // Full image height in pixels
	ImageWidth  int // Full image width in pixels
}

type VisionModelOptions struct {
	hiddenSize        int
	numHeads          int
	headDim           int
	numChannels       int
	patchSize         int
	temporalPatchSize int
	imageSize         int
	spatialMergeSize  int
	outHiddenSize     int
	intermediateSize  int
	eps               float32
}

type VisionPatchEmbed struct {
	Proj  *nn.Conv2D `gguf:"patch_embd_0"`
	Proj1 *nn.Conv2D `gguf:"patch_embd_1"`
	Bias  ml.Tensor  `gguf:"patch_embd.bias"`
}

func (pe *VisionPatchEmbed) Forward(ctx ml.Context, pixelValues ml.Tensor, grid *Grid, opts *VisionModelOptions) ml.Tensor {
	_ = grid // patches are already in merge-block order

	// pixelValues shape: [patchDim, numPatches]
	numPatches := pixelValues.Shape()[1]

	// Reshape to [patchSize*patchSize, temporalPatchSize, numChannels, numPatches]
	pixelValues = pixelValues.Reshape(ctx, opts.patchSize*opts.patchSize, opts.temporalPatchSize, opts.numChannels, numPatches)
	// Permute to [temporalPatchSize, patchSize*patchSize, numChannels, numPatches]
	pixelValues = pixelValues.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)

	// Slice temporal frames for Conv2D (simulate Conv3D)
	in0 := pixelValues.View(ctx, 0, 1, pixelValues.Stride(1), pixelValues.Dim(1), pixelValues.Stride(2), pixelValues.Dim(2), pixelValues.Stride(3), pixelValues.Dim(3)).Contiguous(ctx)
	in0 = in0.Reshape(ctx, opts.patchSize, opts.patchSize, opts.numChannels, numPatches)

	s0, s1 := opts.patchSize, opts.patchSize
	p0, p1 := 0, 0
	d0, d1 := 1, 1
	hiddenStates := pe.Proj.Forward(ctx, in0, s0, s1, p0, p1, d0, d1)

	if pe.Proj1 != nil && opts.temporalPatchSize > 1 {
		in1 := pixelValues.View(ctx, pixelValues.Stride(0), 1, pixelValues.Stride(1), pixelValues.Dim(1), pixelValues.Stride(2), pixelValues.Dim(2), pixelValues.Stride(3), pixelValues.Dim(3)).Contiguous(ctx)
		in1 = in1.Reshape(ctx, opts.patchSize, opts.patchSize, opts.numChannels, numPatches)
		out1 := pe.Proj1.Forward(ctx, in1, s0, s1, p0, p1, d0, d1)
		hiddenStates = hiddenStates.Add(ctx, out1)
	}

	// Flatten to [hidden_size, num_patches]
	hiddenStates = hiddenStates.Reshape(ctx, opts.hiddenSize, numPatches)

	// Add patch bias - reshape from [hidden_size] to [hidden_size, 1] for broadcasting
	if pe.Bias != nil {
		hiddenStates = hiddenStates.Add(ctx, pe.Bias.Reshape(ctx, opts.hiddenSize, 1))
	}

	return hiddenStates
}

type VisionSelfAttention struct {
	QKV    *nn.Linear  `gguf:"attn_qkv"`
	QNorm  *nn.RMSNorm `gguf:"attn_q_norm"`
	KNorm  *nn.RMSNorm `gguf:"attn_k_norm"`
	Output *nn.Linear  `gguf:"attn_out"`
}

func (sa *VisionSelfAttention) Forward(ctx ml.Context, hiddenStates, positions ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	batchSize := hiddenStates.Dim(1)

	// Combined QKV projection: [3*hidden_size, batch_size]
	qkv := sa.QKV.Forward(ctx, hiddenStates)

	// Split using ChunkSections along dim 0 (handles byte offsets correctly)
	// ChunkSections returns views - must make contiguous before further operations
	chunks := qkv.ChunkSections(ctx, 0, opts.hiddenSize, opts.hiddenSize, opts.hiddenSize)
	q := chunks[0].Contiguous(ctx)
	k := chunks[1].Contiguous(ctx)
	v := chunks[2].Contiguous(ctx)

	// Reshape for multi-head attention: [hiddenSize, N] -> [headDim, numHeads, N]
	q = q.Reshape(ctx, opts.headDim, opts.numHeads, batchSize)
	k = k.Reshape(ctx, opts.headDim, opts.numHeads, batchSize)
	v = v.Reshape(ctx, opts.headDim, opts.numHeads, batchSize)

	// Apply Q-norm and K-norm after head reshape
	// Weights are [headDim]=64, tensor is [headDim, numHeads, N]
	q = sa.QNorm.Forward(ctx, q, opts.eps)
	k = sa.KNorm.Forward(ctx, k, opts.eps)

	// Apply rotary position embeddings with vision-style 2D positions.
	// ggml's vision RoPE uses two position dimensions (H/W) with half-rotation pairs.
	// We provide H/W sections and leave the remaining sections empty.
	ropeFreqBase := float32(10000.0)
	section := opts.headDim / 4
	if section <= 0 {
		section = 1
	}
	sections := []int{section, section, 0, 0}
	q = nn.RoPE(ctx, q, positions, opts.headDim/2, ropeFreqBase, 1.0, rope.WithVision(sections))
	k = nn.RoPE(ctx, k, positions, opts.headDim/2, ropeFreqBase, 1.0, rope.WithVision(sections))

	// Scale factor for scaled dot-product attention
	scale := 1.0 / math.Sqrt(float64(opts.headDim))

	// Try flash attention first (ScaledDotProductAttention), fall back to manual
	if sdpa, ok := q.(ml.ScaledDotProductAttention); ok {
		attention := sdpa.ScaledDotProductAttention(ctx, k, v, nil, nil, nil, scale, false)
		attention = attention.Reshape(ctx, opts.hiddenSize, batchSize)
		return sa.Output.Forward(ctx, attention)
	}

	slog.Warn("glmocr: vision attention falling back to manual attention",
		"batchSize", batchSize, "numHeads", opts.numHeads,
		"hint", "set OLLAMA_FLASH_ATTENTION=1 to enable flash attention")

	// Manual attention fallback
	// q, k, v are [headDim, numHeads, batchSize] - GGML treats as 4D with implicit dim 3 = 1
	q = q.Permute(ctx, 0, 2, 1, 3)
	k = k.Permute(ctx, 0, 2, 1, 3)
	v = v.Permute(ctx, 1, 2, 0, 3).Contiguous(ctx)

	// Attention scores
	kq := k.MulmatFullPrec(ctx, q)
	kq = kq.Scale(ctx, scale)
	kq = kq.Softmax(ctx)

	// Attention output: v @ kq (note: v first)
	kqv := v.Mulmat(ctx, kq)
	attention := kqv.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	attention = attention.Reshape(ctx, opts.hiddenSize, batchSize)

	return sa.Output.Forward(ctx, attention)
}

type VisionMLP struct {
	Gate *nn.Linear `gguf:"ffn_gate"`
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
}

func (mlp *VisionMLP) Forward(ctx ml.Context, hiddenStates ml.Tensor) ml.Tensor {
	// SwiGLU: down(silu(gate(x)) * up(x))
	gate := mlp.Gate.Forward(ctx, hiddenStates).SILU(ctx, mlp.Up.Forward(ctx, hiddenStates))
	return mlp.Down.Forward(ctx, gate)
}

type VisionBlock struct {
	Norm1         *nn.RMSNorm `gguf:"ln1"`
	SelfAttention *VisionSelfAttention
	Norm2         *nn.RMSNorm `gguf:"ln2"`
	MLP           *VisionMLP
}

func (b *VisionBlock) Forward(ctx ml.Context, hiddenStates, positions ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	// Pre-norm architecture
	residual := hiddenStates
	hiddenStates = b.Norm1.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = b.SelfAttention.Forward(ctx, hiddenStates, positions, opts)
	hiddenStates = hiddenStates.Add(ctx, residual)

	residual = hiddenStates
	hiddenStates = b.Norm2.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = b.MLP.Forward(ctx, hiddenStates)
	hiddenStates = hiddenStates.Add(ctx, residual)

	return hiddenStates
}

type VisionDownsample struct {
	*nn.Conv2D
}

func (d *VisionDownsample) Forward(ctx ml.Context, hiddenStates ml.Tensor, grid *Grid, opts *VisionModelOptions) ml.Tensor {
	// Apply spatial downsampling via Conv2D
	// Input: [hidden_size, num_patches] where patches are in merge-block order

	if d.Conv2D == nil || d.Weight == nil {
		slog.Error("VisionDownsample weights not loaded - model may be corrupted or incompatible")
		return hiddenStates // Return input unchanged as fallback
	}

	merge := opts.spatialMergeSize
	numOutputTokens := (grid.Height / merge) * (grid.Width / merge)

	// Step 1: Reshape to [hidden_size, merge, merge, num_output_tokens]
	hiddenStates = hiddenStates.Reshape(ctx, opts.hiddenSize, merge, merge, numOutputTokens)

	// Step 2: Permute to [merge, merge, hidden_size, num_output_tokens]
	// ggml semantics: result.ne[perm[i]] = input.ne[i]
	// So permute(2,0,1,3) on [1024,2,2,N] gives: ne[2]=1024, ne[0]=2, ne[1]=2, ne[3]=N -> [2,2,1024,N]
	hiddenStates = hiddenStates.Permute(ctx, 2, 0, 1, 3).Contiguous(ctx)

	// Step 3: Apply Conv2D without bias (bias added after reshape)
	// Note: ggml_conv_2d takes (kernel, input) - kernel must be receiver in ollama
	s0, s1 := merge, merge
	p0, p1 := 0, 0
	d0, d1 := 1, 1
	hiddenStates = d.Weight.Conv2D(ctx, hiddenStates, s0, s1, p0, p1, d0, d1)

	// Step 4: Reshape to [out_hidden_size, num_output_tokens]
	hiddenStates = hiddenStates.Reshape(ctx, opts.outHiddenSize, numOutputTokens)

	// Step 5: Add bias after reshape
	// Reshape bias from [out_hidden_size] to [out_hidden_size, 1] for proper broadcasting
	if d.Bias != nil {
		hiddenStates = hiddenStates.Add(ctx, d.Bias.Reshape(ctx, opts.outHiddenSize, 1))
	}

	return hiddenStates
}

type PatchMerger struct {
	// GGUF tags align with mm.* keys used by the model
	Proj     *nn.Linear    `gguf:"model.fc"`  // mm.model.fc.weight
	PostLN   *nn.LayerNorm `gguf:"post_norm"` // mm.post_norm.weight/bias
	GateProj *nn.Linear    `gguf:"gate"`      // mm.gate.weight
	UpProj   *nn.Linear    `gguf:"up"`        // mm.up.weight
	DownProj *nn.Linear    `gguf:"down"`      // mm.down.weight
}

func (m *PatchMerger) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	// Linear projection
	hiddenStates = m.Proj.Forward(ctx, hiddenStates)

	// Post-projection layer norm + GELU ERF
	hiddenStates = m.PostLN.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = hiddenStates.GELU_ERF(ctx)
	// Force a copy to avoid in-place mutation issues with GELU_ERF
	hiddenStates = hiddenStates.Contiguous(ctx)

	// SwiGLU MLP: down(silu(gate(x)) * up(x))
	gateOut := m.GateProj.Forward(ctx, hiddenStates)
	upOut := m.UpProj.Forward(ctx, hiddenStates)
	gate := gateOut.SILU(ctx, upOut)
	return m.DownProj.Forward(ctx, gate)
}

type VisionModel struct {
	PatchEmbed *VisionPatchEmbed
	Blocks     []VisionBlock `gguf:"blk"`
	PostLN     *nn.RMSNorm   `gguf:"post_ln"`
	// Note: Downsample is applied at the model level so mm.patch_merger stays separate

	*VisionModelOptions
}

func (m *VisionModel) Forward(ctx ml.Context, pixelValues ml.Tensor, grid *Grid) ml.Tensor {
	// Extract patch embeddings from flattened patches
	hiddenStates := m.PatchEmbed.Forward(ctx, pixelValues, grid, m.VisionModelOptions)

	// Create position IDs for RoPE (spatial grid)
	// Patches are already in merge-block order from preprocessing
	positions := m.createPositions(ctx, grid)

	// Process through vision blocks
	for _, block := range m.Blocks {
		hiddenStates = block.Forward(ctx, hiddenStates, positions, m.VisionModelOptions)
	}

	// Post-layernorm
	hiddenStates = m.PostLN.Forward(ctx, hiddenStates, m.eps)

	// Note: Downsample is now applied separately in Model.EncodeMultimodal
	// so mm.patch_merger remains a distinct module

	return hiddenStates
}

func (m *VisionModel) createPositions(ctx ml.Context, grid *Grid) ml.Tensor {
	// Create spatial position IDs for vision RoPE
	// Position layout: [height, width, height, width] - 4 sections for mrope
	// Patches are in MERGE-BLOCK order after VisionPatchEmbed interleaving
	// This follows the GLM-OCR rot_pos_emb layout
	numPatches := grid.Height * grid.Width
	mergeRatio := m.spatialMergeSize

	// Build position arrays in merge-block order
	// Each merge_ratio x merge_ratio block of patches is grouped together
	hpos := make([]int32, numPatches)
	wpos := make([]int32, numPatches)
	ptr := 0
	for y := 0; y < grid.Height; y += mergeRatio {
		for x := 0; x < grid.Width; x += mergeRatio {
			for dy := range mergeRatio {
				for dx := range mergeRatio {
					hpos[ptr] = int32(y + dy)
					wpos[ptr] = int32(x + dx)
					ptr++
				}
			}
		}
	}

	// Build position arrays for 4 sections (mrope). ggml vision RoPE uses only H/W;
	// keep remaining sections zeroed to match its conventions.
	zeros := make([]int32, numPatches)
	s := [][]int32{
		hpos,  // Section 0: height
		wpos,  // Section 1: width
		zeros, // Section 2: unused
		zeros, // Section 3: unused
	}

	return ctx.Input().FromInts(slices.Concat(s...), numPatches*4)
}

func newVisionModel(c fs.Config) *VisionModel {
	hiddenSize := int(c.Uint("vision.embedding_length", 1024))
	numHeads := int(c.Uint("vision.attention.head_count", 16))
	numChannels := int(c.Uint("vision.num_channels", 3))
	patchSize := int(c.Uint("vision.patch_size", 14))
	temporalPatchSize := int(c.Uint("vision.temporal_patch_size", 2))
	imageSize := int(c.Uint("vision.image_size", 336))
	spatialMergeSize := int(c.Uint("vision.spatial_merge_size", 2))
	outHiddenSize := int(c.Uint("vision.out_hidden_size", 1536))
	intermediateSize := int(c.Uint("vision.intermediate_size", 4096))
	eps := c.Float("vision.attention.layer_norm_rms_epsilon", 1e-5)

	return &VisionModel{
		Blocks: make([]VisionBlock, c.Uint("vision.block_count", 24)),
		VisionModelOptions: &VisionModelOptions{
			hiddenSize:        hiddenSize,
			numHeads:          numHeads,
			headDim:           hiddenSize / numHeads,
			numChannels:       numChannels,
			patchSize:         patchSize,
			temporalPatchSize: temporalPatchSize,
			imageSize:         imageSize,
			spatialMergeSize:  spatialMergeSize,
			outHiddenSize:     outHiddenSize,
			intermediateSize:  intermediateSize,
			eps:               eps,
		},
	}
}
