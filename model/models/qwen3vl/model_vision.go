package qwen3vl

import (
	"iter"
	"log/slog"
	"math"
	"slices"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
)

type VisionAttention struct {
	Query  *nn.Linear `gguf:"attn_q"`   // Unified model: separate Q
	Key    *nn.Linear `gguf:"attn_k"`   // Unified model: separate K
	Value  *nn.Linear `gguf:"attn_v"`   // Unified model: separate V
	QKV    *nn.Linear `gguf:"attn_qkv"` // Split model: combined QKV
	Output *nn.Linear `gguf:"attn_out"`
}

func rotateHalf(ctx ml.Context, t ml.Tensor) ml.Tensor {
	x1 := t.Slice(ctx, 0, 0, t.Dim(0)/2, 1)
	x2 := t.Slice(ctx, 0, t.Dim(0)/2, t.Dim(0), 1).Contiguous(ctx)
	return x2.Scale(ctx, -1).Concat(ctx, x1, 0)
}

func applyRotaryPositionalEmbedding(ctx ml.Context, t, cos, sin ml.Tensor) ml.Tensor {
	return t.Mul(ctx, cos).Add(ctx, rotateHalf(ctx, t).Mul(ctx, sin))
}

func (sa *VisionAttention) Forward(ctx ml.Context, hiddenStates, cos, sin ml.Tensor, opts VisionOptions) ml.Tensor {
	var query, key, value ml.Tensor

	if sa.QKV != nil {
		// Split model: combined QKV tensor - split into Q, K, V
		qkv := sa.QKV.Forward(ctx, hiddenStates)
		// qkv shape is [hiddenSize*3, seqLen] - split along first dimension
		hiddenSize := opts.hiddenSize
		seqLen := qkv.Dim(1)
		query = qkv.Slice(ctx, 0, 0, hiddenSize, 1)
		key = qkv.Slice(ctx, 0, hiddenSize, hiddenSize*2, 1)
		value = qkv.Slice(ctx, 0, hiddenSize*2, hiddenSize*3, 1)
		// Use Contiguous(ctx, shape...) to avoid view_src chain; this calls ggml_cont_Nd, creating a truly independent tensor without view_src issues.
		query = query.Contiguous(ctx, opts.headDim(), opts.numHeads, seqLen)
		key = key.Contiguous(ctx, opts.headDim(), opts.numHeads, seqLen)
		value = value.Contiguous(ctx, opts.headDim(), opts.numHeads, seqLen)
	} else {
		// Unified model: separate Q, K, V tensors - use normal Reshape
		query = sa.Query.Forward(ctx, hiddenStates)
		key = sa.Key.Forward(ctx, hiddenStates)
		value = sa.Value.Forward(ctx, hiddenStates)
		query = query.Reshape(ctx, opts.headDim(), opts.numHeads, query.Dim(1))
		key = key.Reshape(ctx, opts.headDim(), opts.numHeads, key.Dim(1))
		value = value.Reshape(ctx, opts.headDim(), opts.numHeads, value.Dim(1))
	}

	query = applyRotaryPositionalEmbedding(ctx, query, cos, sin)
	key = applyRotaryPositionalEmbedding(ctx, key, cos, sin)

	attention := nn.Attention(ctx, query, key, value, math.Pow(float64(opts.headDim()), -0.5), nil)
	attention = attention.Reshape(ctx, opts.hiddenSize, attention.Dim(2))
	return sa.Output.Forward(ctx, attention)
}

type VisionMLP struct {
	FC1 *nn.Linear `gguf:"linear_fc1,alt:ffn_up"`
	FC2 *nn.Linear `gguf:"linear_fc2,alt:ffn_down"`
}

func (mlp *VisionMLP) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts VisionOptions) ml.Tensor {
	fc1Out := mlp.FC1.Forward(ctx, hiddenStates)
	activated := fc1Out.GELU(ctx)
	return mlp.FC2.Forward(ctx, activated)
}

type VisionEncoderLayer struct {
	Norm1     *nn.LayerNorm `gguf:"norm1,alt:ln1"`
	Attention *VisionAttention
	Norm2     *nn.LayerNorm `gguf:"norm2,alt:ln2"`
	MLP       *VisionMLP    `gguf:"mlp"`
}

func (e *VisionEncoderLayer) Forward(ctx ml.Context, hiddenStates, cos, sin ml.Tensor, opts VisionOptions) ml.Tensor {
	residual := hiddenStates
	hiddenStates = e.Norm1.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = e.Attention.Forward(ctx, hiddenStates, cos, sin, opts)
	hiddenStates = hiddenStates.Add(ctx, residual)

	residual = hiddenStates
	hiddenStates = e.Norm2.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = e.MLP.Forward(ctx, hiddenStates, opts)
	return hiddenStates.Add(ctx, residual)
}

type VisionOptions struct {
	hiddenSize,
	numHeads,
	patchSize,
	numChannels,
	spatialMergeSize,
	temporalPatchSize,
	storagePatchSize,
	gridPerSide int

	eps,
	ropeTheta float32

	// isSplitArchitecture indicates split model variant uses different patch embedding
	// Split: [16,16,3,1152] per-channel 2D conv with multiple weight tensors
	// Unified: [16,16,2,3456] 3D conv with merged channels
	isSplitArchitecture bool

	deepstackVisualIndexes []int32
	mropeSections          []int
}

func (o VisionOptions) headDim() int {
	return o.hiddenSize / o.numHeads
}

type VisionPatchMerger struct {
	Norm *nn.LayerNorm `gguf:"norm,alt:ln_merger"`
	FC1  *nn.Linear    `gguf:"fc1,alt:linear_fc1,alt:ffn_up"`
	FC2  *nn.Linear    `gguf:"fc2,alt:linear_fc2,alt:ffn_down"`
}

func (m *VisionPatchMerger) Forward(ctx ml.Context, visionOutputs ml.Tensor, postshuffleNorm bool, opts VisionOptions) ml.Tensor {
	hiddenSize := opts.hiddenSize * opts.spatialMergeSize * opts.spatialMergeSize
	if postshuffleNorm {
		visionOutputs = visionOutputs.Reshape(ctx, hiddenSize, -1)
	}

	// Norm is required
	if m.Norm != nil {
		visionOutputs = m.Norm.Forward(ctx, visionOutputs, opts.eps)
	}
	visionOutputs = visionOutputs.Reshape(ctx, hiddenSize, -1)

	// FC1/FC2 should exist for Qwen3-VL (v.deepstack.X.fc1/fc2 weights)
	if m.FC1 != nil && m.FC2 != nil {
		slog.Debug("DeepstackMerger projecting", "input_dim", hiddenSize, "input_shape", visionOutputs.Shape())
		fc1Out := m.FC1.Forward(ctx, visionOutputs)
		slog.Debug("FC1 output", "shape", fc1Out.Shape())
		activated := fc1Out.GELU(ctx)
		slog.Debug("After GELU", "shape", activated.Shape())
		result := m.FC2.Forward(ctx, activated)
		slog.Debug("FC2 output", "shape", result.Shape())
		return result
	}

	// This shouldn't happen for Qwen3-VL split models
	slog.Warn("DeepstackMerger FC layers are nil", "dim", hiddenSize)
	return visionOutputs
}

type VisionPositionEmbedding struct {
	PositionEmbedding *nn.Embedding `gguf:"pos_embed,alt:position_embd"` // Unified: pos_embed, Split: position_embd
}

func makeSlice2D[T int32 | float32](n0, n1 int) iter.Seq[[]T] {
	return func(yield func([]T) bool) {
		for range n0 {
			if !yield(make([]T, n1)) {
				return
			}
		}
	}
}

func (m *VisionPositionEmbedding) Forward(ctx ml.Context, hiddenStates ml.Tensor, grid *Grid, opts VisionOptions) ml.Tensor {
	// Check if position embedding exists
	if m == nil || m.PositionEmbedding == nil || m.PositionEmbedding.Weight == nil {
		// No position embedding tensor - return unchanged
		return hiddenStates
	}

	n := hiddenStates.Dim(0) // hidden size

	// DEBUG: Log position embedding parameters for split vs nosplit comparison
	slog.Debug("PositionEmbedding.Forward",
		"hiddenStates_shape", hiddenStates.Shape(),
		"grid_width", grid.Width, "grid_height", grid.Height,
		"gridPerSide", opts.gridPerSide, "spatialMergeSize", opts.spatialMergeSize,
		"isSplitArchitecture", opts.isSplitArchitecture)

	// UNIFIED MODEL: Use bilinear interpolation (original upstream code)
	// This is required for proper position encoding in unified models
	indexSlice := slices.Collect(makeSlice2D[int32](4, grid.Height*grid.Width))
	weightSlice := slices.Collect(makeSlice2D[float32](4, grid.Height*grid.Width))

	stepHeight := float32(opts.gridPerSide-1) / float32(grid.Height-1)
	stepWidth := float32(opts.gridPerSide-1) / float32(grid.Width-1)

	var i int
	for h := range grid.Height {
		for w := range grid.Width {
			y, x := float32(h)*stepHeight, float32(w)*stepWidth

			floorY, floorX := int32(y), int32(x)
			ceilY, ceilX := min(floorY+1, int32(opts.gridPerSide-1)), min(floorX+1, int32(opts.gridPerSide-1))

			indexSlice[0][i] = floorY*int32(opts.gridPerSide) + floorX
			indexSlice[1][i] = floorY*int32(opts.gridPerSide) + ceilX
			indexSlice[2][i] = ceilY*int32(opts.gridPerSide) + floorX
			indexSlice[3][i] = ceilY*int32(opts.gridPerSide) + ceilX

			weightSlice[0][i] = (1 - (y - float32(floorY))) * (1 - (x - float32(floorX)))
			weightSlice[1][i] = (1 - (y - float32(floorY))) * (x - float32(floorX))
			weightSlice[2][i] = (y - float32(floorY)) * (1 - (x - float32(floorX)))
			weightSlice[3][i] = (y - float32(floorY)) * (x - float32(floorX))

			i++
		}
	}

	indices := ctx.Input().FromInts(slices.Concat(indexSlice...), grid.Height*grid.Width*4)
	weights := ctx.Input().FromFloats(slices.Concat(weightSlice...), 1, grid.Height*grid.Width*4)

	positionEmbeds := m.PositionEmbedding.Forward(ctx, indices)
	positionEmbeds = positionEmbeds.Mul(ctx, weights)
	positionEmbeds = positionEmbeds.Reshape(ctx, n, -1, 4)

	positionEmbedsChunks := positionEmbeds.Chunk(ctx, 2, 1)
	positionEmbeds = positionEmbedsChunks[0].
		Add(ctx, positionEmbedsChunks[1]).
		Add(ctx, positionEmbedsChunks[2]).
		Add(ctx, positionEmbedsChunks[3])

	slog.Debug("PositionEmbedding after chunk sum",
		"shape", positionEmbeds.Shape(),
		"expected_spatial_merge_output", []int{int(n), grid.Width / opts.spatialMergeSize * grid.Height / opts.spatialMergeSize})

	positionEmbeds = positionEmbeds.Reshape(ctx, -1, grid.Width/opts.spatialMergeSize, opts.spatialMergeSize, grid.Height/opts.spatialMergeSize)
	positionEmbeds = positionEmbeds.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx, n, -1)

	slog.Debug("PositionEmbedding after spatial merge",
		"posEmbed_shape", positionEmbeds.Shape(),
		"hiddenStates_shape", hiddenStates.Shape(),
		"SHAPES_MATCH", positionEmbeds.Dim(1) == hiddenStates.Dim(1))

	return hiddenStates.Add(ctx, positionEmbeds)
}

type VisionModel struct {
	PatchEmbedding    *nn.Conv3D `gguf:"patch_embed,alt:patch_embd"` // Unified model uses 3D conv
	PatchEmbedding1   *nn.Linear // Second kernel for split - manually loaded (tensor: v.patch_embd.weight.1)
	PatchEmbedding2D  *nn.Conv2D // Split model uses 2D conv (not auto-populated, set from PatchEmbedding when detected)
	PositionEmbedding *VisionPositionEmbedding
	Layers            []VisionEncoderLayer `gguf:"blk"`
	PatchMerger       *VisionPatchMerger   `gguf:"merger"`
	PostNorm          *nn.LayerNorm        `gguf:"post_ln"` // Present in split models (1152 dim)
	DeepstackMerger   []*VisionPatchMerger `gguf:"deepstack_merger,alt:deepstack"`

	// Multimodal projector FC layers (set from Model for split models)
	// These are separate from DeepstackMerger - used for main vision projection
	MultimodalFC1 *nn.Linear // FC1: 4608 -> 4608 with GELU (from mm.0.*)
	MultimodalFC2 *nn.Linear // FC2: 4608 -> 4096 (from mm.2.*)

	// Deepstack layer IDs detected from GGUF (e.g., [5,11,17] for 4B or [8,16,24] for 8B)
	deepstackLayerIDs []int

	VisionOptions
}

func (m *VisionModel) positions(ctx ml.Context, grid *Grid) (_, _ ml.Tensor) {
	// Create indices on CPU first
	cpuIndices := ctx.Input().FromInts(slices.Collect(func(yield func(int32) bool) {
		for y := range grid.Height {
			for x := range grid.Width {
				if !yield(int32(y)) {
					return
				}
				if !yield(int32(x)) {
					return
				}
			}
		}
	}), grid.Width*grid.Height*2)

	// Copy indices to GPU layer context for split model compatibility
	gpuIndicesDest := ctx.Layer(0).Zeros(ml.DTypeI32, grid.Width*grid.Height*2)
	indices := cpuIndices.Copy(ctx, gpuIndicesDest)
	// Use Contiguous(shape) to avoid view_src issues - this reshapes and creates independent tensor
	indices = indices.Contiguous(ctx, -1, grid.Width/m.spatialMergeSize, m.spatialMergeSize, grid.Height/m.spatialMergeSize)
	indices = indices.Permute(ctx, 0, 2, 1, 3)
	// Final reshape to 1D also uses Contiguous(shape) to avoid view_src
	indices = indices.Contiguous(ctx, -1)

	halfDim := m.headDim() / 2
	maxGrid := max(grid.Height, grid.Width)

	// Create frequency table
	// freqDim = halfDim/2 because indices contain BOTH y and x coordinates (2x factor)
	// The reshape later will combine them back to produce output matching seqLen
	freqDim := halfDim / 2
	freqCount := maxGrid * freqDim

	ropeTheta := float64(m.ropeTheta)
	freqData := make([]float32, freqCount)
	for i := range maxGrid {
		for j := range freqDim {
			freqData[i*freqDim+j] = float32(float64(i) / math.Pow(ropeTheta, float64(j*2)/float64(halfDim)))
		}
	}

	// Create frequencies tensor on input context (CPU) first
	cpuFrequencies := ctx.Input().FromFloats(freqData, freqDim, maxGrid)

	// For split models, explicitly copy to GPU layer context to avoid scheduler issues
	// Create a GPU destination tensor and copy the CPU data to it
	gpuDest := ctx.Layer(0).Zeros(ml.DTypeF32, freqDim, maxGrid)
	frequencies := cpuFrequencies.Copy(ctx, gpuDest)

	embeds := frequencies.Rows(ctx, indices)
	// Use Contiguous(ctx, shape...) to avoid view_src chain - this calls ggml_cont_Nd
	// which creates a truly independent tensor without view_src issues
	embeds = embeds.Contiguous(ctx, halfDim, 1, -1)
	embeds = embeds.Concat(ctx, embeds, 0)
	return embeds.Cos(ctx), embeds.Sin(ctx)
}

// Forward computes the vision model for an input tensor
func (m *VisionModel) Forward(ctx ml.Context, pixelValues ml.Tensor, grid *Grid) (ml.Tensor, []ml.Tensor) {
	var hiddenStates ml.Tensor

	if m.isSplitArchitecture {
		// Split architecture: patch embeddings are stored as flattened weights.
		// Use the patch-based input tensor from ImageProcessor (shape [patchDim, numPatches]) and apply
		// both kernels to the SAME input, then sum (matches reference behavior).
		if m.PatchEmbedding == nil || m.PatchEmbedding.Weight == nil {
			panic("VisionModel.PatchEmbedding.Weight is nil - split model patch embedding not loaded correctly")
		}
		if len(pixelValues.Shape()) != 2 {
			panic("split vision expects patch tensor [patchDim, numPatches]")
		}

		hasKernel2 := m.PatchEmbedding1 != nil && m.PatchEmbedding1.Weight != nil
		slog.Debug("Split patch embedding forward (Mulmat)",
			"pixel_shape", pixelValues.Shape(),
			"weight_shape", m.PatchEmbedding.Weight.Shape(),
			"has_kernel2", hasKernel2,
			"patchSize", m.patchSize, "hiddenSize", m.hiddenSize,
		)

		// Kernel 0
		h1 := m.PatchEmbedding.Weight.Mulmat(ctx, pixelValues)
		hiddenStates = h1
		// Kernel 1 (optional)
		if hasKernel2 {
			h2 := m.PatchEmbedding1.Weight.Mulmat(ctx, pixelValues)
			hiddenStates = hiddenStates.Add(ctx, h2)
		}

		// Apply bias (v.patch_embd.bias) once after summing kernels.
		// The split GGUF provides this tensor and omitting it shifts the feature distribution.
		if m.PatchEmbedding.Bias != nil {
			bias := m.PatchEmbedding.Bias.Reshape(ctx, -1, 1)
			hiddenStates = hiddenStates.Add(ctx, bias)
		}

		slog.Debug("Split model: after linear patch embedding", "shape", hiddenStates.Shape())
	} else {
		// Unified architecture: conv kernel is [kH, kW, temporal, channels*hidden] - use Conv3D
		slog.Debug("Unified patch embedding BEFORE reshape",
			"pixelValues_shape", pixelValues.Shape(),
			"patchSize", m.patchSize,
			"temporalPatchSize", m.temporalPatchSize,
			"numChannels", m.numChannels,
			"target_reshape", []int{m.patchSize, m.patchSize, m.temporalPatchSize, -1})

		pixelValues = pixelValues.Reshape(ctx, m.patchSize, m.patchSize, m.temporalPatchSize, -1)
		slog.Debug("Unified patch embedding AFTER reshape",
			"pixelValues_shape", pixelValues.Shape())

		// Log Conv3D weight shape
		if m.PatchEmbedding != nil && m.PatchEmbedding.Weight != nil {
			slog.Debug("Unified Conv3D kernel",
				"weight_shape", m.PatchEmbedding.Weight.Shape())
		}

		hiddenStates = m.PatchEmbedding.Forward(ctx, pixelValues, m.numChannels, m.patchSize, m.patchSize, m.temporalPatchSize, 0, 0, 0, 1, 1, 1)
		slog.Debug("Unified patch embedding AFTER Conv3D",
			"hiddenStates_shape", hiddenStates.Shape())

	}

	if m.isSplitArchitecture {
		slog.Debug("Split model: after position embedding", "shape", hiddenStates.Shape())
	}

	hiddenStates = m.PositionEmbedding.Forward(ctx, hiddenStates, grid, m.VisionOptions)

	cos, sin := m.positions(ctx, grid)

	// Verify first layer exists before processing
	if len(m.Layers) == 0 {
		panic("VisionModel.Layers is empty - no vision encoder layers loaded")
	}
	// VisionEncoderLayer is a struct (not pointer), check MLP field directly
	if m.Layers[0].MLP == nil {
		slog.Error("First layer MLP is nil - aliases may not be working",
			"layer0_mlp", m.Layers[0].MLP,
			"layer0_norm1", m.Layers[0].Norm1,
			"layer0_norm2", m.Layers[0].Norm2)
		panic("VisionEncoderLayer.MLP is nil - MLP tensor aliases (ffn_up→linear_fc1) not working")
	}

	// Log deepstack configuration before processing
	slog.Debug("VisionModel.Forward starting",
		"n_layers", len(m.Layers),
		"deepstackVisualIndexes", m.deepstackVisualIndexes,
		"n_deepstack_mergers", len(m.DeepstackMerger))

	deepstackStates := make([]ml.Tensor, len(m.deepstackVisualIndexes))
	for layerIdx, layer := range m.Layers {
		hiddenStates = layer.Forward(ctx, hiddenStates, cos, sin, m.VisionOptions)
		if m.isSplitArchitecture && layerIdx == 0 {
			slog.Debug("Split model: after first vision layer", "layer", layerIdx, "shape", hiddenStates.Shape())
		}
		if dsIdx := slices.Index(m.deepstackVisualIndexes, int32(layerIdx)); dsIdx >= 0 && m.DeepstackMerger[dsIdx] != nil {
			deepstackStates[dsIdx] = m.DeepstackMerger[dsIdx].Forward(ctx, hiddenStates, true, m.VisionOptions)
			slog.Debug("Extracted deepstack from layer", "layerIdx", layerIdx, "dsIdx", dsIdx, "shape", deepstackStates[dsIdx].Shape())
		}
	}

	// PatchMerger may be nil for split models
	if m.PatchMerger != nil && m.PatchMerger.FC1 != nil && m.PatchMerger.FC2 != nil {
		hiddenStates = m.PatchMerger.Forward(ctx, hiddenStates, false, m.VisionOptions)
		slog.Debug("Projected main vision via PatchMerger", "shape", hiddenStates.Shape())
	} else if m.MultimodalFC1 != nil && m.MultimodalFC2 != nil {
		// SPLIT MODEL: Use dedicated mm.0/mm.2 projectors for main vision
		// These are SEPARATE from deepstack FC weights - critical difference!
		// In llama.cpp: model.mm_0_w/mm_1_w vs layer.deepstack_fc1_w/deepstack_fc2_w

		// CRITICAL: Apply PostNorm BEFORE projection (matches llama.cpp post_ln_w)
		if m.PostNorm != nil {
			hiddenStates = m.PostNorm.Forward(ctx, hiddenStates, m.VisionOptions.eps)
			slog.Debug("Applied PostNorm before mm.0/mm.2 projection", "shape", hiddenStates.Shape())
		}

		// Main vision goes through spatial merge (2x2) first, grouping neighboring patches
		// into the channel dimension to preserve local structure before projection.
		w, h := grid.Width, grid.Height
		mergedPatches := (w / m.spatialMergeSize) * (h / m.spatialMergeSize)

		// Collapse each 2x2 spatial block into the channel dimension; patch order is already grouped.
		hiddenStates = hiddenStates.Reshape(ctx, m.hiddenSize*m.spatialMergeSize*m.spatialMergeSize, mergedPatches)

		// FC1: 4608 -> 4608 with GELU activation
		hiddenStates = m.MultimodalFC1.Forward(ctx, hiddenStates)
		hiddenStates = hiddenStates.GELU(ctx)

		// FC2: 4608 -> 4096
		hiddenStates = m.MultimodalFC2.Forward(ctx, hiddenStates)

		slog.Debug("Split Model: Projected main vision via mm.0/mm.2",
			"shape", hiddenStates.Shape(), "mergedPatches", mergedPatches)
	} else if len(deepstackStates) > 0 && len(m.DeepstackMerger) > 0 {
		// FALLBACK PATH: When neither PatchMerger nor mm.0/mm.2 projectors are loaded.
		// This path exists for robustness when working with incomplete or non-standard GGUFs.
		// Expected usage:
		//   - Unified models: PatchMerger is populated from GGUF
		//   - Split models: mm.0/mm.2 projectors are populated from GGUF
		// If neither is available, we attempt to use the last DeepstackMerger as a last resort.
		// Note: This produces suboptimal results but prevents crashes. A warning is logged.
		lastIdx := len(m.DeepstackMerger) - 1
		if m.DeepstackMerger[lastIdx] != nil && m.DeepstackMerger[lastIdx].FC1 != nil && m.DeepstackMerger[lastIdx].FC2 != nil {
			// Apply PostNorm first (same as for deepstack layer 24)
			if m.PostNorm != nil {
				hiddenStates = m.PostNorm.Forward(ctx, hiddenStates, m.VisionOptions.eps)
			}
			// Project using the same FC layers as last deepstack (4608 -> 4096)
			hiddenStates = m.DeepstackMerger[lastIdx].Forward(ctx, hiddenStates, true, m.VisionOptions)
			slog.Warn("Split Model: Using DeepstackMerger[last] FALLBACK for main vision (mm.0/mm.2 not loaded)",
				"lastIdx", lastIdx, "shape", hiddenStates.Shape())
		} else {
			// Fallback: Use average of deepstacks if DeepstackMerger FC is not available
			slog.Warn("Split Model: DeepstackMerger[last] FC not available, using average fallback")
			var sumTensor ml.Tensor
			count := 0
			for _, ds := range deepstackStates {
				if ds != nil {
					if sumTensor == nil {
						sumTensor = ds
					} else {
						sumTensor = sumTensor.Add(ctx, ds)
					}
					count++
				}
			}
			if sumTensor != nil && count > 0 {
				scale := 1.0 / float64(count)
				hiddenStates = sumTensor.Scale(ctx, scale)
				slog.Debug("Split Model: Using AVERAGE of Deepstacks as Main Vision Output", "count", count, "shape", hiddenStates.Shape())
			}
		}
	}
	return hiddenStates, deepstackStates
}

// newVisionModel creates a new instance of the Qwen vision model
func newVisionModel(c fs.Config) *VisionModel {
	// For Qwen3-VL, deepstack features are applied to the first N LLM layers
	// where N = number of loaded DeepstackMerger tensors (inferred from GGUF)
	// This matches llama.cpp's approach: if (ubatch.embd && il < n_deepstack_layers)
	// We'll infer the actual count after loading tensors

	hiddenSize := int(c.Uint("vision.embedding_length", 1280))
	patchSize := int(c.Uint("vision.patch_size", 14))
	numChannels := int(c.Uint("vision.num_channels", 3))

	// For Qwen3-VL Split, pre-initialize DeepstackMerger array for vision_bridge
	// llama.cpp has n_deepstack_layers = 3 (layers 8, 16, 24)
	nDeepstack := 3
	return &VisionModel{
		Layers: make([]VisionEncoderLayer, c.Uint("vision.block_count", 32)),
		DeepstackMerger: func() []*VisionPatchMerger {
			arr := make([]*VisionPatchMerger, nDeepstack)
			for i := range arr {
				arr[i] = &VisionPatchMerger{}
			}
			return arr
		}(),
		VisionOptions: VisionOptions{
			hiddenSize:        hiddenSize,
			numHeads:          int(c.Uint("vision.attention.head_count", 16)),
			patchSize:         patchSize,
			numChannels:       numChannels,
			eps:               c.Float("vision.attention.layer_norm_epsilon", 1e-6),
			ropeTheta:         c.Float("vision.rope.freq_base", 1000000.0),
			spatialMergeSize:  int(c.Uint("vision.spatial_merge_size", 2)),
			temporalPatchSize: int(c.Uint("vision.temporal_patch_size", 2)),
			gridPerSide:       int(math.Sqrt(float64(c.Uint("vision.num_positional_embeddings", 2304)))),
			mropeSections: slices.Collect(func(yield func(int) bool) {
				for _, section := range c.Ints("vision.mrope_sections", []int32{24, 20, 20}) {
					if !yield(int(section)) {
						return
					}
				}
			}),
			isSplitArchitecture: false,
			// For Split GGUFs: deepstack features go to first N LLM layers
			// This matches llama.cpp: if (ubatch.embd && il < n_deepstack_layers)
			deepstackVisualIndexes: []int32{8, 16, 24}, // Deepstack extraction layers matching GGUF tensor naming
		},
	}
}

// calculateDeepstackLayerIDs determines which vision encoder layers to extract deepstack features from
// based on the total number of vision encoder layers. This is used for unified models where
// the layer IDs are not embedded in tensor names.
//
// Qwen3-VL uses 3 deepstack layers at approximately 1/4, 1/2, and 3/4 of the vision encoder depth:
// - 24 layers (4B): layers 5, 11, 17  (n_layers * [6/24, 12/24, 18/24] - 1)
// - 27 layers (8B): layers 8, 16, 24  (n_layers * [9/27, 18/27, 27/27] - 1)
// - 32 layers:      layers 8, 16, 24  (same as 27 for historical reasons)
func calculateDeepstackLayerIDs(nLayers int, nDeepstack int) []int32 {
	if nDeepstack != 3 {
		// Non-standard deepstack count - use evenly spaced layers
		result := make([]int32, nDeepstack)
		for i := 0; i < nDeepstack; i++ {
			// Space evenly from 1/4 to end
			ratio := float64(i+1) / float64(nDeepstack+1) * 1.5 // 1/4 to 3/4 range
			result[i] = int32(float64(nLayers) * ratio)
			if result[i] >= int32(nLayers) {
				result[i] = int32(nLayers) - 1
			}
		}
		return result
	}

	// Standard Qwen3-VL deepstack layer IDs based on vision encoder size
	switch {
	case nLayers <= 24:
		// 4B model: 24 vision layers → extract from layers 5, 11, 17
		return []int32{5, 11, 17}
	case nLayers <= 27:
		// 8B model: 27 vision layers → extract from layers 8, 16, 24
		return []int32{8, 16, 24}
	default:
		// 32+ layers: use same as 27 (layer 24 is the "last significant" deepstack layer)
		return []int32{8, 16, 24}
	}
}

// InferOptionsFromTensors updates VisionOptions by inferring dimensions from actual tensor shapes.
// This is used when config values are incorrect or missing (e.g., split GGUF models).
func (m *VisionModel) InferOptionsFromTensors() {
	// Infer hiddenSize from a layer norm bias tensor shape [hiddenSize]
	if len(m.Layers) > 0 && m.Layers[0].Norm1 != nil && m.Layers[0].Norm1.Bias != nil {
		dims := m.Layers[0].Norm1.Bias.Shape()
		if len(dims) > 0 && dims[0] > 0 {
			m.hiddenSize = int(dims[0])
		}
	}

	// Detect split model architecture from PatchEmbedding shape
	// Unified: [16, 16, 2, 3456] = [kH, kW, temporal, channels*hidden] - 3D conv
	// Split:   [16, 16, 3, 1152] = [kH, kW, channels, hidden] - effectively 2D conv

	// Check if Conv2D (split model) is available first
	if m.PatchEmbedding2D != nil && m.PatchEmbedding2D.Weight != nil {
		m.isSplitArchitecture = true
		m.temporalPatchSize = 1
		dims := m.PatchEmbedding2D.Weight.Shape()
		if len(dims) >= 4 {
			kH, kW := int(dims[0]), int(dims[1])
			if kH == kW && kH > 0 {
				m.patchSize = kH
			}
		}
	} else if m.PatchEmbedding != nil && m.PatchEmbedding.Weight != nil {
		dims := m.PatchEmbedding.Weight.Shape()
		slog.Debug("InferOptionsFromTensors checking PatchEmbedding", "shape", dims, "len", len(dims))

		if len(dims) == 2 {
			// 2D weight [patchDim, hiddenSize] - this is from load-time reshape of split GGUF
			// Shape [768, 1152] means split architecture with patchDim=768, hiddenSize=1152
			patchDim, hiddenSize := int(dims[0]), int(dims[1])
			if hiddenSize == m.hiddenSize && patchDim > 0 {
				m.isSplitArchitecture = true
				m.temporalPatchSize = 1 // 2D reshaped weights don't include temporal dimension
				// Infer patchSize from patchDim = numChannels * temporalPatchSize * patchSize * patchSize
				// 768 = 3 * 16 * 16, so patchSize = sqrt(patchDim / numChannels)
				patchArea := patchDim / m.numChannels
				for ps := 1; ps <= 64; ps++ {
					if ps*ps == patchArea {
						if ps > m.patchSize {
							m.storagePatchSize = ps
							slog.Debug("Detected padded split weights", "patchSize", m.patchSize, "storagePatchSize", m.storagePatchSize)
						} else {
							m.patchSize = ps
						}
						break
					}
				}
				slog.Debug("Detected split architecture from 2D weight",
					"patchDim", patchDim, "hiddenSize", hiddenSize, "patchSize", m.patchSize, "storagePatchSize", m.storagePatchSize)
			}
		} else if len(dims) >= 4 {
			kH, kW, dim2, dim3 := int(dims[0]), int(dims[1]), int(dims[2]), int(dims[3])

			// Set patchSize from kernel dimensions
			if kH == kW && kH > 0 {
				m.patchSize = kH
			}

			// Detect architecture variant:
			// Unified: dim3 = numChannels * hiddenSize (e.g., 3456 = 3 * 1152)
			// Split: dim3 = hiddenSize (e.g., 1152), dim2 = numChannels (e.g., 3)
			if dim3 == m.hiddenSize && dim2 == m.numChannels {
				// Split architecture: [kH, kW, channels, hiddenSize]
				m.isSplitArchitecture = true
				m.temporalPatchSize = 1 // Split 2D weights process single frames (no temporal dimension)
				// Create Conv2D using the same weight/bias tensors
				m.PatchEmbedding2D = &nn.Conv2D{
					Weight: m.PatchEmbedding.Weight,
					Bias:   m.PatchEmbedding.Bias,
				}
			} else if dim3 == m.numChannels*m.hiddenSize {
				// Unified architecture: [kH, kW, temporal, channels*hiddenSize]
				m.isSplitArchitecture = false
				m.temporalPatchSize = dim2
			}
		}
	}

	// Verify numHeads is compatible with hiddenSize
	// For Qwen3VL vision, numHeads=16 with headDim=72 (1152/16=72)
	// Only override if current config produces invalid headDim
	if m.hiddenSize > 0 && m.numHeads > 0 {
		headDim := m.hiddenSize / m.numHeads
		// If headDim is not a reasonable value (64, 72, 80, 96, 128), try to fix
		validHeadDims := map[int]bool{64: true, 72: true, 80: true, 96: true, 128: true}
		if !validHeadDims[headDim] || m.hiddenSize%m.numHeads != 0 {
			// Try common numHeads values for this hiddenSize
			for _, tryHeads := range []int{16, 12, 8, 20, 24} {
				if m.hiddenSize%tryHeads == 0 {
					tryHeadDim := m.hiddenSize / tryHeads
					if validHeadDims[tryHeadDim] {
						m.numHeads = tryHeads
						break
					}
				}
			}
		}
	}

	// Count actual populated layers
	populatedCount := 0
	for _, layer := range m.Layers {
		if layer.Norm1 != nil {
			populatedCount++
		}
	}

	// Infer deepstack configuration from loaded tensors
	// Count how many DeepstackMerger tensors are loaded (v.deepstack.8, v.deepstack.16, etc.)
	nDeepstack := 0
	if m.DeepstackMerger != nil {
		for _, merger := range m.DeepstackMerger {
			if merger != nil && merger.FC2 != nil {
				nDeepstack++
			}
		}
	}

	// Initialize empty DeepstackMerger array if not already done
	if m.DeepstackMerger == nil && nDeepstack == 0 {
		// Try to detect from struct field tags
		// For now, we'll just create an empty array and it will be populated by vision_bridge
		m.DeepstackMerger = []*VisionPatchMerger{}
	}

	// For llama.cpp compatibility: deepstack features extracted from vision encoder layers
	// Use the detected layer IDs from GGUF (stored in deepstackLayerIDs by loadDeepstackMergerWeights)
	// This maps vision encoder layers to DeepstackMerger indices
	if nDeepstack > 0 && len(m.deepstackLayerIDs) == nDeepstack {
		// Use actual layer IDs from GGUF (e.g., [5, 11, 17] for 4B or [8, 16, 24] for 8B)
		m.deepstackVisualIndexes = make([]int32, nDeepstack)
		for i, layerID := range m.deepstackLayerIDs {
			m.deepstackVisualIndexes[i] = int32(layerID)
		}
		slog.Debug("Using detected deepstack layer IDs for vision extraction",
			"n_deepstack_layers", nDeepstack,
			"vision_extraction_layers", m.deepstackVisualIndexes)
	} else if nDeepstack > 0 {
		// For unified models: layer IDs are NOT in tensor names (v.deepstack_merger.0 etc.)
		// Calculate layer IDs based on vision encoder layer count
		// Qwen3-VL uses approximately evenly spaced layers for deepstack
		nLayers := len(m.Layers)
		m.deepstackVisualIndexes = calculateDeepstackLayerIDs(nLayers, nDeepstack)
		slog.Debug("Calculated deepstack layer IDs for unified model",
			"n_vision_layers", nLayers,
			"n_deepstack_layers", nDeepstack,
			"vision_extraction_layers", m.deepstackVisualIndexes)
	}
	if populatedCount > 0 && populatedCount != len(m.Layers) {
		// Resize layers array to match actual populated count
		m.Layers = m.Layers[:populatedCount]
	}
}
