package qwen3vl

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"log/slog"
	"slices"
	"sort"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
)

// detectDeepstackLayerIDs probes the backend for v.deepstack.N.fc1.weight tensors
// and returns the sorted list of layer IDs found (e.g., [5, 11, 17] for 4B or [8, 16, 24] for 8B)
func detectDeepstackLayerIDs(backend ml.Backend) []int {
	var layerIDs []int

	// Probe common deepstack layer indices used by different model sizes
	// 8B model uses layers 8, 16, 24 (with 32 total vision encoder layers)
	// 4B model uses layers 5, 11, 17 (with 24 total vision encoder layers)
	// 2B model uses layers 4, 9, 14 (with 16 total vision encoder layers)
	// We check a wide range to support various model configurations
	candidateLayers := []int{4, 5, 6, 8, 9, 10, 11, 12, 14, 16, 17, 18, 20, 24, 28, 32}

	for _, layerID := range candidateLayers {
		tensorName := fmt.Sprintf("v.deepstack.%d.fc1.weight", layerID)
		if tensor := backend.Get(tensorName); tensor != nil {
			layerIDs = append(layerIDs, layerID)
			slog.Debug("Found deepstack layer", "layerID", layerID, "tensor", tensorName)
		}
	}

	sort.Ints(layerIDs)
	return layerIDs
}

// loadDeepstackMergerWeights manually loads FC weights from GGUF into DeepstackMerger array
// This is needed because vision_bridge array indices (0,1,2) don't match GGUF layer IDs
func (m *Model) loadDeepstackMergerWeights(layerIDs []int) {
	if len(layerIDs) == 0 {
		slog.Debug("No deepstack layers found in GGUF")
		return
	}

	// Ensure DeepstackMerger array is properly sized
	if m.VisionModel.DeepstackMerger == nil || len(m.VisionModel.DeepstackMerger) != len(layerIDs) {
		m.VisionModel.DeepstackMerger = make([]*VisionPatchMerger, len(layerIDs))
		for i := range m.VisionModel.DeepstackMerger {
			m.VisionModel.DeepstackMerger[i] = &VisionPatchMerger{}
		}
	}

	for idx, layerID := range layerIDs {
		prefix := fmt.Sprintf("v.deepstack.%d", layerID)

		// Load FC1
		fc1WeightName := prefix + ".fc1.weight"
		if fc1Weight := m.Backend().Get(fc1WeightName); fc1Weight != nil {
			fc1BiasName := prefix + ".fc1.bias"
			fc1Bias := m.Backend().Get(fc1BiasName)
			m.VisionModel.DeepstackMerger[idx].FC1 = &nn.Linear{
				Weight: fc1Weight,
				Bias:   fc1Bias,
			}
			slog.Debug("Loaded DeepstackMerger FC1", "layer", layerID, "idx", idx, "shape", fc1Weight.Shape())
		} else {
			slog.Warn("DeepstackMerger FC1 weight not found", "layer", layerID, "name", fc1WeightName)
		}

		// Load FC2
		fc2WeightName := prefix + ".fc2.weight"
		if fc2Weight := m.Backend().Get(fc2WeightName); fc2Weight != nil {
			fc2BiasName := prefix + ".fc2.bias"
			fc2Bias := m.Backend().Get(fc2BiasName)
			m.VisionModel.DeepstackMerger[idx].FC2 = &nn.Linear{
				Weight: fc2Weight,
				Bias:   fc2Bias,
			}
			slog.Debug("Loaded DeepstackMerger FC2", "layer", layerID, "idx", idx, "shape", fc2Weight.Shape())
		} else {
			slog.Warn("DeepstackMerger FC2 weight not found", "layer", layerID, "name", fc2WeightName)
		}

		// Load Norm (optional but recommended)
		normWeightName := prefix + ".norm.weight"
		if normWeight := m.Backend().Get(normWeightName); normWeight != nil {
			normBiasName := prefix + ".norm.bias"
			normBias := m.Backend().Get(normBiasName)
			m.VisionModel.DeepstackMerger[idx].Norm = &nn.LayerNorm{
				Weight: normWeight,
				Bias:   normBias,
			}
			slog.Debug("Loaded DeepstackMerger Norm", "layer", layerID, "idx", idx)
		}
	}

	// Store the detected layer IDs for use in vision processing
	m.VisionModel.deepstackLayerIDs = layerIDs

	// CRITICAL: Also update deepstackVisualIndexes in VisionOptions
	// This is what VisionModel.Forward() uses to know which layers to extract deepstack from
	// Without this, Forward() still uses hardcoded [8,16,24] and produces nil tensors!
	deepstackIndexes := make([]int32, len(layerIDs))
	for i, id := range layerIDs {
		deepstackIndexes[i] = int32(id)
	}
	m.VisionModel.deepstackVisualIndexes = deepstackIndexes

	slog.Debug("Deepstack layers detected and loaded", "layerIDs", layerIDs, "deepstackVisualIndexes", deepstackIndexes, "count", len(layerIDs))
}

type Model struct {
	model.Base
	model.TextProcessor

	*TextModel
	*VisionModel `gguf:"v"`

	// Multimodal projector for main vision output (mm.0 -> mm.2 MLP)
	// These are loaded from "mm.0.*" and "mm.2.*" tensors in split GGUF
	// Used instead of PatchMerger for split models
	MultimodalProjectorFC1 *nn.Linear `gguf:"mm.0"` // [4608, 4608] with GELU
	MultimodalProjectorFC2 *nn.Linear `gguf:"mm.2"` // [4608, 4096]

	ImageProcessor

	positionCache []int32

	// Split vision model support
	visionReady   bool       // true if vision encoder is ready (either loaded from main file or separate file)
	visionPath    string     // path to separate vision GGUF file (empty if embedded in main file)
	visionBackend ml.Backend // backend for vision model when loaded separately
}

// HasProjector returns true if the model has a vision projector (vision capability)
// HasProjector checks if the vision encoder is actually loaded with tensors.
// For split models, this returns false until the vision GGUF is loaded.
func (m *Model) HasProjector() bool {
	// Check if a critical vision tensor is actually populated
	// Unified models use Conv3D (PatchEmbedding), split models use Conv2D (PatchEmbedding2D)
	return m.VisionModel != nil && (m.VisionModel.PatchEmbedding != nil || m.VisionModel.PatchEmbedding2D != nil)
}

// ensureVisionReady loads the vision encoder if it hasn't been loaded yet.
// For split GGUF models, this loads vision tensor data from the separate file
// into the main backend's pre-allocated tensors using LoadSecondary.
// For unified models, this just marks vision as ready.
func (m *Model) ensureVisionReady() error {
	if m.visionReady {
		return nil
	}

	slog.Debug("ensureVisionReady", "hasProjector", m.HasProjector(), "visionPath", m.visionPath)

	// If vision layers are already populated (unified model), mark as ready
	if m.HasProjector() {
		// Infer correct vision dimensions from actual tensor shapes (fixes incorrect config defaults)
		m.VisionModel.InferOptionsFromTensors()

		// Auto-detect and load deepstack layers from GGUF
		layerIDs := detectDeepstackLayerIDs(m.Backend())
		if len(layerIDs) > 0 {
			m.loadDeepstackMergerWeights(layerIDs)
		}

		// Manually load the second patch embedding kernel (v.patch_embd.weight.1) for split models
		// The GGUF tag system cannot handle this tensor name format
		if m.VisionModel.PatchEmbedding1 == nil {
			if patchEmbd1Weight := m.Backend().Get("v.patch_embd.weight.1"); patchEmbd1Weight != nil {
				m.VisionModel.PatchEmbedding1 = &nn.Linear{
					Weight: patchEmbd1Weight,
				}
				slog.Info("Loaded second patch embedding kernel (early path)", "tensor", "v.patch_embd.weight.1", "shape", patchEmbd1Weight.Shape())
			}
		}

		// Sync image preprocessing parameters from model config.
		// For split GGUF setups, vision-specific config (e.g., image_mean/std) may
		// live in the vision GGUF, so we refresh from the merged backend config.
		cfg := m.Backend().Config()
		if v := cfg.Floats("vision.image_mean"); len(v) > 0 {
			m.ImageProcessor.imageMean = v
		}
		if v := cfg.Floats("vision.image_std"); len(v) > 0 {
			m.ImageProcessor.imageStd = v
		}

		// Sync temporalPatchSize from VisionModel to ImageProcessor.
		// For split models, use the ORIGINAL temporalPatchSize=1 from the GGUF
		// (the split kernels expect 768-dim input, not 1536-dim).
		m.ImageProcessor.temporalPatchSize = m.VisionModel.temporalPatchSize
		m.ImageProcessor.storagePatchSize = m.VisionModel.storagePatchSize
		slog.Debug("Vision ready", "hiddenSize", m.VisionModel.hiddenSize, "numHeads", m.VisionModel.numHeads, "layers", len(m.VisionModel.Layers), "isSplitArchitecture", m.VisionModel.isSplitArchitecture, "temporalPatchSize", m.VisionModel.temporalPatchSize)
		m.visionReady = true
		return nil
	}

	// If no vision path specified, vision is not available
	if m.visionPath == "" {
		return model.ErrNoVisionModel
	}

	// CRITICAL: Register tensor name aliases BEFORE LoadSecondary!
	// Aliases must be active when tensors are loaded so they map to correct struct fields.
	// Split models (e.g., unsloth) use different naming conventions than the Go struct.
	slog.Debug("Registering split model tensor aliases BEFORE load")

	// Embedding tensors: patch_embd → patch_embed, position_embd → position_embed
	m.Backend().RegisterTensorAlias("v.patch_embed", "v.patch_embd")
	m.Backend().RegisterTensorAlias("v.position_embed", "v.position_embd")

	// Layer norm tensors: ln1 → norm1, ln2 → norm2
	// Split GGUF has v.blk.0.ln1.weight, model expects v.blk.0.norm1.weight
	m.Backend().RegisterTensorAlias("v.blk.*.norm1", "v.blk.*.ln1")
	m.Backend().RegisterTensorAlias("v.blk.*.norm2", "v.blk.*.ln2")

	// MLP tensors: ffn_up → mlp.linear_fc1, ffn_down → mlp.linear_fc2
	// Split GGUF has v.blk.0.ffn_up.weight, model expects v.blk.0.mlp.linear_fc1.weight
	m.Backend().RegisterTensorAlias("v.blk.*.mlp.linear_fc1", "v.blk.*.ffn_up")
	m.Backend().RegisterTensorAlias("v.blk.*.mlp.linear_fc2", "v.blk.*.ffn_down")

	// Deepstack merger tensors: deepstack → deepstack_merger
	m.Backend().RegisterTensorAlias("v.deepstack_merger", "v.deepstack")

	slog.Debug("Split model tensor aliases registered, now loading secondary GGUF")

	// Load vision tensor data from separate GGUF file into main backend
	// LoadSecondary creates tensors that don't exist and loads data into them
	slog.Debug("Loading split vision model into main backend", "path", m.visionPath)

	err := m.Backend().LoadSecondary(context.Background(), m.visionPath, nil)
	if err != nil {
		slog.Error("Failed to load vision model from secondary GGUF", "error", err)
		return err
	}

	slog.Debug("Split vision model loaded from GGUF, re-populating struct")

	// IMPORTANT: After LoadSecondary, tensors exist in backend but VisionModel struct
	// fields haven't been bound to them. Re-populate the VisionModel field.
	// The "v" tag corresponds to `gguf:"v"` on the VisionModel field.
	slog.Debug("Re-populating VisionModel struct after LoadSecondary")
	if m.VisionModel == nil {
		m.VisionModel = newVisionModel(m.Backend().Config())
	}

	// Auto-detect deepstack layer IDs from GGUF tensors (varies by model size)
	// e.g., 8B uses [8, 16, 24], 4B uses [5, 11, 17]
	layerIDs := detectDeepstackLayerIDs(m.Backend())
	nDeepstack := len(layerIDs)
	if nDeepstack == 0 {
		// Fallback to 3 empty slots for vision_bridge
		nDeepstack = 3
		slog.Warn("No deepstack layers detected, using fallback count", "nDeepstack", nDeepstack)
	} else {
		slog.Debug("Detected deepstack layers from GGUF", "layerIDs", layerIDs, "count", nDeepstack)
	}

	// Pre-initialize DeepstackMerger array for vision_bridge
	m.VisionModel.DeepstackMerger = make([]*VisionPatchMerger, nDeepstack)
	for i := range m.VisionModel.DeepstackMerger {
		m.VisionModel.DeepstackMerger[i] = &VisionPatchMerger{}
	}

	model.RepopulateField(m.Base, m.VisionModel, "v")

	// Load deepstack weights using detected layer IDs
	if len(layerIDs) > 0 {
		m.loadDeepstackMergerWeights(layerIDs)
	}

	// Manually load the second patch embedding kernel (v.patch_embd.weight.1)
	// The GGUF tag system cannot handle this tensor because it expects v.patch_embd.1.weight
	// but the actual tensor is named v.patch_embd.weight.1
	patchEmbd1Weight := m.Backend().Get("v.patch_embd.weight.1")
	slog.Info("Looking for second patch embedding kernel", "tensor", "v.patch_embd.weight.1", "found", patchEmbd1Weight != nil)
	if patchEmbd1Weight != nil {
		m.VisionModel.PatchEmbedding1 = &nn.Linear{
			Weight: patchEmbd1Weight,
		}
		slog.Info("Loaded second patch embedding kernel", "tensor", "v.patch_embd.weight.1", "shape", patchEmbd1Weight.Shape())
	} else {
		slog.Warn("No second patch embedding kernel found - dual kernel summing disabled", "tensor", "v.patch_embd.weight.1")
	}

	// Infer correct vision dimensions from actual tensor shapes
	m.VisionModel.InferOptionsFromTensors()

	// Verify that the vision model is now ready
	if !m.HasProjector() {
		slog.Error("Vision tensors still not populated after LoadSecondary and RepopulateField",
			"patchEmbedding", m.VisionModel.PatchEmbedding,
			"layers", len(m.VisionModel.Layers))
		return model.ErrNoVisionModel
	}

	m.visionReady = true

	// Sync image preprocessing parameters from merged config now that vision GGUF is loaded.
	cfg := m.Backend().Config()
	if v := cfg.Floats("vision.image_mean"); len(v) > 0 {
		m.ImageProcessor.imageMean = v
	}
	if v := cfg.Floats("vision.image_std"); len(v) > 0 {
		m.ImageProcessor.imageStd = v
	}
	// Keep temporal/storage patch sizes in sync with the inferred vision options.
	m.ImageProcessor.temporalPatchSize = m.VisionModel.temporalPatchSize
	m.ImageProcessor.storagePatchSize = m.VisionModel.storagePatchSize

	slog.Debug("Split vision model loaded", "layers", len(m.VisionModel.Layers), "hiddenSize", m.VisionModel.hiddenSize, "temporalPatchSize", m.VisionModel.temporalPatchSize)
	return nil
}

func (m *Model) EncodeMultimodal(ctx ml.Context, multimodalData []byte) ([]input.Multimodal, error) {
	// Lazy load vision encoder if needed (supports split GGUF models)
	if err := m.ensureVisionReady(); err != nil {
		return nil, err
	}

	if !m.HasProjector() {
		return nil, model.ErrNoVisionModel
	}

	img, _, err := image.Decode(bytes.NewReader(multimodalData))
	if err != nil {
		return nil, err
	}

	// Always use patch-based inputs produced by ImageProcessor.
	// This keeps tensor memory layout consistent with ggml's expectations and matches the original
	// Qwen3VL runtime path (split and non-split).
	var (
		pixelValues ml.Tensor
		grid        *Grid
	)
	pixelValues, grid, err = m.ProcessImage(ctx, img)
	if err != nil {
		return nil, err
	}

	// Copy multimodal projector references to VisionModel for split model support
	// These are the mm.0/mm.2 tensors that project main vision (separate from deepstack FC)
	if m.MultimodalProjectorFC1 != nil && m.MultimodalProjectorFC2 != nil {
		m.VisionModel.MultimodalFC1 = m.MultimodalProjectorFC1
		m.VisionModel.MultimodalFC2 = m.MultimodalProjectorFC2
		slog.Debug("Copied mm.0/mm.2 projectors to VisionModel",
			"FC1_shape", m.MultimodalProjectorFC1.Weight.Shape(),
			"FC2_shape", m.MultimodalProjectorFC2.Weight.Shape())
	}

	// Calculate tensor dimensions
	visionOutputs, deepstackVisualEmbeds := m.VisionModel.Forward(ctx, pixelValues, grid)

	// Defensive: ensure returned tensors are not views with view_src chains.
	// This helps avoid subtle lifetime/aliasing issues that can manifest as unstable outputs.
	if visionOutputs != nil {
		visionOutputs = visionOutputs.Contiguous(ctx, visionOutputs.Shape()...)
	}
	for i := range deepstackVisualEmbeds {
		if deepstackVisualEmbeds[i] != nil {
			deepstackVisualEmbeds[i] = deepstackVisualEmbeds[i].Contiguous(ctx, deepstackVisualEmbeds[i].Shape()...)
		}
	}

	// For SPLIT models only: concatenate main + deepstack embeddings into single tensor
	// This is needed because split models pass vision through a single tensor interface
	// Format: [main (n_embd) | deepstack_0 (n_embd) | deepstack_1 (n_embd) | deepstack_2 (n_embd)]
	// For UNIFIED models: keep embeddings separate (original Ollama behavior)
	if m.VisionModel.isSplitArchitecture && len(deepstackVisualEmbeds) > 0 {
		// Concatenate along the feature dimension (dim=0 for column-major GGML tensors)
		allEmbeds := []ml.Tensor{visionOutputs}
		allEmbeds = append(allEmbeds, deepstackVisualEmbeds...)

		concatenated := allEmbeds[0].Concat(ctx, allEmbeds[1], 0)
		for i := 2; i < len(allEmbeds); i++ {
			concatenated = concatenated.Concat(ctx, allEmbeds[i], 0)
		}
		concatenated = concatenated.Contiguous(ctx, concatenated.Shape()...)
		slog.Debug("Split model: Concatenated vision + deepstack embeddings",
			"main_shape", visionOutputs.Shape(),
			"n_deepstack", len(deepstackVisualEmbeds),
			"concatenated_shape", concatenated.Shape())
		return []input.Multimodal{{Tensor: concatenated, Data: grid}}, nil
	}

	// Unified model: return embeddings separately (original Ollama behavior)
	if visionOutputs != nil {
		visionOutputs = visionOutputs.Contiguous(ctx, visionOutputs.Shape()...)
	}
	mm := []input.Multimodal{{Tensor: visionOutputs, Data: grid}}
	for i := range deepstackVisualEmbeds {
		mm = append(mm, input.Multimodal{Tensor: deepstackVisualEmbeds[i]})
	}
	return mm, nil
}

var (
	tokenVision      int32 = 151655
	tokenVisionStart int32 = 151652
	tokenVisionEnd   int32 = 151653
)

type modelInput struct {
	*input.Input
	position int32
}

// PostTokenize arranges Qwen 3 VL's inputs for the forward pass
func (m *Model) PostTokenize(inputs []*input.Input) ([]*input.Input, error) {
	m.positionCache = m.positionCache[:0]
	return slices.Collect(func(yield func(*input.Input) bool) {
		for i := range inputs {
			s := []modelInput{{Input: inputs[i]}}
			if mm := inputs[i].Multimodal; mm != nil {
				t := mm[0].Tensor
				s = slices.Repeat([]modelInput{
					{
						position: int32(i + 1),
						Input:    &input.Input{Token: tokenVision},
					},
				}, t.Dim(1)+1+1)

				s[0] = modelInput{
					Input:    &input.Input{Token: tokenVisionStart},
					position: int32(i),
				}

				s[len(s)-1] = modelInput{
					Input:    &input.Input{Token: tokenVisionEnd},
					position: int32(i + mm[0].Data.(*Grid).Width/m.spatialMergeSize + 1),
				}

				s[1] = modelInput{
					Input: &input.Input{
						Token:          tokenVision,
						Multimodal:     inputs[i].Multimodal,
						MultimodalHash: inputs[i].MultimodalHash,
						SameBatch:      t.Dim(1),
					},
					position: int32(i + 1),
				}
			}

			for _, e := range s {
				position := e.position
				if position == 0 && len(m.positionCache) > 0 {
					position = m.positionCache[len(m.positionCache)-1] + 1
				}

				m.positionCache = append(m.positionCache, position)
				if !yield(e.Input) {
					return
				}
			}
		}
	}), nil
}

func (m *Model) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	// ggml mrope requires 4 positions per token: [time, height, width, extra]
	positionSlice := slices.Collect(makeSlice2D[int32](4, len(batch.Positions)))
	for i, id := range batch.Positions {
		if id < int32(len(m.positionCache)) {
			id = m.positionCache[id]
		} else if len(m.positionCache) > 0 {
			id = id - int32(len(m.positionCache)) + m.positionCache[len(m.positionCache)-1] + 1
		}

		positionSlice[0][i] = id
		positionSlice[1][i] = id
		positionSlice[2][i] = id
		// positionSlice[3] is intentionally left as zeros
	}

	hiddenStates := m.TextModel.TokenEmbedding.Forward(ctx, batch.Inputs).Duplicate(ctx)

	var deepstackVisualEmbeds []ml.Tensor
	for _, mi := range batch.Multimodal {
		visionOutputs := mi.Multimodal[0].Tensor

		// Check if embeddings are concatenated (n_embd_inp format)
		// GGML column-major: shape [features, tokens] - features in dim(0)
		nEmbdFull := visionOutputs.Dim(0)
		nEmbd := m.TextModel.Options.hiddenSize
		var extractedDeepstacks []ml.Tensor
		if nEmbdFull > nEmbd && nEmbdFull%nEmbd == 0 {
			// Split concatenated embeddings: [main | deepstack_0 | deepstack_1 | ...] in feature dimension
			nDeepstackLayers := nEmbdFull/nEmbd - 1
			nTokens := visionOutputs.Dim(1)
			slog.Debug("Detected concatenated vision embeddings - splitting",
				"full_dim", nEmbdFull, "n_embd", nEmbd, "n_tokens", nTokens, "n_deepstack", nDeepstackLayers)

			// GGML tensors are column-major. A flattened View() cannot correctly slice rows
			// across all columns because the memory for [rows, tokens] is not laid out as a
			// single contiguous span per row-block. Use Chunk() to split along dim(0).
			chunks := visionOutputs.Chunk(ctx, 0, nEmbd)
			if len(chunks) != nDeepstackLayers+1 {
				panic(fmt.Sprintf("unexpected concatenated vision chunk count: got=%d want=%d (full_dim=%d n_embd=%d)", len(chunks), nDeepstackLayers+1, nEmbdFull, nEmbd))
			}

			// Main vision is chunk 0; deepstacks follow in order.
			visionOutputs = chunks[0].Contiguous(ctx, nEmbd, nTokens)
			if nDeepstackLayers > 0 {
				extractedDeepstacks = make([]ml.Tensor, nDeepstackLayers)
				for i := 0; i < nDeepstackLayers; i++ {
					extractedDeepstacks[i] = chunks[i+1].Contiguous(ctx, nEmbd, nTokens)
				}
			}
		}

		// Copy main vision embeddings into hiddenStates
		ctx.Forward(visionOutputs.Copy(ctx, hiddenStates.View(ctx, mi.Index*hiddenStates.Stride(1), visionOutputs.Dim(0)*visionOutputs.Dim(1))))

		if grid, ok := mi.Multimodal[0].Data.(*Grid); ok {
			w := grid.Width / m.spatialMergeSize
			// M-RoPE position encoding for images:
			// pos[0] already has correct value from positionCache (temporal position)
			// pos[1] and pos[2] are incremented by row/col for 2D spatial encoding
			for i := range visionOutputs.Dim(1) {
				positionSlice[1][mi.Index+i] += int32(i / w)
				positionSlice[2][mi.Index+i] += int32(i % w)
			}
		}

		// Only process additional multimodal elements if deepstackVisualEmbeds wasn't already extracted
		// from concatenated tensor above (split model path)
		if deepstackVisualEmbeds == nil && len(mi.Multimodal) > 1 {
			deepstackVisualEmbeds = make([]ml.Tensor, len(mi.Multimodal[1:]))
			for i, mm := range mi.Multimodal[1:] {
				deepstackVisualEmbeds[i] = ctx.Input().Zeros(mm.Tensor.DType(), hiddenStates.Shape()...)
				ctx.Forward(mm.Tensor.Copy(ctx, deepstackVisualEmbeds[i].View(ctx, mi.Index*deepstackVisualEmbeds[i].Stride(1), mm.Tensor.Dim(0)*mm.Tensor.Dim(1))))
			}
		}

		// Expand extracted deepstacks (from concatenated tensor) to hiddenStates shape
		// Each deepstack has shape [n_embd, n_image_tokens], need to expand to [n_embd, batch_size]
		if len(extractedDeepstacks) > 0 && deepstackVisualEmbeds == nil {
			deepstackVisualEmbeds = make([]ml.Tensor, len(extractedDeepstacks))
			for i, ds := range extractedDeepstacks {
				// Create zeros tensor with same shape as hiddenStates
				deepstackVisualEmbeds[i] = ctx.Input().Zeros(ds.DType(), hiddenStates.Shape()...)
				// Copy deepstack embeddings into the correct position (where image tokens are)
				ctx.Forward(ds.Copy(ctx, deepstackVisualEmbeds[i].View(ctx, mi.Index*deepstackVisualEmbeds[i].Stride(1), ds.Dim(0)*ds.Dim(1))))
			}
			slog.Debug("Expanded deepstacks to hiddenStates shape",
				"n_deepstacks", len(deepstackVisualEmbeds),
				"hiddenStates_shape", hiddenStates.Shape(),
				"ds_shape", extractedDeepstacks[0].Shape())
		}
	}

	positions := ctx.Input().FromInts(slices.Concat(positionSlice...), len(positionSlice[0])*len(positionSlice))
	for i, layer := range m.TextModel.Layers {
		if m.Cache != nil {
			m.Cache.SetLayer(i)
		}

		var outputs ml.Tensor
		if i == len(m.TextModel.Layers)-1 {
			outputs = batch.Outputs
		}

		hiddenStates = layer.Forward(ctx, hiddenStates, positions, outputs, m.Cache, m.Options)
		if i < len(deepstackVisualEmbeds) {
			hiddenStates = hiddenStates.Add(ctx, deepstackVisualEmbeds[i])
		}
	}

	hiddenStates = m.OutputNorm.Forward(ctx, hiddenStates, 1e-06)
	return m.Output.Forward(ctx, hiddenStates), nil
}

func New(c fs.Config) (model.Model, error) {
	m := Model{
		TextProcessor: model.NewBytePairEncoding(
			&model.Vocabulary{
				Values: c.Strings("tokenizer.ggml.tokens"),
				Types:  c.Ints("tokenizer.ggml.token_type"),
				Merges: c.Strings("tokenizer.ggml.merges"),
				AddBOS: c.Bool("tokenizer.ggml.add_bos_token", false),
				BOS:    []int32{int32(c.Uint("tokenizer.ggml.bos_token_id"))},
				AddEOS: c.Bool("tokenizer.ggml.add_eos_token", false),
				EOS: append(
					[]int32{int32(c.Uint("tokenizer.ggml.eos_token_id"))},
					c.Ints("tokenizer.ggml.eos_token_ids")...,
				),
			},
			`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
		),
		TextModel:      newTextModel(c),
		VisionModel:    newVisionModel(c),
		ImageProcessor: newImageProcessor(c),
	}

	m.Cache = kvcache.NewCausalCache(func(ctx ml.Context, layer int, key, positions ml.Tensor) (ml.Tensor, error) {
		m.positionCache = nil
		positions = positions.Repeat(ctx, 1, 4).Reshape(ctx, -1)
		return m.Options.applyRotaryPositionalEmbedding(ctx, key, positions), nil
	})
	return &m, nil
}

// SetVisionPath sets the path to a separate vision GGUF file for split models.
// This should be called before any image processing if the vision model is
// stored in a separate file from the language model.
func (m *Model) SetVisionPath(path string) {
	m.visionPath = path
	m.visionReady = false // Reset ready flag to trigger re-loading
}

// Close cleans up resources used by the model, including the vision backend
// if it was loaded from a separate file.
func (m *Model) Close() {
	if m.visionBackend != nil {
		m.visionBackend.Close()
		m.visionBackend = nil
	}
}

func init() {
	model.Register("qwen3vl", New)
	model.Register("qwen3vlmoe", New)
}
