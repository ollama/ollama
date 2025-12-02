package mistral3

import (
	"bytes"
	"image"
	"slices"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
)

type Model struct {
	model.Base
	model.BytePairEncoding

	*TextModel
	*VisionModel         `gguf:"v"`
	*MultiModalProjector `gguf:"mm"`

	ImageProcessor
}

// Implement MultimodalProcessor interface
var _ model.MultimodalProcessor = (*Model)(nil)

// Implement TextProcessor interface
var _ model.TextProcessor = (*Model)(nil)

func New(c fs.Config) (model.Model, error) {
	m := &Model{
		BytePairEncoding: model.NewBytePairEncoding(
			&model.Vocabulary{
				Values: c.Strings("tokenizer.ggml.tokens"),
				Types:  c.Ints("tokenizer.ggml.token_type"),
				Merges: c.Strings("tokenizer.ggml.merges"),
				AddBOS: c.Bool("tokenizer.ggml.add_bos_token", true),
				BOS:    []int32{int32(c.Uint("tokenizer.ggml.bos_token_id"))},
				AddEOS: c.Bool("tokenizer.ggml.add_eos_token", false),
				EOS: append(
					[]int32{int32(c.Uint("tokenizer.ggml.eos_token_id"))},
					c.Ints("tokenizer.ggml.eos_token_ids")...,
				),
			},
			`[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
		),
		TextModel:           newTextModel(c),
		VisionModel:         newVisionModel(c),
		ImageProcessor:      newImageProcessor(c),
		MultiModalProjector: newMultiModalProjector(c),
	}

	m.Cache = kvcache.NewCausalCache(m.TextModel.Shift)

	return m, nil
}

type PatchMerger struct {
	MergingLayer *nn.Linear `gguf:"merging_layer"`
}

func (pm *PatchMerger) Forward(ctx ml.Context, visionOutputs ml.Tensor, size image.Point, spatialMergeSize int) ml.Tensor {
	d := visionOutputs.Dim(0)
	imageGrid := visionOutputs.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx).Reshape(ctx, size.X, size.Y, d)
	kernel := ctx.Input().Empty(ml.DTypeF32, spatialMergeSize, spatialMergeSize, d)
	patches := kernel.IM2Col(ctx, imageGrid, spatialMergeSize, spatialMergeSize, 0, 0, 1, 1)
	reshaped := patches.Reshape(ctx, d*spatialMergeSize*spatialMergeSize, patches.Dim(1)*patches.Dim(2))
	return pm.MergingLayer.Forward(ctx, reshaped)
}

type MultiModalProjector struct {
	Norm        *nn.RMSNorm  `gguf:"norm"`
	Linear1     *nn.Linear   `gguf:"linear_1"`
	Linear2     *nn.Linear   `gguf:"linear_2"`
	PatchMerger *PatchMerger `gguf:"patch_merger"`

	spatialMergeSize int
	eps              float32
	patchSize        int
}

func (p *MultiModalProjector) Forward(ctx ml.Context, visionOutputs ml.Tensor, size image.Point) (ml.Tensor, image.Point) {
	visionOutputs = p.Norm.Forward(ctx, visionOutputs, p.eps)
	patchSizes := image.Point{size.X / p.patchSize, size.Y / p.patchSize}
	visionOutputs = p.PatchMerger.Forward(ctx, visionOutputs, patchSizes, p.spatialMergeSize)
	visionOutputs = p.Linear1.Forward(ctx, visionOutputs)
	visionOutputs = visionOutputs.GELU(ctx)
	return p.Linear2.Forward(ctx, visionOutputs), image.Point{patchSizes.X / p.spatialMergeSize, patchSizes.Y / p.spatialMergeSize}
}

func newMultiModalProjector(c fs.Config) *MultiModalProjector {
	return &MultiModalProjector{
		spatialMergeSize: int(c.Uint("spatial_merge_size", 2)),
		eps:              c.Float("text_config.rms_norm_eps", 1e-5),
		patchSize:        int(c.Uint("vision.patch_size", 14)),
	}
}

func (m *Model) EncodeMultimodal(ctx ml.Context, multimodalData []byte) ([]input.Multimodal, error) {
	if len(m.VisionModel.Layers) == 0 {
		return nil, model.ErrNoVisionModel
	}

	image, _, err := image.Decode(bytes.NewReader(multimodalData))
	if err != nil {
		return nil, err
	}

	f32s, size, err := m.ImageProcessor.ProcessImage(image)
	if err != nil {
		return nil, err
	}

	pixelValues := ctx.Input().FromFloats(f32s, size.X, size.Y, m.ImageProcessor.numChannels)

	visionOutputs := m.VisionModel.Forward(ctx, pixelValues)
	features, size := m.MultiModalProjector.Forward(ctx, visionOutputs, size)

	// split into patches to be sent to the text transformer
	rows := make([]input.Multimodal, size.Y)
	for i := range rows {
		rows[i].Tensor = features.View(ctx, features.Stride(1)*size.X*i, features.Dim(0), features.Stride(1), size.X)
	}

	return rows, nil
}

// PostTokenize arranges Mistral 3's inputs for the forward pass
// In Mistral 3 and Pixtral, the input patches are arranged as follows:
// [IMG]...[IMG][IMG_BREAK][IMG]...[IMG][IMG_BREAK][IMG]...[IMG][IMG_END]
// Each sequence of [IMG]...[IMG] is a set of patches of vision embeddings
// that can be processed together.
func (m *Model) PostTokenize(inputs []*input.Input) ([]*input.Input, error) {
	var result []*input.Input
	for _, inp := range inputs {
		if len(inp.Multimodal) == 0 {
			result = append(result, inp)
		} else {
			for i, row := range inp.Multimodal {
				// [IMG]
				result = append(result, &input.Input{Token: 10, Multimodal: []input.Multimodal{{Tensor: row.Tensor}}, MultimodalHash: inp.MultimodalHash, SameBatch: row.Tensor.Dim(1)})
				result = append(result, slices.Repeat([]*input.Input{{Token: 10}}, row.Tensor.Dim(1)-1)...)
				if i == len(inp.Multimodal)-1 {
					// [IMG_END]
					result = append(result, &input.Input{Token: 13})
				} else {
					// [IMG_BREAK]
					result = append(result, &input.Input{Token: 12})
				}
			}
		}
	}

	return result, nil
}

func (m *Model) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	positions := ctx.Input().FromInts(batch.Positions, len(batch.Positions))
	positionsScale := m.getScale(ctx, batch.Positions)

	return m.TextModel.Forward(ctx, batch.Inputs, positions, positionsScale, batch.Outputs, batch, m.Cache), nil
}

func init() {
	model.Register("mistral3", New)
}
