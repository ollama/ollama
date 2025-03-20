package mistral3

import (
	"image"
	_ "image/jpeg"
	_ "image/png"

	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/imageproc"
	"github.com/ollama/ollama/model/input"
)

type Model struct {
	model.Base
	*TextModel

	ImageProcessor

	// TODO: Add VisionModel field
	// *VisionModel `gguf:"v,vision"`

	// TODO: Add MultiModalProjector field for combining vision and text features
	// *MultiModalProjector `gguf:"mm"`
}

// Adding ImageProcessor struct
type ImageProcessor struct {
	imageSize   int
	patchSize   int
	numChannels int
	longestEdge int
}

// Function to create a new ImageProcessor
func newImageProcessor(c ml.Config) ImageProcessor {
	return ImageProcessor{
		imageSize:   int(c.Uint("vision.image_size", 1024)),
		patchSize:   int(c.Uint("vision.patch_size", 16)),
		numChannels: int(c.Uint("vision.num_channels", 3)),
		longestEdge: int(c.Uint("vision.longest_edge", 1024)),
	}
}

// Method to process images for the model
func (p *ImageProcessor) ProcessImage(img image.Image) ([]float32, error) {
	// Get output size based on longest edge and patch size
	outputSize := getResizeOutputImageSize(img, p.longestEdge, image.Point{p.patchSize, p.patchSize})

	// Resize the image
	newImage := imageproc.Composite(img)
	newImage = imageproc.Resize(newImage, outputSize, imageproc.ResizeBilinear)

	// Normalize image data
	data := imageproc.Normalize(newImage, imageproc.ClipDefaultMean, imageproc.ClipDefaultSTD, true, true)

	return data, nil
}

// TODO: Implement MultimodalProcessor interface
// var _ model.MultimodalProcessor = (*Model)(nil)

func New(c ml.Config) (model.Model, error) {
	textModel, err := NewTextModel(c)
	if err != nil {
		return nil, err
	}

	m := &Model{
		TextModel: textModel,
		// Initialize the ImageProcessor
		ImageProcessor: newImageProcessor(c),

		// TODO: Initialize VisionModel if present
		// VisionModel: newVisionModel(c),

		// TODO: Initialize MultiModalProjector
		// MultiModalProjector: &MultiModalProjector{...},
	}

	m.Cache = kvcache.NewCausalCache(m.TextModel.Shift)

	return m, nil
}

// Implement EncodeMultimodal method for processing images
func (m *Model) EncodeMultimodal(ctx ml.Context, multimodalData []byte) (any, error) {
	// Check if vision model exists - return error for now
	return nil, model.ErrNoVisionModel

	// This will be implemented when adding the vision model:
	/*
		image, _, err := image.Decode(bytes.NewReader(multimodalData))
		if err != nil {
			return nil, err
		}

		f32s, err := m.ImageProcessor.ProcessImage(image)
		if err != nil {
			return nil, err
		}

		pixelValues, err := ctx.Input().FromFloatSlice(f32s,
			m.ImageProcessor.imageSize,
			m.ImageProcessor.imageSize,
			m.ImageProcessor.numChannels,
		)
		if err != nil {
			return nil, err
		}

		// Will need VisionModel to process this
		// visionOutputs := m.VisionModel.Forward(ctx, pixelValues)
		// visionOutputs = m.MultiModalProjector.Forward(ctx, visionOutputs)
		// return visionOutputs, nil
	*/
}

func (m *Model) Forward(ctx ml.Context, opts input.Options) (ml.Tensor, error) {
	inputs, err := ctx.Input().FromIntSlice(opts.Inputs, len(opts.Inputs))
	if err != nil {
		return nil, err
	}

	positions, err := ctx.Input().FromIntSlice(opts.Positions, len(opts.Positions))
	if err != nil {
		return nil, err
	}

	outputs, err := ctx.Output().FromIntSlice(opts.Outputs, len(opts.Outputs))
	if err != nil {
		return nil, err
	}

	// TODO: Add handling of multimodal inputs when vision model is added
	// Set image embeddings into hidden state if present in opts.Multimodal

	return m.TextModel.Forward(ctx, inputs, positions, outputs, opts, m.Cache), nil
}

func init() {
	model.Register("mistral3", New)
}
