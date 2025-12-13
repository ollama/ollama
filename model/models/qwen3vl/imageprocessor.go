package qwen3vl

import (
	"fmt"
	"image"
	"log/slog"
	"math"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/imageproc"
)

// ImageProcessor contains configuration for the Qwen 3 VL image processing
type ImageProcessor struct {
	numChannels       int
	patchSize         int
	temporalPatchSize int
	storagePatchSize  int
	mergeSize         int
	shortestEdge      int
	longestEdge       int
	factor            int
	rescaleFactor     float32
	imageMean         []float32
	imageStd          []float32
}

// newImageProcessor creates a new image processor with default values
func newImageProcessor(c fs.Config) ImageProcessor {
	patchSize := int(c.Uint("vision.patch_size", 14))
	mergeSize := int(c.Uint("vision.spatial_merge_size", 2))
	// Read temporalPatchSize from GGUF: split models have 1, non-split have 2
	temporalPatchSize := int(c.Uint("vision.temporal_patch_size", 2))

	return ImageProcessor{
		numChannels:       int(c.Uint("vision.num_channels", 3)), // not set
		patchSize:         patchSize,
		temporalPatchSize: temporalPatchSize,
		mergeSize:         mergeSize,
		shortestEdge:      int(c.Uint("vision.shortest_edge", 64<<10)),
		// FIXME(mxyng): the model defined longest edge (16M) is too large for the default
		// context length of 8K and will panic. Adjusting to 2M for now.
		// longestEdge:   int(c.Uint("vision.longest_edge", 16<<20)),
		longestEdge:   2 << 20,
		factor:        patchSize * mergeSize,
		rescaleFactor: 1.0 / 255.0,
		// Qwen-VL family typically uses CLIP normalization; split models may omit these keys.
		imageMean: c.Floats("vision.image_mean", imageproc.ClipDefaultMean[:]),
		imageStd:  c.Floats("vision.image_std", imageproc.ClipDefaultSTD[:]),
	}
}

// SmartResize implements the smart resize algorithm
func (p *ImageProcessor) SmartResize(height, width int) (int, int) {
	factor := p.factor

	if height < factor || width < factor {
		panic(fmt.Sprintf("height:%d or width:%d must be larger than factor:%d", height, width, factor))
	} else if aspectRatio := max(height, width) / min(height, width); aspectRatio > 200 {
		panic(fmt.Sprintf("absolute aspect ratio must be smaller than 200, got %v", aspectRatio))
	}

	round := func(x float64) int { return int(math.RoundToEven(x)) }

	hBar := round(float64(height)/float64(factor)) * factor
	wBar := round(float64(width)/float64(factor)) * factor

	if hBar*wBar > p.longestEdge {
		beta := math.Sqrt(float64(height*width) / float64(p.longestEdge))

		hBar = int(math.Floor(float64(height)/beta/float64(factor))) * factor
		wBar = int(math.Floor(float64(width)/beta/float64(factor))) * factor
	} else if hBar*wBar < p.shortestEdge {
		beta := math.Sqrt(float64(p.shortestEdge) / float64(height*width))

		hBar = int(math.Ceil(float64(height)*beta/float64(factor))) * factor
		wBar = int(math.Ceil(float64(width)*beta/float64(factor))) * factor
	}

	return hBar, wBar
}

type Grid struct {
	Height   int
	Width    int
	Temporal int
}

func (p *ImageProcessor) ProcessImage(ctx ml.Context, img image.Image) (ml.Tensor, *Grid, error) {
	img = imageproc.Composite(img)

	origWidth := img.Bounds().Dx()
	origHeight := img.Bounds().Dy()

	// Calculate smart resize dimensions
	resizedHeight, resizedWidth := p.SmartResize(origHeight, origWidth)

	// Keep resize behavior stable across runs/models.
	resizedImg := imageproc.Resize(img, image.Point{X: resizedWidth, Y: resizedHeight}, imageproc.ResizeBilinear)

	normalizedPixels := imageproc.Normalize(
		resizedImg,
		[3]float32{p.imageMean[0], p.imageMean[1], p.imageMean[2]},
		[3]float32{p.imageStd[0], p.imageStd[1], p.imageStd[2]},
		true, // rescale
		true, // channelFirst
	)

	// Calculate grid dimensions
	grid := &Grid{
		Height:   resizedHeight / p.patchSize,
		Width:    resizedWidth / p.patchSize,
		Temporal: 1, // For single images, temporal dimension is 1
	}

	patches, err := p.createPatches(normalizedPixels, resizedHeight, resizedWidth, grid)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create patches: %v", err)
	}

	patchDim := p.numChannels * p.temporalPatchSize *
		p.patchSize * p.patchSize
	numPatches := grid.Temporal * grid.Height * grid.Width

	pixelValues := ctx.Input().FromFloats(patches, patchDim, numPatches)

	slog.Debug("ImageProcessor.ProcessImage",
		"patch_dim", patchDim,
		"num_patches", numPatches,
		"grid", []int{grid.Height, grid.Width, grid.Temporal},
		"patch_size", p.patchSize,
		"temporal_patch_size", p.temporalPatchSize)

	// Return patches and grid dimensions
	return pixelValues, grid, nil
}

// ProcessImageRaw returns the raw normalized CHW image for use with Conv2D (split models)
// Returns tensor with shape [width, height, channels] and grid dimensions
func (p *ImageProcessor) ProcessImageRaw(ctx ml.Context, img image.Image) (ml.Tensor, *Grid, error) {
	img = imageproc.Composite(img)

	origWidth := img.Bounds().Dx()
	origHeight := img.Bounds().Dy()

	// Calculate smart resize dimensions
	resizedHeight, resizedWidth := p.SmartResize(origHeight, origWidth)

	// Keep resize behavior stable across runs/models.
	resizedImg := imageproc.Resize(img, image.Point{X: resizedWidth, Y: resizedHeight}, imageproc.ResizeBilinear)

	// Normalize to HWC so Conv2D sees width/height/channel ordering expected by ggml_conv_2d
	normalizedPixels := imageproc.Normalize(
		resizedImg,
		[3]float32{p.imageMean[0], p.imageMean[1], p.imageMean[2]},
		[3]float32{p.imageStd[0], p.imageStd[1], p.imageStd[2]},
		true,  // rescale
		false, // channelFirst -> HWC format for WHC tensor layout
	)

	// Calculate grid dimensions (patches after Conv2D)
	grid := &Grid{
		Height:   resizedHeight / p.patchSize,
		Width:    resizedWidth / p.patchSize,
		Temporal: 1,
	}

	// Create tensor with shape [height, width, channels] (row-major HWC) for ggml_conv_2d
	// ggml_conv_2d expects input as [W, H, C, batch]; we keep row-major HWC and the reshape in the
	// vision model will place width/height correctly.
	pixelValues := ctx.Input().FromFloats(normalizedPixels, resizedHeight, resizedWidth, p.numChannels)

	return pixelValues, grid, nil
}

func (p *ImageProcessor) createPatches(pixels []float32, height, width int, grid *Grid) ([]float32, error) {
	channels := p.numChannels
	patchSize := p.patchSize
	mergeSize := p.mergeSize
	temporalPatchSize := p.temporalPatchSize

	storageSize := p.storagePatchSize
	if storageSize == 0 {
		storageSize = patchSize
	}

	// Calculate output dimensions
	numPatches := grid.Temporal * grid.Height * grid.Width
	patchDim := channels * temporalPatchSize * storageSize * storageSize

	result := make([]float32, numPatches*patchDim)
	patchIndex := 0

	// Iterate over grid locations
	for range grid.Temporal {
		for h := 0; h < grid.Height; h += mergeSize {
			for w := 0; w < grid.Width; w += mergeSize {
				// Handle the 2x2 merged patches
				for mh := range mergeSize {
					for mw := range mergeSize {
						baseOffset := patchIndex * patchDim
						// Extract patch data for first temporal frame
						for c := range channels {
							channelOffset := baseOffset + (c * temporalPatchSize * storageSize * storageSize)

							for py := range patchSize {
								for px := range patchSize {
									// Calculate source pixel coordinates
									y := (h+mh)*patchSize + py
									x := (w+mw)*patchSize + px

									// Source index in input tensor (CHW format)
									srcIdx := c*height*width + y*width + x

									// Destination index in first temporal frame (using storageSize for stride)
									dstIdx := channelOffset + (py * storageSize) + px

									if srcIdx < len(pixels) && dstIdx < len(result) {
										result[dstIdx] = pixels[srcIdx]
									}
								}
							}
						}

						// Copy first temporal frame to all other frames
						if temporalPatchSize > 1 {
							for c := range channels {
								channelOffset := baseOffset + (c * temporalPatchSize * storageSize * storageSize)
								firstFrameOffset := channelOffset
								frameSize := storageSize * storageSize

								// Copy first frame to all other frames
								for tp := 1; tp < temporalPatchSize; tp++ {
									currentFrameOffset := channelOffset + (tp * frameSize)
									copy(result[currentFrameOffset:currentFrameOffset+frameSize],
										result[firstFrameOffset:firstFrameOffset+frameSize])
								}
							}
						}
						patchIndex++
					}
				}
			}
		}
	}

	return result, nil
}
