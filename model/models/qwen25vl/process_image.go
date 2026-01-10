package qwen25vl

import (
	"fmt"
	"image"
	"math"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/model/imageproc"
)

// ImageProcessor contains configuration for the Qwen 2.5 VL image processing
type ImageProcessor struct {
	numChannels       int
	patchSize         int
	temporalPatchSize int
	mergeSize         int
	minPixels         int
	maxPixels         int
	factor            int
	rescaleFactor     float32
	imageMean         [3]float32
	imageStd          [3]float32
}

// newImageProcessor creates a new image processor with default values
func newImageProcessor(c fs.Config) ImageProcessor {
	patchSize := int(c.Uint("vision.patch_size", 14))
	mergeSize := int(c.Uint("vision.spatial_merge_size", 2))

	return ImageProcessor{
		numChannels:       int(c.Uint("vision.num_channels", 3)), // not set
		patchSize:         patchSize,
		temporalPatchSize: 2,
		mergeSize:         mergeSize,
		minPixels:         56 * 56,
		maxPixels:         int(c.Uint("vision.max_pixels", 2<<20)), // 2M limit
		factor:            patchSize * mergeSize,
		rescaleFactor:     1.0 / 255.0,
		imageMean:         imageproc.ClipDefaultMean,
		imageStd:          imageproc.ClipDefaultSTD,
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

	if hBar*wBar > p.maxPixels {
		beta := math.Sqrt(float64(height*width) / float64(p.maxPixels))

		hBar = int(math.Floor(float64(height)/beta/float64(factor))) * factor
		wBar = int(math.Floor(float64(width)/beta/float64(factor))) * factor
	} else if hBar*wBar < p.minPixels {
		beta := math.Sqrt(float64(p.minPixels) / float64(height*width))

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

func (p *ImageProcessor) ProcessImage(img image.Image) ([]float32, *Grid, error) {
	img = imageproc.Composite(img)

	origWidth := img.Bounds().Dx()
	origHeight := img.Bounds().Dy()

	// Calculate smart resize dimensions
	resizedHeight, resizedWidth := p.SmartResize(origHeight, origWidth)

	// Resize image using existing functions
	resizedImg := imageproc.Resize(img, image.Point{X: resizedWidth, Y: resizedHeight}, imageproc.ResizeBilinear)

	normalizedPixels := imageproc.Normalize(resizedImg, p.imageMean, p.imageStd, true, true)

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

	// Return patches and grid dimensions
	return patches, grid, nil
}

func (p *ImageProcessor) createPatches(pixels []float32, height, width int, grid *Grid) ([]float32, error) {
	channels := p.numChannels
	patchSize := p.patchSize
	mergeSize := p.mergeSize
	temporalPatchSize := p.temporalPatchSize

	// Calculate output dimensions
	numPatches := grid.Temporal * grid.Height * grid.Width
	patchDim := channels * temporalPatchSize * patchSize * patchSize

	result := make([]float32, numPatches*patchDim)
	patchIndex := 0

	// Single temporal frame handling (copies to all frames)
	for range grid.Temporal {
		for h := 0; h < grid.Height; h += mergeSize {
			for w := 0; w < grid.Width; w += mergeSize {
				// Handle the 2x2 merged patches
				for mh := range mergeSize {
					for mw := range mergeSize {
						baseOffset := patchIndex * patchDim

						// Extract patch data for first temporal frame
						for c := range channels {
							channelOffset := baseOffset + (c * temporalPatchSize * patchSize * patchSize)

							for py := range patchSize {
								for px := range patchSize {
									// Calculate source pixel coordinates
									y := (h+mh)*patchSize + py
									x := (w+mw)*patchSize + px

									// Source index in input tensor (CHW format)
									srcIdx := c*height*width + y*width + x

									// Destination index in first temporal frame
									dstIdx := channelOffset + (py * patchSize) + px

									if srcIdx < len(pixels) && dstIdx < len(result) {
										result[dstIdx] = pixels[srcIdx]
									}
								}
							}
						}

						// Copy first temporal frame to all other frames
						if temporalPatchSize > 1 {
							for c := range channels {
								channelOffset := baseOffset + (c * temporalPatchSize * patchSize * patchSize)
								firstFrameOffset := channelOffset
								frameSize := patchSize * patchSize

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
