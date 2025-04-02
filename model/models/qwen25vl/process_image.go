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
	imageSize         int
	numChannels       int
	patchSize         int
	temporalPatchSize int
	mergeSize         int
	minPixels         int
	maxPixels         int
	factor            int
	rescaleFactor     float32
	imageMean         []float32
	imageStd          []float32
}

// newImageProcessor creates a new image processor with default values
func newImageProcessor(c fs.Config) ImageProcessor {

	patchSize := int(c.Uint("vision.patch_size", 14))
	mergeSize := int(c.Uint("vision.spatial_merge_size", 2))

	return ImageProcessor{
		imageSize:         int(c.Uint("vision.image_size", 560)),
		numChannels:       3,
		patchSize:         patchSize,
		temporalPatchSize: 2,
		mergeSize:         mergeSize,
		minPixels:         56 * 56,
		maxPixels:         28 * 28 * 4 * 1280,
		factor:            patchSize * mergeSize,
		rescaleFactor:     1.0 / 255.0,
		imageMean:         []float32{0.48145466, 0.4578275, 0.40821073},
		imageStd:          []float32{0.26862954, 0.26130258, 0.27577711},
	}
}

// SmartResize implements the smart resize algorithm
func (p *ImageProcessor) SmartResize(height, width int) (int, int) {
	factor := p.factor

	if height < factor || width < factor {
		panic(fmt.Sprintf("height:%d or width:%d must be larger than factor:%d", height, width, factor))
	} else if float64(max(height, width))/float64(min(height, width)) > 200 {
		aspectRatio := float64(max(height, width)) / float64(min(height, width))
		panic(fmt.Sprintf("absolute aspect ratio must be smaller than 200, got %f", aspectRatio))
	}

	round := func(x float64) int {
		return int(math.Round(x))
	}
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

func (p *ImageProcessor) ProcessImage(img image.Image) ([]float32, int, int, int, error) {
	origWidth := img.Bounds().Dx()
	origHeight := img.Bounds().Dy()

	// Calculate smart resize dimensions
	resizedHeight, resizedWidth := p.SmartResize(origHeight, origWidth)

	// Resize image using existing functions
	resizedImg := imageproc.Resize(img, image.Point{X: resizedWidth, Y: resizedHeight}, imageproc.ResizeBilinear)

	normalizedPixels := imageproc.Normalize(
		resizedImg,
		[3]float32{p.imageMean[0], p.imageMean[1], p.imageMean[2]},
		[3]float32{p.imageStd[0], p.imageStd[1], p.imageStd[2]},
		true, // rescale
		true, // channelFirst
	)

	// Calculate grid dimensions
	gridH := resizedHeight / p.patchSize
	gridW := resizedWidth / p.patchSize
	gridT := 1 // For single images, temporal dimension is 1

	patches, err := p.createPatches(normalizedPixels, resizedHeight, resizedWidth, gridH, gridW, gridT)
	if err != nil {
		return nil, 0, 0, 0, fmt.Errorf("failed to create patches: %v", err)
	}

	// Return patches and grid dimensions
	return patches, gridT, gridH, gridW, nil
}

func (p *ImageProcessor) createPatches(pixels []float32, height, width, gridH, gridW, gridT int) ([]float32, error) {
	channels := p.numChannels
	patchSize := p.patchSize
	mergeSize := p.mergeSize
	temporalPatchSize := p.temporalPatchSize

	// Calculate output dimensions
	numPatches := gridT * gridH * gridW
	patchDim := channels * temporalPatchSize * patchSize * patchSize

	// Create output tensor
	result := make([]float32, numPatches*patchDim)

	// Instead of the complex 9D reshape+transpose, directly extract patches
	// in the format expected by the forward pass
	patchIndex := 0

	for t := 0; t < gridT; t++ {
		// For each patch in the grid
		for h := 0; h < gridH; h += mergeSize {
			for w := 0; w < gridW; w += mergeSize {
				// Handle the 2x2 merged patches
				for mh := 0; mh < mergeSize; mh++ {
					for mw := 0; mw < mergeSize; mw++ {
						// For each pixel in the patch
						for py := 0; py < patchSize; py++ {
							for px := 0; px < patchSize; px++ {
								// Calculate source coordinates
								y := (h+mh)*patchSize + py
								x := (w+mw)*patchSize + px

								// For each channel
								for c := 0; c < channels; c++ {
									// Channel-first format (CHW)
									srcIdx := c*height*width + y*width + x

									// Calculate destination index based on the expected layout
									// This is the key part that matches what the model expects
									dstIdx := patchIndex*patchDim +
										(c * temporalPatchSize * patchSize * patchSize) +
										(0 * patchSize * patchSize) + // temporal dim
										(py * patchSize) +
										px

									if srcIdx < len(pixels) && dstIdx < len(result) {
										result[dstIdx] = pixels[srcIdx]
									}
								}
							}
						}

						// Handle temporal dimension padding (if needed)
						for tp := 1; tp < temporalPatchSize; tp++ {
							for py := 0; py < patchSize; py++ {
								for px := 0; px < patchSize; px++ {
									for c := 0; c < channels; c++ {
										srcIdx := patchIndex*patchDim +
											(c * temporalPatchSize * patchSize * patchSize) +
											(0 * patchSize * patchSize) + // first temporal frame
											(py * patchSize) +
											px

										dstIdx := patchIndex*patchDim +
											(c * temporalPatchSize * patchSize * patchSize) +
											(tp * patchSize * patchSize) + // current temporal frame
											(py * patchSize) +
											px

										if srcIdx < len(result) && dstIdx < len(result) {
											result[dstIdx] = result[srcIdx] // Copy from first frame
										}
									}
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
