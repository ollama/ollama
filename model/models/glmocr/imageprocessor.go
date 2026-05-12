package glmocr

import (
	"image"
	"log/slog"
	"math"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/model/imageproc"
)

type ImageProcessor struct {
	imageSize         int
	patchSize         int
	temporalPatchSize int
	spatialMergeSize  int
	minPixels         int
	maxPixels         int
	factor            int
	imageMean         [3]float32
	imageStd          [3]float32
}

func newImageProcessor(c fs.Config) ImageProcessor {
	patchSize := int(c.Uint("vision.patch_size", 14))
	spatialMergeSize := int(c.Uint("vision.spatial_merge_size", 2))
	temporalPatchSize := int(c.Uint("vision.temporal_patch_size", 2))

	// Read normalization values from config if available, otherwise use CLIP defaults
	imageMean := c.Floats("vision.image_mean", imageproc.ClipDefaultMean[:])
	imageStd := c.Floats("vision.image_std", imageproc.ClipDefaultSTD[:])

	// Default max_pixels: 2048 * patchSize^2 * mergeSize^2 * temporal = ~3.2M pixels
	// This limits to ~16k patches (4k output tokens) to keep memory stable without flash attention
	defaultMaxPixels := 2048 * patchSize * patchSize * spatialMergeSize * spatialMergeSize * temporalPatchSize

	return ImageProcessor{
		imageSize:         int(c.Uint("vision.image_size", 336)),
		patchSize:         patchSize,
		temporalPatchSize: temporalPatchSize,
		spatialMergeSize:  spatialMergeSize,
		minPixels:         int(c.Uint("vision.min_pixels", uint32(8*patchSize*patchSize*spatialMergeSize*spatialMergeSize*temporalPatchSize))),
		maxPixels:         int(c.Uint("vision.max_pixels", uint32(defaultMaxPixels))),
		factor:            patchSize * spatialMergeSize,
		imageMean:         [3]float32{imageMean[0], imageMean[1], imageMean[2]},
		imageStd:          [3]float32{imageStd[0], imageStd[1], imageStd[2]},
	}
}

func (p *ImageProcessor) SmartResize(height, width int) (int, int) {
	factor := p.factor
	temporalFactor := p.temporalPatchSize
	numFrames := temporalFactor // single image

	if height < factor || width < factor {
		// Scale up small images
		scale := float64(factor) / float64(min(height, width))
		height = int(math.Ceil(float64(height) * scale))
		width = int(math.Ceil(float64(width) * scale))
	}

	if temporalFactor <= 0 {
		slog.Warn("temporal_patch_size must be > 0, defaulting to 1")
		temporalFactor = 1
	}
	if numFrames < temporalFactor {
		slog.Warn("num_frames must be >= temporal_patch_size, adjusting num_frames", "num_frames", numFrames, "temporal_patch_size", temporalFactor)
		numFrames = temporalFactor
	}
	if aspectRatio := float64(max(height, width)) / float64(min(height, width)); aspectRatio > 200 {
		slog.Warn("aspect ratio exceeds 200, image quality may be affected", "aspect_ratio", aspectRatio)
	}

	round := func(x float64) int { return int(math.RoundToEven(x)) }

	hBar := round(float64(height)/float64(factor)) * factor
	wBar := round(float64(width)/float64(factor)) * factor
	tBar := round(float64(numFrames)/float64(temporalFactor)) * temporalFactor

	if tBar*hBar*wBar > p.maxPixels {
		beta := math.Sqrt(float64(numFrames*height*width) / float64(p.maxPixels))
		hBar = int(math.Floor(float64(height)/beta/float64(factor))) * factor
		wBar = int(math.Floor(float64(width)/beta/float64(factor))) * factor
	} else if tBar*hBar*wBar < p.minPixels {
		beta := math.Sqrt(float64(p.minPixels) / float64(numFrames*height*width))
		hBar = int(math.Ceil(float64(height)*beta/float64(factor))) * factor
		wBar = int(math.Ceil(float64(width)*beta/float64(factor))) * factor
	}

	return hBar, wBar
}

func (p *ImageProcessor) ProcessImage(img image.Image) ([]float32, *Grid, error) {
	img = imageproc.Composite(img)

	origWidth := img.Bounds().Dx()
	origHeight := img.Bounds().Dy()

	// Calculate smart resize dimensions
	resizedHeight, resizedWidth := p.SmartResize(origHeight, origWidth)

	// Resize image
	resizedImg := imageproc.Resize(img, image.Point{X: resizedWidth, Y: resizedHeight}, imageproc.ResizeCatmullrom)

	// Normalize pixels - output format is [C, H, W] with rescale and channelFirst
	// We keep [C, H, W] for patch extraction
	normalizedPixels := imageproc.Normalize(resizedImg, p.imageMean, p.imageStd, true, true)

	// Calculate grid dimensions (after Conv2D patching)
	grid := &Grid{
		Height:      resizedHeight / p.patchSize,
		Width:       resizedWidth / p.patchSize,
		Temporal:    1, // Single image
		ImageHeight: resizedHeight,
		ImageWidth:  resizedWidth,
	}

	patches, err := p.createPatches(normalizedPixels, resizedHeight, resizedWidth, grid)
	if err != nil {
		return nil, nil, err
	}

	return patches, grid, nil
}

func (p *ImageProcessor) createPatches(pixels []float32, height, width int, grid *Grid) ([]float32, error) {
	channels := 3
	patchSize := p.patchSize
	mergeSize := p.spatialMergeSize
	temporalPatchSize := p.temporalPatchSize

	numPatches := grid.Temporal * grid.Height * grid.Width
	patchDim := channels * temporalPatchSize * patchSize * patchSize
	result := make([]float32, numPatches*patchDim)
	patchIndex := 0

	// Single temporal frame handling (copies to all frames)
	for range grid.Temporal {
		for h := 0; h < grid.Height; h += mergeSize {
			for w := 0; w < grid.Width; w += mergeSize {
				for mh := range mergeSize {
					for mw := range mergeSize {
						baseOffset := patchIndex * patchDim
						for c := range channels {
							channelOffset := baseOffset + (c * temporalPatchSize * patchSize * patchSize)
							for py := range patchSize {
								for px := range patchSize {
									y := (h+mh)*patchSize + py
									x := (w+mw)*patchSize + px
									srcIdx := c*height*width + y*width + x
									dstIdx := channelOffset + (py * patchSize) + px
									result[dstIdx] = pixels[srcIdx]
								}
							}

							if temporalPatchSize > 1 {
								frameSize := patchSize * patchSize
								for tp := 1; tp < temporalPatchSize; tp++ {
									currentFrameOffset := channelOffset + (tp * frameSize)
									copy(result[currentFrameOffset:currentFrameOffset+frameSize],
										result[channelOffset:channelOffset+frameSize])
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
