package gemma4

import (
	"image"
	"math"

	"golang.org/x/image/draw"

	"github.com/ollama/ollama/fs"
)

type ImageProcessor struct {
	patchSize   int
	numChannels int
	nMerge      int
	minPixels   int
	maxPixels   int
}

func newImageProcessor(c fs.Config) ImageProcessor {
	patchSize := int(c.Uint("vision.patch_size", 16))
	nMerge := int(c.Uint("vision.projector.scale_factor", 3))
	numChannels := int(c.Uint("vision.num_channels", 3))

	// Token limits from reference: min=40, max=280 output tokens after pooling.
	// Convert to pixel counts: tokens * nMerge^2 * patchSize^2
	minTokens := 40
	maxTokens := 280
	patchArea := patchSize * patchSize * nMerge * nMerge
	minPixels := minTokens * patchArea
	maxPixels := maxTokens * patchArea

	return ImageProcessor{
		patchSize:   patchSize,
		numChannels: numChannels,
		nMerge:      nMerge,
		minPixels:   minPixels,
		maxPixels:   maxPixels,
	}
}

// ProcessImage resizes an image preserving aspect ratio, aligning dimensions
// to (patchSize * nMerge) boundaries, and normalizes pixels to [-1, 1].
// Returns the float32 pixel data and the actual output dimensions.
func (p *ImageProcessor) ProcessImage(img image.Image) ([]float32, int, int, error) {
	// Compute target size preserving aspect ratio
	alignSize := p.patchSize * p.nMerge
	targetW, targetH := p.smartResize(img.Bounds().Dx(), img.Bounds().Dy(), alignSize)

	// Resize directly without alpha compositing, matching MLX reference.
	dst := image.NewRGBA(image.Rect(0, 0, targetW, targetH))
	draw.BiLinear.Scale(dst, dst.Bounds(), img, img.Bounds(), draw.Over, nil)

	// Normalize to [-1, 1] using mean=0.5, std=0.5: (pixel/255 - 0.5) / 0.5 = 2*pixel/255 - 1
	data := p.pack(dst)
	return data, targetW, targetH, nil
}

// smartResize computes target dimensions that preserve aspect ratio and
// align to alignSize boundaries. It scales the image to fill the maximum
// patch budget (maxPixels), matching the MLX reference.
func (p *ImageProcessor) smartResize(origW, origH, alignSize int) (int, int) {
	totalPx := origW * origH

	var targetW, targetH int
	if p.maxPixels > 0 && totalPx > 0 {
		factor := math.Sqrt(float64(p.maxPixels) / float64(totalPx))
		targetH = max(alignSize, int(math.Floor(factor*float64(origH)/float64(alignSize)))*alignSize)
		targetW = max(alignSize, int(math.Floor(factor*float64(origW)/float64(alignSize)))*alignSize)
	} else {
		targetH = max(alignSize, (origH/alignSize)*alignSize)
		targetW = max(alignSize, (origW/alignSize)*alignSize)
	}

	return targetW, targetH
}

// pack extracts RGB values from an image and normalizes to [-1, 1].
// Returns channel-first layout: [R..., G..., B...].
func (p *ImageProcessor) pack(img image.Image) []float32 {
	bounds := img.Bounds()
	w := bounds.Dx()
	h := bounds.Dy()
	size := w * h

	pixelVals := make([]float32, 3*size)
	rOff, gOff, bOff := 0, size, 2*size

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			c := img.At(x, y)
			r, g, b, _ := c.RGBA()
			idx := (y-bounds.Min.Y)*w + (x - bounds.Min.X)

			// Normalize [0, 255] -> [-1, 1]: 2 * (val/255) - 1
			pixelVals[rOff+idx] = float32(r>>8)/255.0*2.0 - 1.0
			pixelVals[gOff+idx] = float32(g>>8)/255.0*2.0 - 1.0
			pixelVals[bOff+idx] = float32(b>>8)/255.0*2.0 - 1.0
		}
	}

	return pixelVals
}
