package gemma4

import (
	"image"
	"log/slog"
	"math"

	"golang.org/x/image/draw"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/internal/gemma4vision"
)

type ImageProcessor struct {
	patchSize   int
	numChannels int
	nMerge      int
	patchArea   int
}

func newImageProcessor(c fs.Config) ImageProcessor {
	patchSize := int(c.Uint("vision.patch_size", 16))
	nMerge := int(c.Uint("vision.projector.scale_factor", 3))
	numChannels := int(c.Uint("vision.num_channels", 3))
	patchArea := patchSize * patchSize * nMerge * nMerge

	return ImageProcessor{
		patchSize:   patchSize,
		numChannels: numChannels,
		nMerge:      nMerge,
		patchArea:   patchArea,
	}
}

// ProcessImage resizes an image preserving aspect ratio, aligning dimensions
// to (patchSize * nMerge) boundaries, and normalizes pixels to [-1, 1].
// Uses default Gemma 4 visual token budgets (70 / 560).
func (p *ImageProcessor) ProcessImage(img image.Image) ([]float32, int, int, error) {
	minTok, maxTok := gemma4vision.NormalizeGemma4ImageBudgets(0, 0)
	return p.ProcessImageWithBudgets(img, minTok, maxTok)
}

// ProcessImageWithBudgets applies minTokens/maxTokens as visual token budgets
// (see gemma4vision) converted through patchArea to pixel caps for resize.
func (p *ImageProcessor) ProcessImageWithBudgets(img image.Image, minTokens, maxTokens int) ([]float32, int, int, error) {
	minPixels := minTokens * p.patchArea
	maxPixels := maxTokens * p.patchArea
	return p.processImage(img, minPixels, maxPixels)
}

func (p *ImageProcessor) processImage(img image.Image, minPixels, maxPixels int) ([]float32, int, int, error) {
	alignSize := p.patchSize * p.nMerge
	targetW, targetH := p.smartResize(img.Bounds().Dx(), img.Bounds().Dy(), alignSize, minPixels, maxPixels)

	dst := image.NewRGBA(image.Rect(0, 0, targetW, targetH))
	draw.BiLinear.Scale(dst, dst.Bounds(), img, img.Bounds(), draw.Over, nil)

	data := p.pack(dst)
	return data, targetW, targetH, nil
}

// smartResize picks output dimensions aligned to alignSize, scaled to respect
// maxPixels, then grown if below minPixels when possible without exceeding maxPixels.
func (p *ImageProcessor) smartResize(origW, origH, alignSize, minPixels, maxPixels int) (int, int) {
	totalPx := origW * origH

	var targetW, targetH int
	if maxPixels > 0 && totalPx > 0 {
		factor := math.Sqrt(float64(maxPixels) / float64(totalPx))
		targetH = max(alignSize, int(math.Floor(factor*float64(origH)/float64(alignSize)))*alignSize)
		targetW = max(alignSize, int(math.Floor(factor*float64(origW)/float64(alignSize)))*alignSize)
	} else {
		targetH = max(alignSize, (origH/alignSize)*alignSize)
		targetW = max(alignSize, (origW/alignSize)*alignSize)
	}

	if minPixels > 0 && targetW*targetH < minPixels {
		growth := math.Sqrt(float64(minPixels) / float64(targetW*targetH))
		tw := max(alignSize, int(math.Ceil(growth*float64(targetW)/float64(alignSize)))*alignSize)
		th := max(alignSize, int(math.Ceil(growth*float64(targetH)/float64(alignSize)))*alignSize)
		if tw*th <= maxPixels {
			return tw, th
		}
		slog.Warn("gemma4 vision: min token budget could not be satisfied within max budget; using max-budget resize only",
			"min_pixels", minPixels, "max_pixels", maxPixels, "target_w", targetW, "target_h", targetH)
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
