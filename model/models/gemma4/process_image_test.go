package gemma4

import (
	"image"
	"testing"
)

func TestImageProcessorSmartResizeRespectsMaxPixels(t *testing.T) {
	p := ImageProcessor{
		patchSize:   16,
		nMerge:      3,
		numChannels: 3,
		patchArea:   16 * 16 * 3 * 3,
	}
	align := p.patchSize * p.nMerge
	// Large original; cap with a small max pixel budget
	maxTok := 70
	minTok := 70
	minPx := minTok * p.patchArea
	maxPx := maxTok * p.patchArea
	w, h := p.smartResize(4096, 4096, align, minPx, maxPx)
	if w*h > maxPx {
		t.Fatalf("output pixels %d*%d=%d > max %d", w, h, w*h, maxPx)
	}
	if w%align != 0 || h%align != 0 {
		t.Fatalf("dimensions not aligned: %dx%d align %d", w, h, align)
	}
}

func TestProcessImageWithBudgetsTinyImage(t *testing.T) {
	p := ImageProcessor{
		patchSize:   16,
		nMerge:      3,
		numChannels: 3,
		patchArea:   16 * 16 * 3 * 3,
	}
	img := image.NewRGBA(image.Rect(0, 0, 48, 48))
	_, w, h, err := p.ProcessImageWithBudgets(img, 70, 560)
	if err != nil {
		t.Fatal(err)
	}
	if w < p.patchSize*p.nMerge || h < p.patchSize*p.nMerge {
		t.Fatalf("output too small: %dx%d", w, h)
	}
}
