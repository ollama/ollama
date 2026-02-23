package lfm2

import (
	"image"
	"image/color"
	"testing"
)

func TestProcessImageSingleTile(t *testing.T) {
	p := ImageProcessor{
		patchSize:          16,
		downsampleFactor:   2,
		numChannels:        3,
		imageMean:          [3]float32{0.5, 0.5, 0.5},
		imageStd:           [3]float32{0.5, 0.5, 0.5},
		doImageSplitting:   true,
		minTiles:           2,
		maxTiles:           10,
		useThumbnail:       true,
		tileSize:           512,
		minImageTokens:     64,
		maxImageTokens:     256,
		maxPixelsTolerance: 2.0,
	}

	img := image.NewRGBA(image.Rect(0, 0, 320, 320))
	out, layout, err := p.ProcessImage(img)
	if err != nil {
		t.Fatalf("ProcessImage returned error: %v", err)
	}

	if layout.rows != 1 || layout.cols != 1 || layout.hasThumbnail {
		t.Fatalf("layout = %+v, want rows=1 cols=1 hasThumbnail=false", layout)
	}
	if len(out) != 1 {
		t.Fatalf("len(out) = %d, want 1", len(out))
	}
	if out[0].size != (image.Point{X: 320, Y: 320}) {
		t.Fatalf("single image size = %+v, want 320x320", out[0].size)
	}
	if out[0].thumbnail {
		t.Fatalf("single image should not be marked as thumbnail")
	}
}

func TestProcessImageDynamicTiling(t *testing.T) {
	p := ImageProcessor{
		patchSize:          16,
		downsampleFactor:   2,
		numChannels:        3,
		imageMean:          [3]float32{0.5, 0.5, 0.5},
		imageStd:           [3]float32{0.5, 0.5, 0.5},
		doImageSplitting:   true,
		minTiles:           2,
		maxTiles:           10,
		useThumbnail:       true,
		tileSize:           512,
		minImageTokens:     64,
		maxImageTokens:     256,
		maxPixelsTolerance: 2.0,
	}

	// Wide image that should trigger multi-tile splitting.
	img := image.NewRGBA(image.Rect(0, 0, 3000, 1000))
	fill := color.RGBA{R: 120, G: 90, B: 60, A: 255}
	for y := range 1000 {
		for x := range 3000 {
			img.Set(x, y, fill)
		}
	}

	out, layout, err := p.ProcessImage(img)
	if err != nil {
		t.Fatalf("ProcessImage returned error: %v", err)
	}

	if layout.rows*layout.cols <= 1 {
		t.Fatalf("expected multi-tile layout, got %+v", layout)
	}
	if !layout.hasThumbnail {
		t.Fatalf("expected thumbnail for multi-tile layout")
	}

	wantLen := layout.rows*layout.cols + 1
	if len(out) != wantLen {
		t.Fatalf("len(out) = %d, want %d", len(out), wantLen)
	}

	for i := range layout.rows * layout.cols {
		if out[i].size != (image.Point{X: 512, Y: 512}) {
			t.Fatalf("tile[%d] size = %+v, want 512x512", i, out[i].size)
		}
		if out[i].thumbnail {
			t.Fatalf("tile[%d] should not be marked as thumbnail", i)
		}
	}

	thumb := out[len(out)-1]
	if !thumb.thumbnail {
		t.Fatalf("last chunk should be thumbnail")
	}
	if thumb.size.X%32 != 0 || thumb.size.Y%32 != 0 {
		t.Fatalf("thumbnail size = %+v, want dimensions aligned to 32", thumb.size)
	}
}
