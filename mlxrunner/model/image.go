package model

import (
	"fmt"
	"image"
	"math"

	"golang.org/x/image/draw"
)

// FitImagePixels rounds image sides to multiples of factor and resizes to the
// given pixel budget while preserving the aspect ratio.
func FitImagePixels(height, width, factor, minPixels, maxPixels int32) (int32, int32, error) {
	if height <= 0 || width <= 0 {
		return 0, 0, fmt.Errorf("invalid image size %dx%d", width, height)
	}
	if factor <= 0 || minPixels <= 0 || maxPixels <= 0 {
		return 0, 0, fmt.Errorf("invalid image resize bounds or patch factor")
	}
	longer, shorter := float64(max(height, width)), float64(min(height, width))
	if longer/shorter > 200 {
		return 0, 0, fmt.Errorf("image aspect ratio %.0f exceeds 200", longer/shorter)
	}

	f := float64(factor)
	// The reference uses Python round (half to even).
	hBar := math.RoundToEven(float64(height)/f) * f
	wBar := math.RoundToEven(float64(width)/f) * f
	if hBar*wBar > float64(maxPixels) {
		beta := math.Sqrt(float64(height) * float64(width) / float64(maxPixels))
		hBar = math.Max(f, math.Floor(float64(height)/beta/f)*f)
		wBar = math.Max(f, math.Floor(float64(width)/beta/f)*f)
	} else if hBar*wBar < float64(minPixels) {
		beta := math.Sqrt(float64(minPixels) / (float64(height) * float64(width)))
		hBar = math.Ceil(float64(height)*beta/f) * f
		wBar = math.Ceil(float64(width)*beta/f) * f
	}
	return int32(hBar), int32(wBar), nil
}

// DropImageAlpha flattens a non-opaque image to straight RGB: the reference
// drops alpha via RGB conversion before resizing, not by compositing.
func DropImageAlpha(img image.Image, bounds image.Rectangle) image.Image {
	if o, ok := img.(interface{ Opaque() bool }); ok && o.Opaque() {
		return img
	}
	flat := image.NewNRGBA(bounds)
	draw.Draw(flat, bounds, img, bounds.Min, draw.Src)
	for i := 3; i < len(flat.Pix); i += 4 {
		flat.Pix[i] = 0xff
	}
	return flat
}
