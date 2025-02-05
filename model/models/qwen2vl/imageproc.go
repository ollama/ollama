package qwen2vl

import (
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"math"

	"github.com/ollama/ollama/model/imageproc"
)

const (
	DefaultFactor    = 28
	DefaultMinPixels = 56 * 56
	DefaultMaxPixels = 14 * 14 * 4 * 1280
)

// smartResize calculates the size of the image to resize to based on the
// factor, minPixels, and maxPixels.
func smartResize(size image.Point, factor, minPixels, maxPixels int) image.Point {
	// 1. Both dimensions of size are divisible by factor
	// 2. The area of the image is between minPixels and maxPixels
	// 3. The aspect ratio of the image is as close to 1:1 as possible

	if size.Y < factor || size.X < factor {
		panic("image is too small to resize")
	} else if max(size.X, size.Y)/min(size.X, size.Y) > 200 {
		panic("aspect ratio must be less than 200:1")
	}

	f := float64(factor)
	width := float64(size.X)
	height := float64(size.Y)

	xBar := math.Round(width/f) * f
	yBar := math.Round(height/f) * f

	if xBar*yBar > float64(maxPixels) {
		beta := math.Sqrt(height * width / float64(maxPixels))
		xBar = math.Floor(width/beta/f) * f
		yBar = math.Floor(height/beta/f) * f
	} else if xBar*yBar < float64(minPixels) {
		beta := math.Sqrt(float64(minPixels) / (height * width))
		xBar = math.Ceil(width*beta/f) * f
		yBar = math.Ceil(height*beta/f) * f
	}

	return image.Point{int(xBar), int(yBar)}
}

func resizeImage(img image.Image, format string, size image.Point) image.Image {
	if format == "png" {
		img = imageproc.Composite(img)
	}

	return imageproc.Resize(img, size, imageproc.ResizeBilinear)
}

func Preprocess(imageData io.Reader) ([]float32, map[string]any, error) {
	img, format, err := image.Decode(imageData)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to decode image: %w", err)
	}

	size := smartResize(img.Bounds().Max, DefaultFactor, DefaultMinPixels, DefaultMaxPixels)
	img = resizeImage(img, format, size)

	data := imageproc.Normalize(img, imageproc.ClipDefaultMean, imageproc.ClipDefaultSTD, true, true)

	opts := map[string]any{}
	return data, opts, nil
}
