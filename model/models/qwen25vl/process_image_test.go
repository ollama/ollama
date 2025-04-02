package qwen25vl

import (
	"image"
	_ "image/jpeg" // Register JPEG decoder
	"testing"
)

func TestSmartResize(t *testing.T) {
	type smartResizeCase struct {
		TestImage image.Image
		Expected  image.Point
	}

	// Create an image processor with default values
	processor := ImageProcessor{
		imageSize:   560, // Example value
		numChannels: 3,
		factor:      28,
		minPixels:   56 * 56,
		maxPixels:   14 * 14 * 4 * 1280,
	}

	cases := []smartResizeCase{
		{
			TestImage: image.NewRGBA(image.Rect(0, 0, 1024, 1024)),
			Expected:  image.Point{980, 980},
		},
		{
			TestImage: image.NewRGBA(image.Rect(0, 0, 1024, 768)),
			Expected:  image.Point{1036, 756},
		},
		{
			TestImage: image.NewRGBA(image.Rect(0, 0, 2000, 2000)),
			Expected:  image.Point{980, 980},
		},
	}

	for _, c := range cases {
		b := c.TestImage.Bounds().Max
		x, y := processor.SmartResize(b.X, b.Y)
		actual := image.Point{x, y}
		if actual != c.Expected {
			t.Errorf("expected: %v, actual: %v", c.Expected, actual)
		}
	}
}
