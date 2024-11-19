package imageproc

import (
	"image"
	"image/color"
	"image/draw"
	"reflect"
	"testing"
)

func createImage(width, height int, fillCol color.RGBA) image.Image {
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	draw.Draw(img, img.Bounds(), &image.Uniform{fillCol}, image.Point{}, draw.Src)
	return img
}

func TestComposite(t *testing.T) {
	tests := []struct {
		name         string
		img          image.Image
		expectedRGBA color.RGBA
	}{
		{
			name:         "Transparent image",
			img:          createImage(5, 5, color.RGBA{0, 0, 0, 0}),
			expectedRGBA: color.RGBA{255, 255, 255, 255},
		},
		{
			name:         "Solid red image",
			img:          createImage(5, 5, color.RGBA{255, 0, 0, 255}),
			expectedRGBA: color.RGBA{255, 0, 0, 255},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resultImg := Composite(tt.img)

			// Check the pixel values in the resulting image
			for x := range resultImg.Bounds().Dx() {
				for y := range resultImg.Bounds().Dy() {
					r, g, b, a := resultImg.At(x, y).RGBA()
					expectedR, expectedG, expectedB, expectedA := tt.expectedRGBA.RGBA()

					if r != expectedR || g != expectedG || b != expectedB || a != expectedA {
						t.Errorf("Pixel mismatch at (%d, %d): got (%d, %d, %d, %d), want (%d, %d, %d, %d)",
							x, y, r, g, b, a, expectedR, expectedG, expectedB, expectedA)
					}
				}
			}
		})
	}
}

func TestResize(t *testing.T) {
	tests := []struct {
		name     string
		img      image.Image
		newSize  image.Point
		method   int
		expected image.Point
	}{
		{
			name:     "Resize with bilinear interpolation",
			img:      createImage(5, 5, color.RGBA{255, 0, 0, 255}),
			newSize:  image.Point{10, 10},
			method:   ResizeBilinear,
			expected: image.Point{10, 10},
		},
		{
			name:     "Resize with nearest neighbor",
			img:      createImage(10, 10, color.RGBA{0, 255, 0, 255}),
			newSize:  image.Point{5, 5},
			method:   ResizeNearestNeighbor,
			expected: image.Point{5, 5},
		},
		{
			name:     "Resize with catmullrom",
			img:      createImage(1024, 1024, color.RGBA{0, 0, 255, 255}),
			newSize:  image.Point{10, 10},
			method:   ResizeCatmullrom,
			expected: image.Point{10, 10},
		},
		{
			name:     "Resize with approx bilinear",
			img:      createImage(1024, 768, color.RGBA{100, 100, 100, 255}),
			newSize:  image.Point{4, 3},
			method:   ResizeApproxBilinear,
			expected: image.Point{4, 3},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resizedImg := Resize(tt.img, tt.newSize, tt.method)

			if resizedImg.Bounds().Dx() != tt.expected.X || resizedImg.Bounds().Dy() != tt.expected.Y {
				t.Errorf("Unexpected size for resized image: got (%d, %d), want (%d, %d)",
					resizedImg.Bounds().Dx(), resizedImg.Bounds().Dy(), tt.expected.X, tt.expected.Y)
			}
		})
	}
}

func TestResizeInvalidMethod(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic for invalid resizing method, but did not panic")
		}
	}()

	img := createImage(10, 10, color.RGBA{0, 0, 0, 255})
	Resize(img, image.Point{5, 5}, -1)
}

func TestNormalize(t *testing.T) {
	tests := []struct {
		name         string
		img          image.Image
		mean         [3]float32
		std          [3]float32
		rescale      bool
		channelFirst bool
		expected     []float32
	}{
		{
			name:         "Rescale with channel first",
			img:          createImage(2, 2, color.RGBA{128, 128, 128, 255}),
			mean:         ImageNetStandardMean,
			std:          ImageNetStandardSTD,
			rescale:      true,
			channelFirst: true,
			expected: []float32{
				0.003921628, 0.003921628, 0.003921628, 0.003921628, // R values
				0.003921628, 0.003921628, 0.003921628, 0.003921628, // G values
				0.003921628, 0.003921628, 0.003921628, 0.003921628, // B values
			},
		},
		{
			name:         "Rescale without channel first",
			img:          createImage(2, 2, color.RGBA{255, 0, 0, 255}),
			mean:         [3]float32{0.0, 0.0, 0.0},
			std:          [3]float32{1.0, 1.0, 1.0},
			rescale:      true,
			channelFirst: false,
			expected: []float32{
				1.0, 0.0, 0.0,
				1.0, 0.0, 0.0,
				1.0, 0.0, 0.0,
				1.0, 0.0, 0.0,
			},
		},
		{
			name:         "No rescale with mean/std adjustment",
			img:          createImage(2, 2, color.RGBA{100, 150, 200, 255}),
			mean:         ClipDefaultMean,
			std:          ClipDefaultSTD,
			rescale:      false,
			channelFirst: false,
			expected: []float32{
				-1.7922626, -1.7520971, -1.4802198,
				-1.7922626, -1.7520971, -1.4802198,
				-1.7922626, -1.7520971, -1.4802198,
				-1.7922626, -1.7520971, -1.4802198,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := Normalize(tt.img, tt.mean, tt.std, tt.rescale, tt.channelFirst)

			if !reflect.DeepEqual(result, tt.expected) {
				t.Errorf("Test %s failed: got %v, want %v", tt.name, result, tt.expected)
			}
		})
	}
}
