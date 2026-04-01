package llama4

import (
	"cmp"
	"image"
	"image/color"
	"reflect"
	"slices"
	"testing"

	gocmp "github.com/google/go-cmp/cmp"
)

func TestFactors(t *testing.T) {
	tests := []struct {
		name     string
		input    int
		expected []int
	}{
		{
			name:     "factors of 1",
			input:    1,
			expected: []int{1},
		},
		{
			name:     "factors of 2",
			input:    2,
			expected: []int{1, 2},
		},
		{
			name:     "factors of 6",
			input:    6,
			expected: []int{1, 2, 3, 6},
		},
		{
			name:     "factors of 28",
			input:    28,
			expected: []int{1, 2, 4, 7, 14, 28},
		},
		{
			name:     "factors of 49",
			input:    49,
			expected: []int{1, 7, 49},
		},
		{
			name:     "factors of 97 (prime)",
			input:    97,
			expected: []int{1, 97},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			actual := factors(tt.input)
			if !reflect.DeepEqual(actual, tt.expected) {
				t.Errorf("factors(%d) = %v; want %v", tt.input, actual, tt.expected)
			}
		})
	}
}

func TestSupportedResolutions(t *testing.T) {
	expectedResolutions := []image.Point{
		{X: 3360, Y: 336},
		{X: 672, Y: 2688},
		{X: 336, Y: 1344},
		{X: 336, Y: 4032},
		{X: 1008, Y: 1344},
		{X: 1344, Y: 1008},
		{X: 336, Y: 1680},
		{X: 1680, Y: 336},
		{X: 336, Y: 5040},
		{X: 4032, Y: 336},
		{X: 2352, Y: 336},
		{X: 2688, Y: 672},
		{X: 1344, Y: 336},
		{X: 5376, Y: 336},
		{X: 2352, Y: 672},
		{X: 672, Y: 1008},
		{X: 1008, Y: 672},
		{X: 336, Y: 5376},
		{X: 1680, Y: 1008},
		{X: 5040, Y: 336},
		{X: 336, Y: 3024},
		{X: 3024, Y: 336},
		{X: 336, Y: 2688},
		{X: 672, Y: 1344},
		{X: 336, Y: 672},
		{X: 336, Y: 2352},
		{X: 2016, Y: 672},
		{X: 1008, Y: 336},
		{X: 336, Y: 3360},
		{X: 336, Y: 4368},
		{X: 1008, Y: 1680},
		{X: 336, Y: 4704},
		{X: 4704, Y: 336},
		{X: 1344, Y: 672},
		{X: 672, Y: 336},
		{X: 2688, Y: 336},
		{X: 3696, Y: 336},
		{X: 2016, Y: 336},
		{X: 1344, Y: 1344},
		{X: 1008, Y: 1008},
		{X: 672, Y: 672},
		{X: 336, Y: 336},
		{X: 4368, Y: 336},
		{X: 672, Y: 2016},
		{X: 336, Y: 1008},
		{X: 336, Y: 3696},
		{X: 672, Y: 1680},
		{X: 1680, Y: 672},
		{X: 336, Y: 2016},
		{X: 672, Y: 2352},
	}

	sortResolutionFunc := func(a, b image.Point) int {
		return cmp.Or(cmp.Compare(a.X, b.X), cmp.Compare(a.Y, b.Y))
	}

	slices.SortStableFunc(expectedResolutions, sortResolutionFunc)

	imgProc := ImageProcessor{
		imageSize:        336,
		patchSize:        16,
		numChannels:      3,
		maxUpscalingSize: 448,
	}

	actualResolutions := imgProc.supportedResolutions()
	slices.SortStableFunc(actualResolutions, sortResolutionFunc)

	if diff := gocmp.Diff(expectedResolutions, actualResolutions); diff != "" {
		t.Errorf("supportedResolutions() mismatch (-want +got):\n%s", diff)
	}
}

func TestBestResolution(t *testing.T) {
	tests := []struct {
		name        string
		size        image.Point
		resolutions []image.Point
		max         bool
		expected    image.Point
	}{
		{
			"normal",
			image.Point{800, 600},
			[]image.Point{
				{300, 200},
				{640, 480},
				{800, 600},
				{1024, 768},
				{1600, 1200},
			},
			false,
			image.Point{800, 600},
		},
		{
			"max",
			image.Point{800, 600},
			[]image.Point{
				{300, 200},
				{640, 480},
				{800, 600},
				{1024, 768},
				{1600, 1200},
			},
			true,
			image.Point{1600, 1200},
		},
		{
			"mid",
			image.Point{1000, 700},
			[]image.Point{
				{300, 200},
				{640, 480},
				{800, 600},
				{1024, 768},
				{1600, 1200},
			},
			false,
			image.Point{1024, 768},
		},
		{
			"smol",
			image.Point{100, 100},
			[]image.Point{
				{300, 200},
				{640, 480},
				{800, 600},
				{1024, 768},
				{1600, 1200},
			},
			false,
			image.Point{300, 200},
		},
		{
			"huge",
			image.Point{10000, 10000},
			[]image.Point{
				{300, 200},
				{640, 480},
				{800, 600},
				{1024, 768},
				{1600, 1200},
			},
			false,
			image.Point{1600, 1200},
		},
	}

	p := ImageProcessor{}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			actual := p.bestResolution(tt.size, tt.resolutions, tt.max)
			if diff := gocmp.Diff(tt.expected, actual); diff != "" {
				t.Errorf("best resolution mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestMaxResolution(t *testing.T) {
	tests := []struct {
		name      string
		origRes   image.Point
		targetRes image.Point
		expected  image.Point
	}{
		{
			"normal",
			image.Point{800, 600},
			image.Point{800, 600},
			image.Point{800, 600},
		},
		{
			"skew",
			image.Point{800, 600},
			image.Point{1100, 700},
			image.Point{933, 700},
		},
	}

	p := ImageProcessor{}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			actual := p.maxResolution(tt.origRes, tt.targetRes)
			if !reflect.DeepEqual(actual, tt.expected) {
				t.Errorf("max resolution; got %v want %v", actual, tt.expected)
			}
		})
	}
}

func TestProcessImage(t *testing.T) {
	imgProc := ImageProcessor{
		imageSize:        336,
		patchSize:        16,
		numChannels:      3,
		maxUpscalingSize: 448,
	}

	generateImage := func(seed int) image.Image {
		width, height := 20, 10
		img := image.NewRGBA(image.Rect(0, 0, width, height))

		for x := range width {
			// Use the seed to vary color generation
			r := uint8((seed + x*11) % 256)
			g := uint8((seed + x*17) % 256)
			b := uint8((seed + x*23) % 256)

			c := color.RGBA{R: r, G: g, B: b, A: 255}
			for y := range height {
				img.Set(x, y, c)
			}
		}

		return img
	}

	pixelsLocal, pixelsGlobal, targetSize, err := imgProc.ProcessImage(generateImage(12))
	if err != nil {
		t.Error(err)
	}

	if n := len(pixelsLocal); n != 336*336*3 {
		t.Errorf("unexpected size of f32s: %d", n)
	}

	if n := len(pixelsGlobal); n > 0 {
		t.Errorf("unexpected size of f32s: %d", n)
	}

	if !targetSize.Eq(image.Point{336, 336}) {
		t.Errorf("unexpected target size: %v", targetSize)
	}
}
