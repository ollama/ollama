package imageproc

import (
	"image"
	"reflect"
	"testing"
)

func testEq(a, b any) bool {
	va := reflect.ValueOf(a)
	vb := reflect.ValueOf(b)

	if va.Kind() != reflect.Slice || vb.Kind() != reflect.Slice {
		return false
	}

	if va.Len() != vb.Len() {
		return false
	}

	for i := range va.Len() {
		if !reflect.DeepEqual(va.Index(i).Interface(), vb.Index(i).Interface()) {
			return false
		}
	}
	return true
}

func TestAspectRatios(t *testing.T) {
	type AspectCase struct {
		MaxTiles int
		Expected []image.Point
	}

	cases := []AspectCase{
		{
			MaxTiles: 1,
			Expected: []image.Point{{1, 1}},
		},
		{
			MaxTiles: 2,
			Expected: []image.Point{{1, 1}, {1, 2}, {2, 1}},
		},
		{
			MaxTiles: 3,
			Expected: []image.Point{{1, 1}, {1, 2}, {1, 3}, {2, 1}, {3, 1}},
		},
		{
			MaxTiles: 4,
			Expected: []image.Point{{1, 1}, {1, 2}, {1, 3}, {1, 4}, {2, 1}, {2, 2}, {3, 1}, {4, 1}},
		},
	}

	for _, c := range cases {
		actual := GetSupportedAspectRatios(c.MaxTiles)

		if !testEq(actual, c.Expected) {
			t.Errorf("incorrect aspect ratio: '%#v'. expected: '%#v'", actual, c.Expected)
		}
	}
}

func TestGetImageSizeFitToCanvas(t *testing.T) {
	type ImageSizeCase struct {
		ImageRect  image.Point
		CanvasRect image.Point
		TileSize   int
		Expected   image.Point
	}

	cases := []ImageSizeCase{
		{
			ImageRect:  image.Point{400, 400},
			CanvasRect: image.Point{640, 480},
			TileSize:   200,
			Expected:   image.Point{400, 400},
		},
		{
			ImageRect:  image.Point{1024, 768},
			CanvasRect: image.Point{640, 480},
			TileSize:   200,
			Expected:   image.Point{640, 480},
		},
		{
			ImageRect:  image.Point{500, 500},
			CanvasRect: image.Point{1000, 1000},
			TileSize:   750,
			Expected:   image.Point{750, 750},
		},
		{
			ImageRect:  image.Point{500, 1000},
			CanvasRect: image.Point{2000, 2000},
			TileSize:   2000,
			Expected:   image.Point{1000, 2000},
		},
		{
			ImageRect:  image.Point{4000, 3000},
			CanvasRect: image.Point{2000, 1000},
			TileSize:   1000,
			Expected:   image.Point{1333, 1000},
		},
		{
			ImageRect:  image.Point{667, 1000},
			CanvasRect: image.Point{1000, 1000},
			TileSize:   560,
			Expected:   image.Point{667, 1000},
		},
	}

	for _, c := range cases {
		actual := GetImageSizeFitToCanvas(c.ImageRect, c.CanvasRect, c.TileSize)

		if actual != c.Expected {
			t.Errorf("incorrect image rect: '%#v'. expected: '%#v'", actual, c.Expected)
		}
	}
}

func TestGetOptimalTiledCanvas(t *testing.T) {
	type TiledCanvasSizeCase struct {
		ImageSize     image.Point
		MaxImageTiles int
		TileSize      int
		Expected      image.Point
	}

	cases := []TiledCanvasSizeCase{
		{
			ImageSize:     image.Point{1024, 768},
			MaxImageTiles: 4,
			TileSize:      1000,
			Expected:      image.Point{4000, 1000},
		},
		{
			ImageSize:     image.Point{1024, 768},
			MaxImageTiles: 4,
			TileSize:      560,
			Expected:      image.Point{1120, 1120},
		},
	}

	for _, c := range cases {
		actual := GetOptimalTiledCanvas(c.ImageSize, c.MaxImageTiles, c.TileSize)

		if actual != c.Expected {
			t.Errorf("incorrect tiled canvas: '%#v'. expected: '%#v'", actual, c.Expected)
		}
	}
}

func TestSplitToTiles(t *testing.T) {
	type SplitCase struct {
		TestImage    image.Image
		NumTilesSize image.Point
		Expected     []image.Image
	}

	cases := []SplitCase{
		{
			TestImage:    image.NewRGBA(image.Rect(0, 0, 1024, 768)),
			NumTilesSize: image.Point{1, 1},
			Expected:     []image.Image{image.NewRGBA(image.Rect(0, 0, 1024, 768))},
		},
		{
			TestImage:    image.NewRGBA(image.Rect(0, 0, 1000, 500)),
			NumTilesSize: image.Point{2, 1},
			Expected: []image.Image{
				image.NewRGBA(image.Rect(0, 0, 500, 500)),
				image.NewRGBA(image.Rect(500, 0, 1000, 500)),
			},
		},
		{
			TestImage:    image.NewRGBA(image.Rect(0, 0, 1000, 1000)),
			NumTilesSize: image.Point{2, 2},
			Expected: []image.Image{
				image.NewRGBA(image.Rect(0, 0, 500, 500)),
				image.NewRGBA(image.Rect(500, 0, 1000, 500)),
				image.NewRGBA(image.Rect(0, 500, 500, 1000)),
				image.NewRGBA(image.Rect(500, 500, 1000, 1000)),
			},
		},
	}

	for _, c := range cases {
		actual := SplitToTiles(c.TestImage, c.NumTilesSize)

		if len(actual) != len(c.Expected) {
			t.Errorf("incorrect number of images '%d': expected: '%d'", len(actual), len(c.Expected))
		}

		for i := range actual {
			if actual[i].Bounds() != c.Expected[i].Bounds() {
				t.Errorf("image size incorrect: '%#v': expected: '%#v'", actual[i].Bounds(), c.Expected[i].Bounds())
			}
		}
	}
}

func TestResize(t *testing.T) {
	type ResizeCase struct {
		TestImage           image.Image
		OutputSize          image.Point
		MaxImageTiles       int
		ExpectedImage       image.Image
		ExpectedAspectRatio image.Point
	}

	cases := []ResizeCase{
		{
			TestImage:           image.NewRGBA(image.Rect(0, 0, 200, 200)),
			OutputSize:          image.Point{100, 100},
			MaxImageTiles:       1,
			ExpectedImage:       image.NewRGBA(image.Rect(0, 0, 100, 100)),
			ExpectedAspectRatio: image.Point{1, 1},
		},
		{
			TestImage:           image.NewRGBA(image.Rect(0, 0, 200, 200)),
			OutputSize:          image.Point{100, 100},
			MaxImageTiles:       2,
			ExpectedImage:       image.NewRGBA(image.Rect(0, 0, 100, 100)),
			ExpectedAspectRatio: image.Point{1, 2},
		},
		{
			TestImage:           image.NewRGBA(image.Rect(0, 0, 2560, 1920)),
			OutputSize:          image.Point{560, 560},
			MaxImageTiles:       4,
			ExpectedImage:       image.NewRGBA(image.Rect(0, 0, 1120, 840)),
			ExpectedAspectRatio: image.Point{2, 2},
		},
		{
			TestImage:           image.NewRGBA(image.Rect(0, 0, 1024, 768)),
			OutputSize:          image.Point{560, 560},
			MaxImageTiles:       4,
			ExpectedImage:       image.NewRGBA(image.Rect(0, 0, 1024, 768)),
			ExpectedAspectRatio: image.Point{2, 2},
		},
	}

	for _, c := range cases {
		actualImage, actualAspectRatio := ResizeImage(c.TestImage, c.OutputSize, c.MaxImageTiles)

		if actualImage.Bounds() != c.ExpectedImage.Bounds() {
			t.Errorf("image size incorrect: '%#v': expected: '%#v'", actualImage.Bounds(), c.ExpectedImage.Bounds())
		}

		if actualAspectRatio != c.ExpectedAspectRatio {
			t.Errorf("canvas size incorrect: '%#v': expected: '%#v'", actualAspectRatio, c.ExpectedAspectRatio)
		}
	}
}

func TestPad(t *testing.T) {
	type PadCase struct {
		TestImage   image.Image
		OutputSize  image.Point
		AspectRatio image.Point
		Expected    image.Image
	}

	cases := []PadCase{
		{
			TestImage:   image.NewRGBA(image.Rect(0, 0, 1000, 667)),
			OutputSize:  image.Point{560, 560},
			AspectRatio: image.Point{2, 2},
			Expected:    image.NewRGBA(image.Rect(0, 0, 1120, 1120)),
		},
	}

	for _, c := range cases {
		actual := PadImage(c.TestImage, c.OutputSize, c.AspectRatio)

		if actual.Bounds() != c.Expected.Bounds() {
			t.Errorf("image size incorrect: '%#v': expected: '%#v'", actual.Bounds(), c.Expected.Bounds())
		}
	}
}

func TestPackImages(t *testing.T) {
	type PackCase struct {
		TestImage   image.Image
		AspectRatio image.Point
	}

	mean := [3]float32{0.48145466, 0.4578275, 0.40821073}
	std := [3]float32{0.26862954, 0.26130258, 0.27577711}

	cases := []PackCase{
		{
			TestImage:   image.NewRGBA(image.Rect(0, 0, 1120, 1120)),
			AspectRatio: image.Point{2, 2},
		},
		{
			TestImage:   image.NewRGBA(image.Rect(0, 0, 560, 560)),
			AspectRatio: image.Point{1, 1},
		},
		{
			TestImage:   image.NewRGBA(image.Rect(0, 0, 1120, 560)),
			AspectRatio: image.Point{1, 2},
		},
	}

	for _, c := range cases {
		PackImages(c.TestImage, c.AspectRatio, mean, std)
	}
}
