package mllama

import (
	"bytes"
	"image"
	"image/png"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestAspectRatios(t *testing.T) {
	type aspectCase struct {
		MaxTiles int
		Expected []image.Point
	}

	cases := []aspectCase{
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
		actual := getSupportedAspectRatios(c.MaxTiles)

		if diff := cmp.Diff(actual, c.Expected); diff != "" {
			t.Errorf("mismatch (-got +want):\n%s", diff)
		}
	}
}

func TestGetImageSizeFitToCanvas(t *testing.T) {
	type imageSizeCase struct {
		ImageRect  image.Point
		CanvasRect image.Point
		TileSize   int
		Expected   image.Point
	}

	cases := []imageSizeCase{
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
		actual := getImageSizeFitToCanvas(c.ImageRect, c.CanvasRect, c.TileSize)

		if actual != c.Expected {
			t.Errorf("incorrect image rect: '%#v'. expected: '%#v'", actual, c.Expected)
		}
	}
}

func TestGetOptimalTiledCanvas(t *testing.T) {
	type tiledCanvasSizeCase struct {
		ImageSize     image.Point
		MaxImageTiles int
		TileSize      int
		Expected      image.Point
	}

	cases := []tiledCanvasSizeCase{
		{
			ImageSize:     image.Point{1024, 768},
			MaxImageTiles: 4,
			TileSize:      1000,
			Expected:      image.Point{2000, 1000},
		},
		{
			ImageSize:     image.Point{1024, 768},
			MaxImageTiles: 4,
			TileSize:      560,
			Expected:      image.Point{1120, 1120},
		},
		{
			ImageSize:     image.Point{800, 600},
			MaxImageTiles: 4,
			TileSize:      560,
			Expected:      image.Point{1120, 1120},
		},
		{
			ImageSize:     image.Point{640, 480},
			MaxImageTiles: 4,
			TileSize:      560,
			Expected:      image.Point{1120, 560},
		},
		{
			ImageSize:     image.Point{320, 200},
			MaxImageTiles: 4,
			TileSize:      560,
			Expected:      image.Point{560, 560},
		},
		{
			ImageSize:     image.Point{1320, 200},
			MaxImageTiles: 4,
			TileSize:      560,
			Expected:      image.Point{1680, 560},
		},
		{
			ImageSize:     image.Point{2000, 200},
			MaxImageTiles: 4,
			TileSize:      560,
			Expected:      image.Point{2240, 560},
		},
		{
			ImageSize:     image.Point{10000, 200},
			MaxImageTiles: 4,
			TileSize:      560,
			Expected:      image.Point{2240, 560},
		},
		{
			ImageSize:     image.Point{480, 640},
			MaxImageTiles: 4,
			TileSize:      560,
			Expected:      image.Point{560, 1120},
		},
		{
			ImageSize:     image.Point{200, 320},
			MaxImageTiles: 4,
			TileSize:      560,
			Expected:      image.Point{560, 560},
		},
		{
			ImageSize:     image.Point{200, 1320},
			MaxImageTiles: 4,
			TileSize:      560,
			Expected:      image.Point{560, 1680},
		},
		{
			ImageSize:     image.Point{200, 2000},
			MaxImageTiles: 4,
			TileSize:      560,
			Expected:      image.Point{560, 2240},
		},
		{
			ImageSize:     image.Point{200, 10000},
			MaxImageTiles: 4,
			TileSize:      560,
			Expected:      image.Point{560, 2240},
		},
		{
			ImageSize:     image.Point{10000, 10000},
			MaxImageTiles: 4,
			TileSize:      560,
			Expected:      image.Point{1120, 1120},
		},
	}

	for _, c := range cases {
		actual := getOptimalTiledCanvas(c.ImageSize, c.MaxImageTiles, c.TileSize)

		if actual != c.Expected {
			t.Errorf("incorrect tiled canvas: '%#v'. expected: '%#v'", actual, c.Expected)
		}
	}
}

func TestSplitToTiles(t *testing.T) {
	type splitCase struct {
		TestImage    image.Image
		NumTilesSize image.Point
		Expected     []image.Image
	}

	cases := []splitCase{
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
		actual := splitToTiles(c.TestImage, c.NumTilesSize)

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
	type resizeCase struct {
		TestImage           image.Image
		OutputSize          image.Point
		MaxImageTiles       int
		ExpectedImage       image.Image
		ExpectedAspectRatio image.Point
	}

	cases := []resizeCase{
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
			ExpectedAspectRatio: image.Point{1, 1},
		},
		{
			TestImage:           image.NewRGBA(image.Rect(0, 0, 10, 10)),
			OutputSize:          image.Point{560, 560},
			MaxImageTiles:       4,
			ExpectedImage:       image.NewRGBA(image.Rect(0, 0, 560, 560)),
			ExpectedAspectRatio: image.Point{1, 1},
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
		actualImage, actualAspectRatio := resizeImage(c.TestImage, "png", c.OutputSize, c.MaxImageTiles)

		if actualImage.Bounds() != c.ExpectedImage.Bounds() {
			t.Errorf("image size incorrect: '%#v': expected: '%#v'", actualImage.Bounds(), c.ExpectedImage.Bounds())
		}

		if actualAspectRatio != c.ExpectedAspectRatio {
			t.Errorf("aspect ratio incorrect: '%#v': expected: '%#v'", actualAspectRatio, c.ExpectedAspectRatio)
		}
	}
}

func TestPad(t *testing.T) {
	type padCase struct {
		TestImage   image.Image
		OutputSize  image.Point
		AspectRatio image.Point
		Expected    image.Image
	}

	cases := []padCase{
		{
			TestImage:   image.NewRGBA(image.Rect(0, 0, 1000, 667)),
			OutputSize:  image.Point{560, 560},
			AspectRatio: image.Point{2, 2},
			Expected:    image.NewRGBA(image.Rect(0, 0, 1120, 1120)),
		},
	}

	for _, c := range cases {
		actual := padImage(c.TestImage, c.OutputSize, c.AspectRatio)

		if actual.Bounds() != c.Expected.Bounds() {
			t.Errorf("image size incorrect: '%#v': expected: '%#v'", actual.Bounds(), c.Expected.Bounds())
		}
	}
}

func TestPackImages(t *testing.T) {
	type packCase struct {
		TestImage    image.Image
		AspectRatio  image.Point
		ExpectedVals int
	}

	cases := []packCase{
		{
			TestImage:    image.NewRGBA(image.Rect(0, 0, 1120, 1120)),
			AspectRatio:  image.Point{2, 2},
			ExpectedVals: 2 * 2 * 3 * 560 * 560,
		},
		{
			TestImage:    image.NewRGBA(image.Rect(0, 0, 560, 560)),
			AspectRatio:  image.Point{1, 1},
			ExpectedVals: 1 * 1 * 3 * 560 * 560,
		},
		{
			TestImage:    image.NewRGBA(image.Rect(0, 0, 1120, 560)),
			AspectRatio:  image.Point{1, 2},
			ExpectedVals: 1 * 2 * 3 * 560 * 560,
		},
	}

	for _, c := range cases {
		actualVals := packImages(c.TestImage, c.AspectRatio)
		if len(actualVals) != c.ExpectedVals {
			t.Errorf("packed image size incorrect: '%d': expected: '%d'", len(actualVals), c.ExpectedVals)
		}
	}
}

func TestPreprocess(t *testing.T) {
	type preprocessCase struct {
		TestImage             image.Image
		ExpectedVals          int
		ExpectedAspectRatioID int
	}

	cases := []preprocessCase{
		{
			TestImage:             image.NewRGBA(image.Rect(0, 0, 10, 10)),
			ExpectedVals:          0,
			ExpectedAspectRatioID: 1,
		},
		{
			TestImage:             image.NewRGBA(image.Rect(0, 0, 1024, 768)),
			ExpectedVals:          0,
			ExpectedAspectRatioID: 6,
		},
	}

	for _, c := range cases {
		var buf bytes.Buffer
		err := png.Encode(&buf, c.TestImage)
		if err != nil {
			t.Fatal(err)
		}

		imgData, opts, err := Preprocess(&buf)
		if err != nil {
			t.Fatalf("error processing: %q", err)
		}

		if len(imgData) == 0 {
			t.Errorf("no image data returned")
		}

		ar, ok := opts["aspectRatioIndex"]
		if !ok {
			t.Fatalf("no aspect ratio found")
		}

		aspectRatioID := ar.(int)

		if aspectRatioID != c.ExpectedAspectRatioID {
			t.Errorf("aspect ratio incorrect: '%d': expected: '%d'", aspectRatioID, c.ExpectedAspectRatioID)
		}
	}
}
