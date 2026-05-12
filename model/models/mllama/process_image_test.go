package mllama

import (
	"image"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestSupportedAspectRatios(t *testing.T) {
	cases := []struct {
		p    ImageProcessor
		want []supportedAspectRatio
	}{
		{
			p: ImageProcessor{maxNumTiles: 1},
			want: []supportedAspectRatio{
				{1, 1, 1},
			},
		},
		{
			p: ImageProcessor{maxNumTiles: 2},
			want: []supportedAspectRatio{
				{1, 1, 1},
				{2, 1, 2},
				{3, 2, 1},
			},
		},
		{
			p: ImageProcessor{maxNumTiles: 3},
			want: []supportedAspectRatio{
				{1, 1, 1},
				{2, 1, 2},
				{3, 1, 3},
				{4, 2, 1},
				{5, 3, 1},
			},
		},
		{
			p: ImageProcessor{maxNumTiles: 4},
			want: []supportedAspectRatio{
				{1, 1, 1},
				{2, 1, 2},
				{3, 1, 3},
				{4, 1, 4},
				{5, 2, 1},
				{6, 2, 2},
				{7, 3, 1},
				{8, 4, 1},
			},
		},
	}

	for _, tt := range cases {
		actual := tt.p.supportedAspectRatios()
		if diff := cmp.Diff(actual, tt.want, cmp.AllowUnexported(supportedAspectRatio{})); diff != "" {
			t.Errorf("mismatch (-got +want):\n%s", diff)
		}
	}
}

func TestFitToCanvas(t *testing.T) {
	cases := []struct {
		p      ImageProcessor
		image  image.Point
		canvas image.Point
		expect image.Point
	}{
		{
			p:      ImageProcessor{imageSize: 200},
			image:  image.Point{400, 400},
			canvas: image.Point{640, 480},
			expect: image.Point{400, 400},
		},
		{
			p:      ImageProcessor{imageSize: 200},
			image:  image.Point{1024, 768},
			canvas: image.Point{640, 480},
			expect: image.Point{640, 480},
		},
		{
			p:      ImageProcessor{imageSize: 750},
			image:  image.Point{500, 500},
			canvas: image.Point{1000, 1000},
			expect: image.Point{750, 750},
		},
		{
			p:      ImageProcessor{imageSize: 2000},
			image:  image.Point{500, 1000},
			canvas: image.Point{2000, 2000},
			expect: image.Point{1000, 2000},
		},
		{
			p:      ImageProcessor{imageSize: 1000},
			image:  image.Point{4000, 3000},
			canvas: image.Point{2000, 1000},
			expect: image.Point{1333, 1000},
		},
		{
			p:      ImageProcessor{imageSize: 560},
			image:  image.Point{667, 1000},
			canvas: image.Point{1000, 1000},
			expect: image.Point{667, 1000},
		},
	}

	for _, tt := range cases {
		actual := tt.p.fitToCanvas(tt.image, tt.canvas)
		if diff := cmp.Diff(actual, tt.expect); diff != "" {
			t.Errorf("mismatch (-got +want):\n%s", diff)
		}
	}
}

func TestOptimalTiledCanvas(t *testing.T) {
	cases := []struct {
		p      ImageProcessor
		image  image.Point
		expect image.Point
	}{
		{
			p:      ImageProcessor{maxNumTiles: 4, imageSize: 1000},
			image:  image.Point{1024, 768},
			expect: image.Point{2000, 1000},
		},
		{
			p:      ImageProcessor{maxNumTiles: 4, imageSize: 560},
			image:  image.Point{1024, 768},
			expect: image.Point{1120, 1120},
		},
		{
			p:      ImageProcessor{maxNumTiles: 4, imageSize: 560},
			image:  image.Point{800, 600},
			expect: image.Point{1120, 1120},
		},
		{
			p:      ImageProcessor{maxNumTiles: 4, imageSize: 560},
			image:  image.Point{640, 480},
			expect: image.Point{1120, 560},
		},
		{
			p:      ImageProcessor{maxNumTiles: 4, imageSize: 560},
			image:  image.Point{320, 200},
			expect: image.Point{560, 560},
		},
		{
			p:      ImageProcessor{maxNumTiles: 4, imageSize: 560},
			image:  image.Point{1320, 200},
			expect: image.Point{1680, 560},
		},
		{
			p:      ImageProcessor{maxNumTiles: 4, imageSize: 560},
			image:  image.Point{2000, 200},
			expect: image.Point{2240, 560},
		},
		{
			p:      ImageProcessor{maxNumTiles: 4, imageSize: 560},
			image:  image.Point{10000, 200},
			expect: image.Point{2240, 560},
		},
		{
			p:      ImageProcessor{maxNumTiles: 4, imageSize: 560},
			image:  image.Point{480, 640},
			expect: image.Point{560, 1120},
		},
		{
			p:      ImageProcessor{maxNumTiles: 4, imageSize: 560},
			image:  image.Point{200, 320},
			expect: image.Point{560, 560},
		},
		{
			p:      ImageProcessor{maxNumTiles: 4, imageSize: 560},
			image:  image.Point{200, 1320},
			expect: image.Point{560, 1680},
		},
		{
			p:      ImageProcessor{maxNumTiles: 4, imageSize: 560},
			image:  image.Point{200, 2000},
			expect: image.Point{560, 2240},
		},
		{
			p:      ImageProcessor{maxNumTiles: 4, imageSize: 560},
			image:  image.Point{200, 10000},
			expect: image.Point{560, 2240},
		},
		{
			p:      ImageProcessor{maxNumTiles: 4, imageSize: 560},
			image:  image.Point{10000, 10000},
			expect: image.Point{1120, 1120},
		},
	}

	for _, tt := range cases {
		actual := tt.p.optimalTiledCanvas(tt.image)
		if diff := cmp.Diff(actual, tt.expect); diff != "" {
			t.Errorf("mismatch (-got +want):\n%s", diff)
		}
	}
}

func TestSplitToTiles(t *testing.T) {
	cases := []struct {
		imageMax image.Point
		numTiles image.Point
		expect   []image.Image
	}{
		{
			imageMax: image.Point{1024, 768},
			numTiles: image.Point{1, 1},
			expect:   []image.Image{image.NewRGBA(image.Rect(0, 0, 1024, 768))},
		},
		{
			imageMax: image.Point{1000, 500},
			numTiles: image.Point{2, 1},
			expect: []image.Image{
				image.NewRGBA(image.Rect(0, 0, 500, 500)),
				image.NewRGBA(image.Rect(500, 0, 1000, 500)),
			},
		},
		{
			imageMax: image.Point{1000, 1000},
			numTiles: image.Point{2, 2},
			expect: []image.Image{
				image.NewRGBA(image.Rect(0, 0, 500, 500)),
				image.NewRGBA(image.Rect(500, 0, 1000, 500)),
				image.NewRGBA(image.Rect(0, 500, 500, 1000)),
				image.NewRGBA(image.Rect(500, 500, 1000, 1000)),
			},
		},
	}

	var p ImageProcessor

	for _, tt := range cases {
		actual := p.splitToTiles(image.NewRGBA(image.Rectangle{Max: tt.imageMax}), tt.numTiles)

		if len(actual) != len(tt.expect) {
			t.Errorf("incorrect number of images '%d': expect: '%d'", len(actual), len(tt.expect))
		}

		for i := range actual {
			if actual[i].Bounds() != tt.expect[i].Bounds() {
				t.Errorf("image size incorrect: '%#v': expect: '%#v'", actual[i].Bounds(), tt.expect[i].Bounds())
			}
		}
	}
}

func TestResize(t *testing.T) {
	cases := []struct {
		p                 ImageProcessor
		imageMax          image.Point
		expectImage       image.Image
		expectAspectRatio image.Point
	}{
		{
			p:                 ImageProcessor{maxNumTiles: 1, imageSize: 100},
			imageMax:          image.Point{200, 200},
			expectImage:       image.NewRGBA(image.Rect(0, 0, 100, 100)),
			expectAspectRatio: image.Point{1, 1},
		},
		{
			p:                 ImageProcessor{maxNumTiles: 2, imageSize: 100},
			imageMax:          image.Point{200, 200},
			expectImage:       image.NewRGBA(image.Rect(0, 0, 100, 100)),
			expectAspectRatio: image.Point{1, 1},
		},
		{
			p:                 ImageProcessor{maxNumTiles: 4, imageSize: 560},
			imageMax:          image.Point{10, 10},
			expectImage:       image.NewRGBA(image.Rect(0, 0, 560, 560)),
			expectAspectRatio: image.Point{1, 1},
		},
		{
			p:                 ImageProcessor{maxNumTiles: 4, imageSize: 560},
			imageMax:          image.Point{2560, 1920},
			expectImage:       image.NewRGBA(image.Rect(0, 0, 1120, 840)),
			expectAspectRatio: image.Point{2, 2},
		},
		{
			p:                 ImageProcessor{maxNumTiles: 4, imageSize: 560},
			imageMax:          image.Point{1024, 768},
			expectImage:       image.NewRGBA(image.Rect(0, 0, 1024, 768)),
			expectAspectRatio: image.Point{2, 2},
		},
	}

	for _, tt := range cases {
		actualImage, actualAspectRatio := tt.p.resize(image.Rectangle{Max: tt.imageMax})

		if actualImage.Bounds() != tt.expectImage.Bounds() {
			t.Errorf("image size incorrect: '%#v': expect: '%#v'", actualImage.Bounds(), tt.expectImage.Bounds())
		}

		if actualAspectRatio != tt.expectAspectRatio {
			t.Errorf("aspect ratio incorrect: '%#v': expect: '%#v'", actualAspectRatio, tt.expectAspectRatio)
		}
	}
}

func TestPad(t *testing.T) {
	cases := []struct {
		p           ImageProcessor
		imageMax    image.Point
		aspectRatio image.Point
		expect      image.Image
	}{
		{
			p:           ImageProcessor{maxNumTiles: 4, imageSize: 560},
			imageMax:    image.Point{1000, 667},
			aspectRatio: image.Point{2, 2},
			expect:      image.NewRGBA(image.Rect(0, 0, 1120, 1120)),
		},
	}

	for _, tt := range cases {
		actual := tt.p.pad(image.Rectangle{Max: tt.imageMax}, tt.aspectRatio)

		if actual.Bounds() != tt.expect.Bounds() {
			t.Errorf("image size incorrect: '%#v': expect: '%#v'", actual.Bounds(), tt.expect.Bounds())
		}
	}
}

func TestPackImages(t *testing.T) {
	cases := []struct {
		imageMax    image.Point
		aspectRatio image.Point
		expectVals  int
	}{
		{
			imageMax:    image.Point{1120, 1120},
			aspectRatio: image.Point{2, 2},
			expectVals:  2 * 2 * 3 * 560 * 560,
		},
		{
			imageMax:    image.Point{560, 560},
			aspectRatio: image.Point{1, 1},
			expectVals:  1 * 1 * 3 * 560 * 560,
		},
		{
			imageMax:    image.Point{1120, 560},
			aspectRatio: image.Point{1, 2},
			expectVals:  1 * 2 * 3 * 560 * 560,
		},
	}

	for _, tt := range cases {
		var p ImageProcessor
		actualVals := p.pack(image.NewRGBA(image.Rectangle{Max: tt.imageMax}), tt.aspectRatio)
		if len(actualVals) != tt.expectVals {
			t.Errorf("packed image size incorrect: '%d': expect: '%d'", len(actualVals), tt.expectVals)
		}
	}
}

func TestPreprocess(t *testing.T) {
	cases := []struct {
		imageMax            image.Point
		expectAspectRatioID int
	}{
		{
			imageMax:            image.Point{10, 10},
			expectAspectRatioID: 1,
		},
		{
			imageMax:            image.Point{1024, 768},
			expectAspectRatioID: 6,
		},
	}

	p := ImageProcessor{imageSize: 560, maxNumTiles: 4}
	for _, tt := range cases {
		img, aspectRatio, err := p.ProcessImage(image.NewRGBA(image.Rectangle{Max: tt.imageMax}))
		if err != nil {
			t.Fatalf("error processing: %q", err)
		}

		if len(img) == 0 {
			t.Errorf("no image data returned")
		}

		if aspectRatio.rank != tt.expectAspectRatioID {
			t.Errorf("aspect ratio incorrect: '%d': expect: '%d'", aspectRatio, tt.expectAspectRatioID)
		}
	}
}
