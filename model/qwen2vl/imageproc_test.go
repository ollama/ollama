package qwen2vl

import (
	"bytes"
	"image"
	"image/png"
	"testing"
)

func TestSmartResize(t *testing.T) {
	type smartResizeCase struct {
		TestImage image.Image
		Expected  image.Point
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
		actual := smartResize(b, DefaultFactor, DefaultMinPixels, DefaultMaxPixels)
		if actual != c.Expected {
			t.Errorf("expected: %v, actual: %v", c.Expected, actual)
		}
	}
}

func TestPreprocess(t *testing.T) {
	type preprocessCase struct {
		TestImage   image.Image
		ExpectedLen int
	}

	cases := []preprocessCase{
		{
			TestImage:   image.NewRGBA(image.Rect(0, 0, 256, 256)),
			ExpectedLen: 252 * 252 * 3 * 1,
		},
		{
			TestImage:   image.NewRGBA(image.Rect(0, 0, 2000, 2000)),
			ExpectedLen: 980 * 980 * 3 * 1,
		},
	}

	for _, c := range cases {
		var buf bytes.Buffer
		err := png.Encode(&buf, c.TestImage)
		if err != nil {
			t.Fatal(err)
		}

		imgData, _, err := Preprocess(&buf)
		if err != nil {
			t.Fatalf("error processing: %q", err)
		}

		switch len(imgData) {
		case 0:
			t.Errorf("no image data returned")
		case c.ExpectedLen:
			// ok
		default:
			t.Errorf("unexpected image data length: %d, expected: %d", len(imgData), c.ExpectedLen)
		}
	}
}
