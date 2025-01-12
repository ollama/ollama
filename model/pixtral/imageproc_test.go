package pixtral

import (
	"bytes"
	"encoding/binary"
	"image"
	"image/png"
	"math"
	"os"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestGetNumImageTokens(t *testing.T) {
	type numImageTokensCase struct {
		ImageSize image.Point
		PatchSize image.Point
		Expected  image.Point
	}

	cases := []numImageTokensCase{
		{
			ImageSize: image.Point{1024, 764},
			PatchSize: image.Point{16, 16},
			Expected:  image.Point{64, 48},
		},
		{
			ImageSize: image.Point{800, 600},
			PatchSize: image.Point{16, 16},
			Expected:  image.Point{50, 38},
		},
		{
			ImageSize: image.Point{640, 480},
			PatchSize: image.Point{16, 16},
			Expected:  image.Point{40, 30},
		},
		{
			ImageSize: image.Point{320, 200},
			PatchSize: image.Point{16, 16},
			Expected:  image.Point{20, 13},
		},
		{
			ImageSize: image.Point{1320, 200},
			PatchSize: image.Point{16, 16},
			Expected:  image.Point{83, 13},
		},
		{
			ImageSize: image.Point{2000, 200},
			PatchSize: image.Point{16, 16},
			Expected:  image.Point{125, 13},
		},
		{
			ImageSize: image.Point{10000, 200},
			PatchSize: image.Point{16, 16},
			Expected:  image.Point{625, 13},
		},
		{
			ImageSize: image.Point{1131, 577},
			PatchSize: image.Point{16, 16},
			Expected:  image.Point{71, 37},
		},
		{
			ImageSize: image.Point{16, 16},
			PatchSize: image.Point{16, 16},
			Expected:  image.Point{1, 1},
		},
	}

	for _, c := range cases {
		actual := getNumImageTokens(c.ImageSize, c.PatchSize)

		if diff := cmp.Diff(actual, c.Expected); diff != "" {
			t.Errorf("mismatch (-got +want):\n%s", diff)
		}
	}
}

func TestGetResizeOutputImageSize(t *testing.T) {
	type resizeCase struct {
		Image       image.Image
		LongestEdge int
		PatchSize   image.Point
		Expected    image.Point
	}

	cases := []resizeCase{
		{
			Image:       image.NewRGBA(image.Rect(0, 0, 1024, 768)),
			LongestEdge: 1024,
			PatchSize:   image.Point{16, 16},
			Expected:    image.Point{1024, 768},
		},
		{
			Image:       image.NewRGBA(image.Rect(0, 0, 1162, 690)),
			LongestEdge: 1024,
			PatchSize:   image.Point{16, 16},
			Expected:    image.Point{1024, 624},
		},
		{
			Image:       image.NewRGBA(image.Rect(0, 0, 300, 200)),
			LongestEdge: 1024,
			PatchSize:   image.Point{16, 16},
			Expected:    image.Point{304, 208},
		},
		{
			Image:       image.NewRGBA(image.Rect(0, 0, 1862, 522)),
			LongestEdge: 1024,
			PatchSize:   image.Point{16, 16},
			Expected:    image.Point{1024, 288},
		},
	}

	for _, c := range cases {
		actual := getResizeOutputImageSize(c.Image, c.LongestEdge, c.PatchSize)

		if diff := cmp.Diff(actual, c.Expected); diff != "" {
			t.Errorf("mismatch (-got +want):\n%s", diff)
		}
	}
}

func TestResize(t *testing.T) {
	type resizeCase struct {
		Image       image.Image
		LongestEdge int
		PatchSize   image.Point
		Expected    image.Image
	}

	cases := []resizeCase{
		{
			Image:       image.NewRGBA(image.Rect(0, 0, 1862, 522)),
			LongestEdge: 1024,
			PatchSize:   image.Point{16, 16},
			Expected:    image.NewRGBA(image.Rect(0, 0, 1024, 288)),
		},
		{
			Image:       image.NewRGBA(image.Rect(0, 0, 10, 10)),
			LongestEdge: 1024,
			PatchSize:   image.Point{16, 16},
			Expected:    image.NewRGBA(image.Rect(0, 0, 16, 16)),
		},
	}

	for _, c := range cases {
		actual := resizeImage(c.Image, "png", c.LongestEdge, c.PatchSize)

		if actual.Bounds() != c.Expected.Bounds() {
			t.Errorf("image size incorrect: '%#v': expected: '%#v'", actual.Bounds(), c.Expected.Bounds())
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
			TestImage:   image.NewRGBA(image.Rect(0, 0, 10, 10)),
			ExpectedLen: 16 * 16 * 3 * 1,
		},
		{
			TestImage:   image.NewRGBA(image.Rect(0, 0, 2000, 2000)),
			ExpectedLen: 1024 * 1024 * 3 * 1,
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

func TestPreprocessImages(t *testing.T) {
	for _, testFile := range []string{"flight.png", "sportsball.png"} {
		f, err := os.Open(testFile)
		if err != nil {
			t.Skipf("skipping test, no test image found at %s", testFile)
		}
		defer f.Close()

		imgData, _, err := Preprocess(f)
		if err != nil {
			t.Fatalf("error processing: %q", err)
		}

		byteData := make([]byte, len(imgData)*4) // float32 is 4 bytes
		for i, f := range imgData {
			binary.LittleEndian.PutUint32(byteData[i*4:], math.Float32bits(f))
		}

		outputPath := "processed_" + testFile + ".bin"
		err = os.WriteFile(outputPath, byteData, 0o644)
		if err != nil {
			t.Fatalf("error writing processed image: %q", err)
		}
	}
}
