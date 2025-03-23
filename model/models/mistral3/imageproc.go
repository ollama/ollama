package mistral3

import (
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"math"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/imageproc"
)

func getNumImageTokens(imageSize, patchSize image.Point) image.Point {
	return image.Point{
		(imageSize.X-1)/patchSize.X + 1,
		(imageSize.Y-1)/patchSize.Y + 1,
	}
}

func getResizeOutputImageSize(img image.Image, longestEdge int, patchSize image.Point) image.Point {
	b := img.Bounds()
	ratio := math.Max(float64(b.Max.Y)/float64(longestEdge), float64(b.Max.X)/float64(longestEdge))

	newSize := img.Bounds().Max

	if ratio > 1.0 {
		newSize = image.Point{
			int(math.Floor(float64(b.Max.X) / ratio)),
			int(math.Floor(float64(b.Max.Y) / ratio)),
		}
	}

	tokens := getNumImageTokens(newSize, patchSize)
	return image.Point{
		tokens.X * patchSize.X,
		tokens.Y * patchSize.Y,
	}
}

func resizeImage(img image.Image, format string, longestEdge int, patchSize image.Point) image.Image {
	if format == "png" {
		img = imageproc.Composite(img)
	}

	newSize := getResizeOutputImageSize(img, longestEdge, patchSize)

	// todo should be ResizeBicubic, but it doesn't exist
	return imageproc.Resize(img, newSize, imageproc.ResizeBilinear)
}

func Preprocess(imageData io.Reader) ([]float32, map[string]any, error) {
	img, format, err := image.Decode(imageData)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to decode image: %w", err)
	}

	longestEdge := 1024
	patchSize := image.Point{16, 16}

	img = resizeImage(img, format, longestEdge, patchSize)

	data := imageproc.Normalize(img, imageproc.ClipDefaultMean, imageproc.ClipDefaultSTD, true, true)

	opts := map[string]any{}
	return data, opts, nil
}

type ImageProcessor struct {
	imageSize   int
	patchSize   int
	numChannels int
	longestEdge int
}

func newImageProcessor(c ml.Config) ImageProcessor {
	return ImageProcessor{
		imageSize:   int(c.Uint("vision.image_size", 1540)),
		patchSize:   int(c.Uint("vision.patch_size", 14)),
		numChannels: int(c.Uint("vision.num_channels", 3)),
		longestEdge: int(c.Uint("vision.longest_edge", 1540)),
	}
}

func (p *ImageProcessor) ProcessImage(img image.Image) ([]float32, error) {
	outputSize := getResizeOutputImageSize(img, p.longestEdge, image.Point{p.patchSize, p.patchSize})
	newImage := imageproc.Composite(img)
	newImage = imageproc.Resize(newImage, outputSize, imageproc.ResizeBilinear)
	data := imageproc.Normalize(newImage, imageproc.ClipDefaultMean, imageproc.ClipDefaultSTD, true, true)
	return data, nil
}
