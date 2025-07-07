package gemma3

import (
	"bytes"
	"fmt"
	"image"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/model/imageproc"
)

type ImageProcessor struct {
	imageSize   int
	patchSize   int
	numChannels int
	mean        [3]float32
	stdDev      [3]float32
}

func newImageProcessor(c fs.Config) ImageProcessor {
	return ImageProcessor{
		imageSize:   int(c.Uint("vision.image_size", 448)),
		patchSize:   int(c.Uint("vision.patch_size", 16)),
		numChannels: 3,
		mean:        [3]float32{0.48145466, 0.4578275, 0.40821073},
		stdDev:      [3]float32{0.26862954, 0.26130258, 0.27577711},
	}
}

func (p *ImageProcessor) pack(img image.Image, mean, std [3]float32) []float32 {
	var pixelVals, rVals, gVals, bVals []float32

	bounds := img.Bounds()
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			c := img.At(x, y)
			r, g, b, _ := c.RGBA()
			rVal := float32(r>>8) / 255.0
			gVal := float32(g>>8) / 255.0
			bVal := float32(b>>8) / 255.0

			rVal = (rVal - mean[0]) / std[0]
			gVal = (gVal - mean[1]) / std[1]
			bVal = (bVal - mean[2]) / std[2]

			rVals = append(rVals, rVal)
			gVals = append(gVals, gVal)
			bVals = append(bVals, bVal)
		}
	}

	pixelVals = append(pixelVals, rVals...)
	pixelVals = append(pixelVals, gVals...)
	pixelVals = append(pixelVals, bVals...)

	return pixelVals
}

// ProcessImage processes image data for model consumption
func (p *ImageProcessor) ProcessImage(data []byte) ([]float32, error) {
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("failed to decode image: %w", err)
	}

	outputSize := image.Point{p.imageSize, p.imageSize}
	newImage := imageproc.Composite(img)
	newImage = imageproc.Resize(newImage, outputSize, imageproc.ResizeBilinear)

	return p.pack(newImage, p.mean, p.stdDev), nil
}

// ProcessImageWithSize processes image data and returns the original image dimensions
func (p *ImageProcessor) ProcessImageWithSize(data []byte) ([]float32, image.Point, error) {
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, image.Point{}, err
	}

	// Get original image dimensions
	bounds := img.Bounds()
	size := image.Point{bounds.Dx(), bounds.Dy()}

	outputSize := image.Point{p.imageSize, p.imageSize}
	newImage := imageproc.Composite(img)
	newImage = imageproc.Resize(newImage, outputSize, imageproc.ResizeBilinear)

	return p.pack(newImage, p.mean, p.stdDev), size, nil
}
