//go:build mlx

package gemma3

import (
	"image"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/model/imageproc"
)

type ImageProcessor struct {
	imageSize, patchSize, numChannels int
}

func newImageProcessor(c fs.Config) ImageProcessor {
	return ImageProcessor{
		imageSize:   int(c.Uint("vision.image_size")),
		patchSize:   int(c.Uint("vision.patch_size")),
		numChannels: int(c.Uint("vision.num_channels")),
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

func (p ImageProcessor) ProcessImage(img image.Image) ([]float32, error) {
	outputSize := image.Point{p.imageSize, p.imageSize}
	newImage := imageproc.Composite(img)
	newImage = imageproc.Resize(newImage, outputSize, imageproc.ResizeBilinear)

	data := p.pack(newImage, imageproc.ImageNetStandardMean, imageproc.ImageNetStandardSTD)
	return data, nil
}
