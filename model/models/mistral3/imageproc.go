package mistral3

import (
	"image"
	_ "image/jpeg"
	_ "image/png"
	"math"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/model/imageproc"
)

type ImageProcessor struct {
	imageSize   int
	patchSize   int
	numChannels int
	longestEdge int
}

func newImageProcessor(c fs.Config) ImageProcessor {
	return ImageProcessor{
		imageSize:   int(c.Uint("vision.image_size", 1540)),
		patchSize:   int(c.Uint("vision.patch_size", 14)),
		numChannels: int(c.Uint("vision.num_channels", 3)),
		longestEdge: int(c.Uint("vision.longest_edge", 1540)),
	}
}

// ProcessImage prepares an image for the vision model by:
// 1. Compositing transparent images
// 2. Resizing to fit model constraints while preserving aspect ratio
// 3. Normalizing pixel values
// Returns normalized image data and the final size in pixels
func (p *ImageProcessor) ProcessImage(img image.Image) ([]float32, image.Point, error) {
	img = imageproc.Composite(img)

	size := img.Bounds().Size()
	ratio := max(float64(size.Y)/float64(p.longestEdge), float64(size.X)/float64(p.longestEdge))
	if ratio > 1.0 {
		size = image.Point{
			int(math.Floor(float64(size.X) / ratio)),
			int(math.Floor(float64(size.Y) / ratio)),
		}
	}

	patchesX := (size.X-1)/p.patchSize + 1
	patchesY := (size.Y-1)/p.patchSize + 1
	size = image.Point{
		patchesX * p.patchSize,
		patchesY * p.patchSize,
	}

	img = imageproc.Resize(img, size, imageproc.ResizeBilinear)
	data := imageproc.Normalize(img, imageproc.ClipDefaultMean, imageproc.ClipDefaultSTD, true, true)
	return data, size, nil
}
