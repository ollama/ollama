package lfm2

import (
	"image"
	stdimage "image/draw"
	"math"
	"slices"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/model/imageproc"
)

type ImageProcessor struct {
	imageSize, patchSize, numChannels int
	downsampleFactor                  int
	imageMean, imageStd               [3]float32

	doImageSplitting bool
	minTiles         int
	maxTiles         int
	useThumbnail     bool
	tileSize         int

	minImageTokens     int
	maxImageTokens     int
	maxPixelsTolerance float64
}

type processedVisionImage struct {
	data      []float32
	size      image.Point
	row       int
	col       int
	thumbnail bool
}

type processedVisionLayout struct {
	rows         int
	cols         int
	hasThumbnail bool
}

func newImageProcessor(c fs.Config) ImageProcessor {
	mean := c.Floats("vision.image_mean")
	std := c.Floats("vision.image_std")

	processor := ImageProcessor{
		imageSize:          int(c.Uint("vision.image_size", 256)),
		patchSize:          int(c.Uint("vision.patch_size", 16)),
		numChannels:        int(c.Uint("vision.num_channels", 3)),
		downsampleFactor:   int(c.Uint("vision.projector.scale_factor", 2)),
		imageMean:          [3]float32{0.5, 0.5, 0.5},
		imageStd:           [3]float32{0.5, 0.5, 0.5},
		doImageSplitting:   c.Bool("vision.do_image_splitting", true),
		minTiles:           int(c.Uint("vision.min_tiles", 2)),
		maxTiles:           int(c.Uint("vision.max_tiles", 10)),
		useThumbnail:       c.Bool("vision.use_thumbnail", true),
		tileSize:           int(c.Uint("vision.tile_size", 512)),
		minImageTokens:     int(c.Uint("vision.min_image_tokens", 64)),
		maxImageTokens:     int(c.Uint("vision.max_image_tokens", 256)),
		maxPixelsTolerance: float64(c.Float("vision.max_pixels_tolerance", 2.0)),
	}

	if len(mean) >= 3 {
		processor.imageMean = [3]float32{mean[0], mean[1], mean[2]}
	}
	if len(std) >= 3 {
		processor.imageStd = [3]float32{std[0], std[1], std[2]}
	}

	// Keep defaults aligned with HF unless explicitly configured.
	if processor.downsampleFactor <= 0 {
		processor.downsampleFactor = 2
	}
	if processor.patchSize <= 0 {
		processor.patchSize = 16
	}
	if processor.tileSize <= 0 {
		processor.tileSize = 512
	}
	if processor.minTiles <= 0 {
		processor.minTiles = 2
	}
	if processor.maxTiles < processor.minTiles {
		processor.maxTiles = processor.minTiles
	}
	if processor.minImageTokens <= 0 {
		processor.minImageTokens = 64
	}
	if processor.maxImageTokens < processor.minImageTokens {
		processor.maxImageTokens = processor.minImageTokens
	}
	if processor.maxPixelsTolerance <= 0 {
		processor.maxPixelsTolerance = 2.0
	}

	return processor
}

func (p ImageProcessor) ProcessImage(img image.Image) ([]processedVisionImage, processedVisionLayout, error) {
	img = imageproc.Composite(img)

	orig := img.Bounds().Size()
	resizedWidth, resizedHeight := p.smartResize(orig.Y, orig.X)

	layout := processedVisionLayout{rows: 1, cols: 1}
	if p.shouldSplit(orig.Y, orig.X) {
		gridWidth, gridHeight, targetWidth, targetHeight := p.gridLayout(orig.Y, orig.X)
		layout.rows = gridHeight
		layout.cols = gridWidth
		layout.hasThumbnail = p.useThumbnail && gridWidth*gridHeight != 1

		resized := imageproc.Resize(img, image.Point{X: targetWidth, Y: targetHeight}, imageproc.ResizeBilinear)
		images := make([]processedVisionImage, 0, gridWidth*gridHeight+1)
		for row := 0; row < gridHeight; row++ {
			for col := 0; col < gridWidth; col++ {
				rect := image.Rect(
					col*p.tileSize,
					row*p.tileSize,
					(col+1)*p.tileSize,
					(row+1)*p.tileSize,
				)
				tile := cropImage(resized, rect)
				images = append(images, processedVisionImage{
					data: imageproc.Normalize(tile, p.imageMean, p.imageStd, true, true),
					size: tile.Bounds().Size(),
					row:  row + 1,
					col:  col + 1,
				})
			}
		}

		if layout.hasThumbnail {
			thumbnail := imageproc.Resize(img, image.Point{X: resizedWidth, Y: resizedHeight}, imageproc.ResizeBilinear)
			images = append(images, processedVisionImage{
				data:      imageproc.Normalize(thumbnail, p.imageMean, p.imageStd, true, true),
				size:      thumbnail.Bounds().Size(),
				thumbnail: true,
			})
		}

		return images, layout, nil
	}

	single := imageproc.Resize(img, image.Point{X: resizedWidth, Y: resizedHeight}, imageproc.ResizeBilinear)
	return []processedVisionImage{{
		data: imageproc.Normalize(single, p.imageMean, p.imageStd, true, true),
		size: single.Bounds().Size(),
	}}, layout, nil
}

func (p ImageProcessor) shouldSplit(height, width int) bool {
	if !p.doImageSplitting || p.minTiles == 1 && p.maxTiles == 1 {
		return false
	}

	totalFactor := p.patchSize * p.downsampleFactor
	hBar := max(p.patchSize, roundByFactor(height, totalFactor))
	wBar := max(p.patchSize, roundByFactor(width, totalFactor))

	limit := float64(p.maxImageTokens * p.patchSize * p.patchSize * p.downsampleFactor * p.downsampleFactor)
	limit *= p.maxPixelsTolerance

	return float64(hBar*wBar) > limit
}

func (p ImageProcessor) smartResize(height, width int) (int, int) {
	totalFactor := p.patchSize * p.downsampleFactor
	minPixels := p.minImageTokens * p.patchSize * p.patchSize * p.downsampleFactor * p.downsampleFactor
	maxPixels := p.maxImageTokens * p.patchSize * p.patchSize * p.downsampleFactor * p.downsampleFactor

	hBar := max(totalFactor, roundByFactor(height, totalFactor))
	wBar := max(totalFactor, roundByFactor(width, totalFactor))

	if hBar*wBar > maxPixels {
		beta := math.Sqrt(float64(height*width) / float64(maxPixels))
		hBar = max(totalFactor, int(math.Floor(float64(height)/beta/float64(totalFactor)))*totalFactor)
		wBar = max(totalFactor, int(math.Floor(float64(width)/beta/float64(totalFactor)))*totalFactor)
	} else if hBar*wBar < minPixels {
		beta := math.Sqrt(float64(minPixels) / float64(height*width))
		hBar = int(math.Ceil(float64(height)*beta/float64(totalFactor))) * totalFactor
		wBar = int(math.Ceil(float64(width)*beta/float64(totalFactor))) * totalFactor
	}

	return wBar, hBar
}

func (p ImageProcessor) gridLayout(height, width int) (gridWidth, gridHeight, targetWidth, targetHeight int) {
	aspectRatio := float64(width) / float64(height)
	targetRatios := p.targetRatios()
	bestRatio := clipImageSize{width: 1, height: 1}
	bestRatioDiff := math.MaxFloat64
	area := float64(width * height)

	for _, ratio := range targetRatios {
		targetAspect := float64(ratio.width) / float64(ratio.height)
		ratioDiff := math.Abs(aspectRatio - targetAspect)

		if ratioDiff < bestRatioDiff {
			bestRatioDiff = ratioDiff
			bestRatio = ratio
			continue
		}

		if ratioDiff == bestRatioDiff {
			targetArea := float64(p.tileSize * p.tileSize * ratio.width * ratio.height)
			if area > 0.5*targetArea {
				bestRatio = ratio
			}
		}
	}

	return bestRatio.width, bestRatio.height, p.tileSize * bestRatio.width, p.tileSize * bestRatio.height
}

type clipImageSize struct {
	width  int
	height int
}

func (p ImageProcessor) targetRatios() []clipImageSize {
	targetRatios := make([]clipImageSize, 0, p.maxTiles*p.maxTiles)
	for n := p.minTiles; n <= p.maxTiles; n++ {
		for w := 1; w <= n; w++ {
			for h := 1; h <= n; h++ {
				if w*h < p.minTiles || w*h > p.maxTiles {
					continue
				}
				targetRatios = append(targetRatios, clipImageSize{width: w, height: h})
			}
		}
	}

	unique := targetRatios[:0]
	for _, ratio := range targetRatios {
		if slices.Contains(unique, ratio) {
			continue
		}
		unique = append(unique, ratio)
	}

	slices.SortFunc(unique, func(a, b clipImageSize) int {
		return a.width*a.height - b.width*b.height
	})

	return unique
}

func roundByFactor(number, factor int) int {
	if factor <= 0 {
		return number
	}
	return int(math.RoundToEven(float64(number)/float64(factor))) * factor
}

func cropImage(img image.Image, rect image.Rectangle) image.Image {
	dst := image.NewRGBA(image.Rect(0, 0, rect.Dx(), rect.Dy()))
	stdimage.Draw(dst, dst.Bounds(), img, rect.Min, stdimage.Src)
	return dst
}
