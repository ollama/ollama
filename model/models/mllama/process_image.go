package mllama

import (
	"image"
	"image/color"
	"math"
	"slices"

	"golang.org/x/image/draw"

	"github.com/ollama/ollama/fs"
)

type ImageProcessor struct {
	imageSize, numChannels, maxNumTiles int
}

func newImageProcessor(c fs.Config) ImageProcessor {
	return ImageProcessor{
		imageSize:   int(c.Uint("vision.image_size")),
		numChannels: int(c.Uint("vision.num_channels")),
		maxNumTiles: int(c.Uint("vision.max_num_tiles")),
	}
}

func (p *ImageProcessor) supportedAspectRatios(maxTiles int) []image.Point {
	ratios := []image.Point{}

	for w := range maxTiles {
		for h := range maxTiles {
			if (w+1)*(h+1) <= maxTiles {
				ratios = append(ratios, image.Point{w + 1, h + 1})
			}
		}
	}

	return ratios
}

func (p *ImageProcessor) clip(a, a_min, a_max int) int {
	if a < a_min {
		return a_min
	} else if a > a_max {
		return a_max
	}

	return a
}

func (p *ImageProcessor) fitToCanvas(imageSize, canvasSize image.Point, tileSize int) image.Point {
	targetWidth := p.clip(imageSize.X, tileSize, canvasSize.X)
	targetHeight := p.clip(imageSize.Y, tileSize, canvasSize.Y)

	scaleWidth := float64(targetWidth) / float64(imageSize.X)
	scaleHeight := float64(targetHeight) / float64(imageSize.Y)

	var w, h int

	if scaleWidth < scaleHeight {
		w = targetWidth
		h = min(int(math.Floor(float64(imageSize.Y)*scaleWidth)), targetHeight)
	} else {
		w = min(int(math.Floor(float64(imageSize.X)*scaleHeight)), targetWidth)
		h = targetHeight
	}

	return image.Point{w, h}
}

func (p *ImageProcessor) optimalTiledCanvas(imageSize image.Point, maxImageTiles, tileSize int) image.Point {
	possibleTileArrangements := p.supportedAspectRatios(maxImageTiles)
	possibleCanvasSizes := []image.Point{}
	for _, pta := range possibleTileArrangements {
		possibleCanvasSizes = append(possibleCanvasSizes, image.Point{pta.X * tileSize, pta.Y * tileSize})
	}

	scales := []float64{}

	for _, pcs := range possibleCanvasSizes {
		scaleHeight := float64(pcs.Y) / float64(imageSize.Y)
		scaleWidth := float64(pcs.X) / float64(imageSize.X)

		if scaleWidth > scaleHeight {
			scales = append(scales, scaleHeight)
		} else {
			scales = append(scales, scaleWidth)
		}
	}

	var minUpscale float64
	var maxDownscale float64
	var upscale bool

	for _, s := range scales {
		if s > 1.0 {
			upscale = true
			if minUpscale == 0 {
				minUpscale = s
			} else {
				minUpscale = math.Min(minUpscale, s)
			}
		} else {
			maxDownscale = math.Max(maxDownscale, s)
		}
	}

	selectedScale := maxDownscale
	if upscale {
		selectedScale = minUpscale
	}

	var selectedCanvas image.Point
	for n, pcs := range possibleCanvasSizes {
		if scales[n] == selectedScale {
			// choose the smallest possible canvas
			if selectedCanvas.X == 0 && selectedCanvas.Y == 0 {
				selectedCanvas = pcs
			} else if pcs.X*pcs.Y < selectedCanvas.X*selectedCanvas.Y {
				selectedCanvas = pcs
			}
		}
	}
	return selectedCanvas
}

func (p *ImageProcessor) splitToTiles(img image.Image, numTilesSize image.Point) []image.Image {
	b := img.Bounds()
	width := b.Max.X - b.Min.X
	height := b.Max.Y - b.Min.Y
	tileHeight := height / numTilesSize.Y
	tileWidth := width / numTilesSize.X

	images := []image.Image{}

	for h := range numTilesSize.Y {
		for w := range numTilesSize.X {
			rect := image.Rect(tileWidth*w, tileHeight*h, tileWidth*(w+1), tileHeight*(h+1))
			images = append(images, img.(interface {
				SubImage(image.Rectangle) image.Image
			}).SubImage(rect))
		}
	}

	return images
}

// remove the "alpha" channel by drawing over a prefilled image
//
//nolint:unused
func (p *ImageProcessor) compositeImage(img image.Image) image.Image {
	dst := image.NewRGBA(img.Bounds())

	white := color.RGBA{255, 255, 255, 255}
	draw.Draw(dst, dst.Bounds(), &image.Uniform{white}, image.Point{}, draw.Src)
	draw.Draw(dst, dst.Bounds(), img, img.Bounds().Min, draw.Over)

	return dst
}

func (p *ImageProcessor) resize(img image.Image, outputSize image.Point, maxImageTiles int) (image.Image, image.Point) {
	b := img.Bounds()
	tileSize := outputSize.Y

	canvasSize := p.optimalTiledCanvas(b.Max, maxImageTiles, tileSize)
	aspectRatio := image.Point{canvasSize.X / tileSize, canvasSize.Y / tileSize}
	newSize := p.fitToCanvas(b.Max, canvasSize, tileSize)

	dst := image.NewRGBA(image.Rect(0, 0, newSize.X, newSize.Y))

	// scaling choices:
	//   NearestNeighbor	fast, blocky output
	//   ApproxBiLinear	fast, medium quality
	//   BiLinear		slow, high quality
	//   CatmullRom		very slow, very high quality
	draw.BiLinear.Scale(dst, dst.Rect, img, b, draw.Over, nil)

	return dst, aspectRatio
}

func (p *ImageProcessor) pad(img image.Image, outputSize, aspectRatio image.Point) image.Image {
	paddedSize := image.Point{
		X: outputSize.X * aspectRatio.X,
		Y: outputSize.Y * aspectRatio.Y,
	}

	dst := image.NewRGBA(image.Rect(0, 0, paddedSize.X, paddedSize.Y))
	draw.Draw(dst, img.Bounds(), img, image.Point{0, 0}, draw.Over)

	return dst
}

func (p *ImageProcessor) pack(img image.Image, aspectRatio image.Point, mean, std [3]float32) []float32 {
	subImages := p.splitToTiles(img, aspectRatio)

	var pixelVals []float32

	for _, subImg := range subImages {
		bounds := subImg.Bounds()
		var rVals, gVals, bVals []float32
		for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
			for x := bounds.Min.X; x < bounds.Max.X; x++ {
				c := subImg.At(x, y)
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
	}

	return pixelVals
}

func (p ImageProcessor) ProcessImage(img image.Image) ([]float32, int, error) {
	outputSize := image.Point{p.imageSize, p.imageSize}

	// clip values
	mean := [3]float32{0.48145466, 0.4578275, 0.40821073}
	std := [3]float32{0.26862954, 0.26130258, 0.27577711}

	newImage, aspectRatio := p.resize(img, outputSize, p.maxNumTiles)
	newImage = p.pad(newImage, outputSize, aspectRatio)

	data := p.pack(newImage, aspectRatio, mean, std)
	aspectRatioIndex := slices.Index(p.supportedAspectRatios(p.maxNumTiles), aspectRatio) + 1
	return data, aspectRatioIndex, nil
}
