package mllama

import (
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"math"
	"slices"

	"golang.org/x/image/draw"

	"github.com/ollama/ollama/model/imageproc"
)

func getSupportedAspectRatios(maxTiles int) []image.Point {
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

func clip(a, a_min, a_max int) int {
	if a < a_min {
		return a_min
	} else if a > a_max {
		return a_max
	}

	return a
}

func getOptimalTiledCanvas(imageSize image.Point, maxImageTiles, tileSize int) image.Point {
	possibleTileArrangements := getSupportedAspectRatios(maxImageTiles)
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

func getImageSizeFitToCanvas(imageSize, canvasSize image.Point, tileSize int) image.Point {
	targetWidth := clip(imageSize.X, tileSize, canvasSize.X)
	targetHeight := clip(imageSize.Y, tileSize, canvasSize.Y)

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

func resizeImage(img image.Image, format string, outputSize image.Point, maxImageTiles int) (image.Image, image.Point) {
	if format == "png" {
		img = imageproc.Composite(img)
	}

	b := img.Bounds()
	tileSize := outputSize.Y

	canvasSize := getOptimalTiledCanvas(b.Max, maxImageTiles, tileSize)
	aspectRatio := image.Point{canvasSize.X / tileSize, canvasSize.Y / tileSize}
	newSize := getImageSizeFitToCanvas(b.Max, canvasSize, tileSize)

	return imageproc.Resize(img, newSize, imageproc.ResizeBilinear), aspectRatio
}

func padImage(img image.Image, outputSize, aspectRatio image.Point) image.Image {
	paddedSize := image.Point{
		X: outputSize.X * aspectRatio.X,
		Y: outputSize.Y * aspectRatio.Y,
	}

	dst := image.NewRGBA(image.Rect(0, 0, paddedSize.X, paddedSize.Y))
	draw.Draw(dst, img.Bounds(), img, image.Point{0, 0}, draw.Over)

	return dst
}

func splitToTiles(img image.Image, numTilesSize image.Point) []image.Image {
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

func packImages(img image.Image, aspectRatio image.Point) []float32 {
	subImages := splitToTiles(img, aspectRatio)

	var pixelVals []float32

	rescale := true
	channelFirst := true

	for _, subImg := range subImages {
		vals := imageproc.Normalize(subImg, imageproc.ClipDefaultMean, imageproc.ClipDefaultSTD, rescale, channelFirst)
		pixelVals = append(pixelVals, vals...)
	}

	return pixelVals
}

func Preprocess(imageData io.Reader) ([]float32, map[string]any, error) {
	outputSize := image.Point{560, 560}
	maxTiles := 4

	img, format, err := image.Decode(imageData)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to decode image: %w", err)
	}

	newImage, aspectRatio := resizeImage(img, format, outputSize, maxTiles)
	newImage = padImage(newImage, outputSize, aspectRatio)

	data := packImages(newImage, aspectRatio)
	aspectRatioIndex := slices.Index(getSupportedAspectRatios(maxTiles), aspectRatio) + 1

	opts := map[string]any{
		"aspectRatioIndex": aspectRatioIndex,
	}

	return data, opts, nil
}
