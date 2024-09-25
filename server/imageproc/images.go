package imageproc

import (
	"bytes"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"math"

	"golang.org/x/image/draw"
)

func GetSupportedAspectRatios(maxTiles int) []image.Point {
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

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func GetImageSizeFitToCanvas(imageSize, canvasSize image.Point, tileSize int) image.Point {
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

func GetOptimalTiledCanvas(imageSize image.Point, maxImageTiles, tileSize int) image.Point {
	possibleTileArrangements := GetSupportedAspectRatios(maxImageTiles)
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

	selectedCanvas := possibleCanvasSizes[0]
	for n, pcs := range possibleCanvasSizes {
		if scales[n] == selectedScale {
			// choose the largest possible canvas
			if pcs.X*pcs.Y > selectedCanvas.X*selectedCanvas.Y {
				selectedCanvas = pcs
			}
		}
	}
	return selectedCanvas
}

func SplitToTiles(img image.Image, numTilesSize image.Point) []image.Image {
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

func ResizeImage(img image.Image, outputSize image.Point, maxImageTiles int) (image.Image, image.Point) {
	b := img.Bounds()
	tileSize := outputSize.Y

	canvasSize := GetOptimalTiledCanvas(b.Max, maxImageTiles, tileSize)
	aspectRatio := image.Point{canvasSize.X / tileSize, canvasSize.Y / tileSize}

	newSize := GetImageSizeFitToCanvas(b.Max, canvasSize, tileSize)

	dst := image.NewRGBA(image.Rect(0, 0, newSize.X, newSize.Y))
	draw.ApproxBiLinear.Scale(dst, dst.Rect, img, b, draw.Over, nil)

	return dst, aspectRatio
}

func PadImage(img image.Image, outputSize, aspectRatio image.Point) image.Image {
	paddedSize := image.Point{
		X: outputSize.X * aspectRatio.X,
		Y: outputSize.Y * aspectRatio.Y,
	}

	dst := image.NewRGBA(image.Rect(0, 0, paddedSize.X, paddedSize.Y))
	centerX := (paddedSize.X - img.Bounds().Max.X) / 2
	centerY := (paddedSize.Y - img.Bounds().Max.Y) / 2
	pos := image.Rect(centerX, centerY, centerX+img.Bounds().Max.X, centerY+img.Bounds().Max.Y)

	draw.Draw(dst, pos, img, image.Point{0, 0}, draw.Over)

	return dst
}

func PackImages(img image.Image, aspectRatio image.Point, mean, std [3]float32) []float32 {
	subImages := SplitToTiles(img, aspectRatio)

	var pixelVals []float32

	for _, subImg := range subImages {
		bounds := subImg.Bounds()
		rVals := []float32{}
		gVals := []float32{}
		bVals := []float32{}
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

func Preprocess(imageData []byte) ([]float32, int, error) {
	// todo: need guard in here for bad image data

	// mllama values
	outputSize := image.Point{560, 560}
	maxTiles := 4

	// clip values
	mean := [3]float32{0.48145466, 0.4578275, 0.40821073}
	std := [3]float32{0.26862954, 0.26130258, 0.27577711}

	img, _, err := image.Decode(bytes.NewReader(imageData))
	if err != nil {
		return nil, 0, fmt.Errorf("failed to decode image: %w", err)
	}

	newImage, aspectRatio := ResizeImage(img, outputSize, maxTiles)
	newImage = PadImage(newImage, outputSize, aspectRatio)

	// todo: need to scale (dim) by 1/256

	data := PackImages(newImage, aspectRatio, mean, std)
	supportedRatios := GetSupportedAspectRatios(maxTiles)
	var aspectRatioIndex int
	for n, r := range supportedRatios {
		if r == aspectRatio {
			aspectRatioIndex = n + 1
			break
		}
	}

	return data, aspectRatioIndex, nil
}
