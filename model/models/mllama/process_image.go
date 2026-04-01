package mllama

import (
	"image"
	"math"
	"slices"

	"golang.org/x/image/draw"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/model/imageproc"
)

type supportedAspectRatio struct {
	rank, width, height int
}

func (a supportedAspectRatio) Point() image.Point {
	return image.Point{a.width, a.height}
}

func (a supportedAspectRatio) numTiles() int {
	return a.width * a.height
}

type ImageProcessor struct {
	imageSize, numChannels, maxNumTiles int

	mean, std [3]float32
}

func newImageProcessor(c fs.Config) ImageProcessor {
	return ImageProcessor{
		imageSize:   int(c.Uint("vision.image_size")),
		numChannels: int(c.Uint("vision.num_channels")),
		maxNumTiles: int(c.Uint("vision.max_num_tiles")),

		mean: imageproc.ClipDefaultMean,
		std:  imageproc.ClipDefaultSTD,
	}
}

func (p ImageProcessor) supportedAspectRatios() (ratios []supportedAspectRatio) {
	for w := 1; w <= p.maxNumTiles; w++ {
		for h := 1; h <= p.maxNumTiles/w; h++ {
			ratios = append(ratios, supportedAspectRatio{len(ratios) + 1, w, h})
		}
	}
	return ratios
}

func (p ImageProcessor) fitToCanvas(imageSize, canvasSize image.Point) image.Point {
	tw := min(max(imageSize.X, p.imageSize), canvasSize.X)
	th := min(max(imageSize.Y, p.imageSize), canvasSize.Y)

	r := min(
		float64(tw)/float64(imageSize.X),
		float64(th)/float64(imageSize.Y),
	)

	w := min(int(math.Floor(float64(imageSize.X)*r)), tw)
	h := min(int(math.Floor(float64(imageSize.Y)*r)), th)

	return image.Point{w, h}
}

func (p ImageProcessor) optimalTiledCanvas(imageSize image.Point) image.Point {
	possibleTileArrangements := p.supportedAspectRatios()
	possibleCanvasSizes := make([]image.Point, len(possibleTileArrangements))
	for i, pta := range possibleTileArrangements {
		possibleCanvasSizes[i] = image.Point{pta.width * p.imageSize, pta.height * p.imageSize}
	}

	scales := make([]float64, len(possibleCanvasSizes))
	for i, pcs := range possibleCanvasSizes {
		scales[i] = min(
			float64(pcs.Y)/float64(imageSize.Y),
			float64(pcs.X)/float64(imageSize.X),
		)
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
				minUpscale = min(minUpscale, s)
			}
		} else {
			maxDownscale = max(maxDownscale, s)
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

func (p ImageProcessor) splitToTiles(img image.Image, numTilesSize image.Point) []image.Image {
	b := img.Bounds()
	width := b.Max.X - b.Min.X
	height := b.Max.Y - b.Min.Y
	tileHeight := height / numTilesSize.Y
	tileWidth := width / numTilesSize.X

	images := make([]image.Image, 0, numTilesSize.Y*numTilesSize.X)

	for h := range numTilesSize.Y {
		for w := range numTilesSize.X {
			rect := image.Rect(tileWidth*w, tileHeight*h, tileWidth*(w+1), tileHeight*(h+1))
			if subImg, ok := img.(interface {
				SubImage(image.Rectangle) image.Image
			}); ok {
				images = append(images, subImg.SubImage(rect))
			} else {
				// Handle the case where img does not implement SubImage
				// This is a fallback and may not be efficient
				newImg := image.NewRGBA(rect)
				draw.Draw(newImg, rect, img, rect.Min, draw.Src)
				images = append(images, newImg)
			}
		}
	}

	return images
}

func (p ImageProcessor) resize(img image.Image) (image.Image, image.Point) {
	b := img.Bounds()

	canvasSize := p.optimalTiledCanvas(b.Max)
	aspectRatio := image.Point{canvasSize.X / p.imageSize, canvasSize.Y / p.imageSize}
	newSize := p.fitToCanvas(b.Max, canvasSize)

	dst := image.NewRGBA(image.Rect(0, 0, newSize.X, newSize.Y))

	// scaling choices:
	//   NearestNeighbor	fast, blocky output
	//   ApproxBiLinear	fast, medium quality
	//   BiLinear		slow, high quality
	//   CatmullRom		very slow, very high quality
	draw.BiLinear.Scale(dst, dst.Rect, img, b, draw.Over, nil)

	return dst, aspectRatio
}

func (p ImageProcessor) pad(img image.Image, aspectRatio image.Point) image.Image {
	paddedSize := image.Point{
		X: p.imageSize * aspectRatio.X,
		Y: p.imageSize * aspectRatio.Y,
	}

	dst := image.NewRGBA(image.Rect(0, 0, paddedSize.X, paddedSize.Y))
	draw.Draw(dst, img.Bounds(), img, image.Point{0, 0}, draw.Over)

	return dst
}

func (p ImageProcessor) pack(img image.Image, aspectRatio image.Point) []float32 {
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

				rVal = (rVal - p.mean[0]) / p.std[0]
				gVal = (gVal - p.mean[1]) / p.std[1]
				bVal = (bVal - p.mean[2]) / p.std[2]

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

func (p ImageProcessor) ProcessImage(img image.Image) ([]float32, supportedAspectRatio, error) {
	newImage, newImageRatio := p.resize(img)
	newImage = p.pad(newImage, newImageRatio)
	pixelValues := p.pack(newImage, newImageRatio)

	supportedAspectRatios := p.supportedAspectRatios()
	aspectRatioID := slices.IndexFunc(supportedAspectRatios, func(i supportedAspectRatio) bool {
		return i.width == newImageRatio.X && i.height == newImageRatio.Y
	})

	return pixelValues, supportedAspectRatios[aspectRatioID], nil
}
