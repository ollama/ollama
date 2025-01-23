package imageproc

import (
	"image"
	"image/color"

	"golang.org/x/image/draw"
)

var (
	ImageNetDefaultMean  = [3]float32{0.485, 0.456, 0.406}
	ImageNetDefaultSTD   = [3]float32{0.229, 0.224, 0.225}
	ImageNetStandardMean = [3]float32{0.5, 0.5, 0.5}
	ImageNetStandardSTD  = [3]float32{0.5, 0.5, 0.5}
	ClipDefaultMean      = [3]float32{0.48145466, 0.4578275, 0.40821073}
	ClipDefaultSTD       = [3]float32{0.26862954, 0.26130258, 0.27577711}
)

const (
	ResizeBilinear = iota
	ResizeNearestNeighbor
	ResizeApproxBilinear
	ResizeCatmullrom
)

// Composite returns an image with the alpha channel removed by drawing over a white background.
func Composite(img image.Image) image.Image {
	dst := image.NewRGBA(img.Bounds())

	white := color.RGBA{255, 255, 255, 255}
	draw.Draw(dst, dst.Bounds(), &image.Uniform{white}, image.Point{}, draw.Src)
	draw.Draw(dst, dst.Bounds(), img, img.Bounds().Min, draw.Over)

	return dst
}

// Resize returns an image which has been scaled to a new size.
func Resize(img image.Image, newSize image.Point, method int) image.Image {
	dst := image.NewRGBA(image.Rect(0, 0, newSize.X, newSize.Y))

	kernels := map[int]draw.Interpolator{
		ResizeBilinear:        draw.BiLinear,
		ResizeNearestNeighbor: draw.NearestNeighbor,
		ResizeApproxBilinear:  draw.ApproxBiLinear,
		ResizeCatmullrom:      draw.CatmullRom,
	}

	kernel, ok := kernels[method]
	if !ok {
		panic("no resizing method found")
	}

	kernel.Scale(dst, dst.Rect, img, img.Bounds(), draw.Over, nil)

	return dst
}

// Normalize returns a slice of float32 containing each of the r, g, b values for an image normalized around a value.
func Normalize(img image.Image, mean, std [3]float32, rescale bool, channelFirst bool) []float32 {
	var pixelVals []float32

	bounds := img.Bounds()
	if channelFirst {
		var rVals, gVals, bVals []float32
		for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
			for x := bounds.Min.X; x < bounds.Max.X; x++ {
				c := img.At(x, y)
				r, g, b, _ := c.RGBA()
				var rVal, gVal, bVal float32
				if rescale {
					rVal = float32(r>>8) / 255.0
					gVal = float32(g>>8) / 255.0
					bVal = float32(b>>8) / 255.0
				}

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
	} else {
		for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
			for x := bounds.Min.X; x < bounds.Max.X; x++ {
				c := img.At(x, y)
				r, g, b, _ := c.RGBA()
				var rVal, gVal, bVal float32
				if rescale {
					rVal = float32(r>>8) / 255.0
					gVal = float32(g>>8) / 255.0
					bVal = float32(b>>8) / 255.0
				}

				rVal = (rVal - mean[0]) / std[0]
				gVal = (gVal - mean[1]) / std[1]
				bVal = (bVal - mean[2]) / std[2]

				pixelVals = append(pixelVals, rVal, gVal, bVal)
			}
		}
	}

	return pixelVals
}
