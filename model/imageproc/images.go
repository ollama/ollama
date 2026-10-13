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

func Composite(img image.Image) image.Image {
	white := color.RGBA{255, 255, 255, 255}
	return CompositeColor(img, white)
}

func CompositeColor(img image.Image, color color.Color) image.Image {
	dst := image.NewRGBA(img.Bounds())
	draw.Draw(dst, dst.Bounds(), &image.Uniform{color}, image.Point{}, draw.Src)
	draw.Draw(dst, dst.Bounds(), img, img.Bounds().Min, draw.Over)
	return dst
}

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
