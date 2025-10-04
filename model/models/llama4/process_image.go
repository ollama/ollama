package llama4

import (
	"cmp"
	"image"
	"math"
	"slices"
	"sort"

	"golang.org/x/image/draw"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/model/imageproc"
)

type ImageProcessor struct {
	imageSize, patchSize, numChannels, maxUpscalingSize int
}

func newImageProcessor(c fs.Config) ImageProcessor {
	return ImageProcessor{
		imageSize:        int(c.Uint("vision.image_size")),
		patchSize:        int(c.Uint("vision.patch_size")),
		numChannels:      int(c.Uint("vision.num_channels", 3)),
		maxUpscalingSize: int(c.Uint("vision.max_upscaling_size", 448)),
	}
}

func factors(n int) []int {
	var result []int
	seen := make(map[int]bool)

	for i := 1; i <= n/2; i++ {
		if n%i == 0 && !seen[i] {
			result = append(result, i)
			seen[i] = true
		}
	}

	result = append(result, n)
	sort.Ints(result)

	return result
}

func (p ImageProcessor) supportedResolutions() []image.Point {
	var resolutions []image.Point

	aspectMap := make(map[float64][]image.Point)
	for i := p.patchSize; i >= 1; i-- {
		for _, f := range factors(i) {
			x := f
			y := i / f
			k := float64(y) / float64(x)
			aspectMap[k] = append(aspectMap[k], image.Point{x, y})
		}
	}

	for _, v := range aspectMap {
		for _, i := range v {
			resolutions = append(resolutions, image.Point{i.X * p.imageSize, i.Y * p.imageSize})
		}
	}

	return resolutions
}

func (p ImageProcessor) bestResolution(img image.Point, possibleResolutions []image.Point, resizeToMaxCanvas bool) image.Point {
	w, h := img.X, img.Y

	scales := make([]float64, len(possibleResolutions))

	for i, res := range possibleResolutions {
		scaleW := float64(res.X) / float64(w)
		scaleH := float64(res.Y) / float64(h)
		scale := min(scaleW, scaleH)

		scales[i] = scale
	}

	minAboveOne := func(scales []float64) (float64, bool) {
		min := math.MaxFloat64
		found := false

		for _, s := range scales {
			if s >= 1.0 && s < min {
				min = s
				found = true
			}
		}

		return min, found
	}

	bestScale, ok := minAboveOne(scales)
	if resizeToMaxCanvas || !ok {
		bestScale = slices.Max(scales)
	}

	var bestOptions []image.Point
	for i, scale := range scales {
		if math.Abs(scale-bestScale) < 1e-6 {
			bestOptions = append(bestOptions, possibleResolutions[i])
		}
	}

	var chosenResolution image.Point
	if len(bestOptions) > 1 {
		chosenResolution = slices.MinFunc(bestOptions, func(a, b image.Point) int {
			return cmp.Compare(a.X*a.Y, b.X*b.Y)
		})
	} else {
		chosenResolution = bestOptions[0]
	}

	return chosenResolution
}

func (p ImageProcessor) maxResolution(imageRes, targetRes image.Point) image.Point {
	scaleW := float64(targetRes.X) / float64(imageRes.X)
	scaleH := float64(targetRes.Y) / float64(imageRes.Y)

	var newRes image.Point
	if scaleW < scaleH {
		newRes = image.Point{
			targetRes.X,
			int(min(math.Floor(float64(imageRes.Y)*scaleW), float64(targetRes.Y))),
		}
	} else {
		newRes = image.Point{
			int(min(math.Floor(float64(imageRes.X)*scaleH), float64(targetRes.X))),
			targetRes.Y,
		}
	}

	return newRes
}

func (p ImageProcessor) pad(src image.Image, outputSize image.Point) image.Image {
	dst := image.NewRGBA(image.Rect(0, 0, outputSize.X, outputSize.Y))
	draw.Draw(dst, src.Bounds(), src, image.Point{}, draw.Over)
	return dst
}

func (p ImageProcessor) ProcessImage(img image.Image) (pixelsLocal, pixelsGlobal []float32, targetSize image.Point, _ error) {
	img = imageproc.Composite(img)

	targetSize = p.bestResolution(img.Bounds().Max, p.supportedResolutions(), false)
	targetSizeWithoutDistortion := targetSize
	if p.maxUpscalingSize > 0 {
		targetSizeWithoutDistortion = p.maxResolution(img.Bounds().Max, targetSize)
		targetSizeWithoutDistortion.X = min(max(img.Bounds().Max.X, p.maxUpscalingSize), targetSize.X)
		targetSizeWithoutDistortion.Y = min(max(img.Bounds().Max.Y, p.maxUpscalingSize), targetSize.Y)
	}

	newSizeWithoutDistortion := p.maxResolution(img.Bounds().Max, targetSizeWithoutDistortion)

	padded := p.pad(imageproc.Resize(img, newSizeWithoutDistortion, imageproc.ResizeBilinear), targetSize)
	pixelsLocal = imageproc.Normalize(padded, imageproc.ImageNetStandardMean, imageproc.ImageNetStandardSTD, true, true)

	if targetSize.X/p.imageSize*targetSize.Y/p.imageSize > 1 {
		padded := imageproc.Resize(img, image.Point{p.imageSize, p.imageSize}, imageproc.ResizeBilinear)
		pixelsGlobal = imageproc.Normalize(padded, imageproc.ImageNetStandardMean, imageproc.ImageNetStandardSTD, true, true)
	}

	return pixelsLocal, pixelsGlobal, targetSize, nil
}
