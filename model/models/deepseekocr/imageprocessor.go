package deepseekocr

import (
	"bytes"
	"image"
	"image/color"
	"math"
	"slices"

	"golang.org/x/image/draw"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/imageproc"
)

type ratio struct {
	x, y int
}

func ProcessImage(ctx ml.Context, bts []byte) (ml.Tensor, ml.Tensor, []int, error) {
	img, _, err := image.Decode(bytes.NewReader(bts))
	if err != nil {
		return nil, nil, nil, err
	}

	minNum, maxNum, imageSize, baseSize := 2, 9, 640, 1024
	var targetRatios []ratio
	for n := minNum; n <= maxNum; n++ {
		for i := 1; i <= n; i++ {
			for j := 1; j <= n; j++ {
				if i*j <= maxNum && i*j >= minNum && !slices.Contains(targetRatios, ratio{i, j}) {
					targetRatios = append(targetRatios, ratio{i, j})
				}
			}
		}
	}

	targetRatio := findBestAspectRatio(targetRatios, img.Bounds().Dx(), img.Bounds().Dy(), imageSize)
	targetWidth, targetHeight := imageSize*targetRatio.x, imageSize*targetRatio.y
	blocks := targetRatio.x * targetRatio.y

	mean := imageproc.ImageNetStandardMean
	std := imageproc.ImageNetStandardSTD

	var patches []float32
	resized := imageproc.Resize(img, image.Point{X: targetWidth, Y: targetHeight}, imageproc.ResizeBilinear)
	for i := range blocks {
		patch := image.NewRGBA(image.Rect(0, 0, imageSize, imageSize))
		draw.Draw(patch, patch.Bounds(), resized, image.Point{
			X: i % (targetWidth / imageSize) * imageSize,
			Y: i / (targetWidth / imageSize) * imageSize,
		}, draw.Over)

		patches = append(patches, imageproc.Normalize(patch, mean, std, true, true)...)
	}

	img = imageproc.CompositeColor(img, color.Gray{})
	img = imageproc.Pad(img, image.Point{X: baseSize, Y: baseSize}, color.Gray{127}, draw.BiLinear)

	return ctx.Input().FromFloats(patches, imageSize, imageSize, 3, blocks),
		ctx.Input().FromFloats(imageproc.Normalize(img, mean, std, true, true), baseSize, baseSize, 3),
		[]int{targetRatio.x, targetRatio.y},
		nil
}

func findBestAspectRatio(targetRatios []ratio, width, height, imageSize int) ratio {
	bestDiff := math.MaxFloat64
	best := ratio{1, 1}
	realRatio := float64(width) / float64(height)
	for _, target := range targetRatios {
		targetRatio := float64(target.x) / float64(target.y)
		diff := math.Abs(realRatio - targetRatio)
		if diff < bestDiff {
			bestDiff = diff
			best = target
		} else if diff == bestDiff {
			if float64(width*height) > 0.5*float64(imageSize*imageSize*best.x*best.y) {
				best = target
			}
		}
	}
	return best
}
