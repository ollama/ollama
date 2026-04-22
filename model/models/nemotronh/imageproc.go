package nemotronh

import (
	"errors"
	"image"
	"math"
	"slices"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/model/imageproc"
)

type ImageProcessor struct {
	imageSize      int
	patchSize      int
	numChannels    int
	maxTiles       int
	minNumPatches  int
	maxNumPatches  int
	useThumbnail   bool
	projectorScale int
	imageMean      [3]float32
	imageStd       [3]float32
}

type processedVisionTile struct {
	data []float32
	size image.Point
}

func newImageProcessor(c fs.Config) ImageProcessor {
	mean := c.Floats("vision.image_mean")
	std := c.Floats("vision.image_std")

	processor := ImageProcessor{
		imageSize:      int(c.Uint("vision.image_size", 512)),
		patchSize:      int(c.Uint("vision.patch_size", 16)),
		numChannels:    int(c.Uint("vision.num_channels", 3)),
		maxTiles:       int(c.Uint("vision.max_tiles", 12)),
		minNumPatches:  int(c.Uint("vision.min_num_patches")),
		maxNumPatches:  int(c.Uint("vision.max_num_patches")),
		useThumbnail:   c.Bool("vision.use_thumbnail", true),
		projectorScale: int(c.Uint("vision.projector.scale_factor", 2)),
		imageMean:      imageproc.ClipDefaultMean,
		imageStd:       imageproc.ClipDefaultSTD,
	}

	if len(mean) >= 3 {
		processor.imageMean = [3]float32{mean[0], mean[1], mean[2]}
	}
	if len(std) >= 3 {
		processor.imageStd = [3]float32{std[0], std[1], std[2]}
	}
	if processor.imageSize <= 0 {
		processor.imageSize = 512
	}
	if processor.patchSize <= 0 {
		processor.patchSize = 16
	}
	if processor.numChannels <= 0 {
		processor.numChannels = 3
	}
	if processor.maxTiles <= 0 {
		processor.maxTiles = 12
	}
	if processor.projectorScale <= 0 {
		processor.projectorScale = 2
	}

	return processor
}

func (p ImageProcessor) ProcessImage(img image.Image) ([]processedVisionTile, error) {
	img = imageproc.Composite(img)
	if p.useDynamicResolution() {
		return p.processDynamicImage(img)
	}

	return p.processTiledImage(img), nil
}

func (p ImageProcessor) useDynamicResolution() bool {
	return p.minNumPatches > 0 || p.maxNumPatches > 0
}

func (p ImageProcessor) processTiledImage(img image.Image) []processedVisionTile {
	bounds := img.Bounds()
	origWidth := bounds.Dx()
	origHeight := bounds.Dy()
	targetRatios := nemotronTargetRatios(p.maxTiles)
	gridWidth, gridHeight := findClosestAspectRatio(float64(origWidth)/float64(origHeight), targetRatios, origWidth, origHeight, p.imageSize)

	targetWidth := p.imageSize * gridWidth
	targetHeight := p.imageSize * gridHeight
	resized := resizeImageBicubicCHW(img, targetWidth, targetHeight)

	tiles := make([]processedVisionTile, 0, gridWidth*gridHeight+1)
	for row := range gridHeight {
		for col := range gridWidth {
			tile := cropCHWRegion(
				resized,
				targetWidth,
				targetHeight,
				p.numChannels,
				col*p.imageSize,
				row*p.imageSize,
				p.imageSize,
				p.imageSize,
			)
			tiles = append(tiles, processedVisionTile{
				data: normalizeVisionCHW(tile, p.imageMean, p.imageStd),
				size: image.Point{X: p.imageSize, Y: p.imageSize},
			})
		}
	}

	if p.useThumbnail && len(tiles) > 1 {
		thumbnail := resizeImageBicubicCHW(img, p.imageSize, p.imageSize)
		tiles = append(tiles, processedVisionTile{
			data: normalizeVisionCHW(thumbnail, p.imageMean, p.imageStd),
			size: image.Point{X: p.imageSize, Y: p.imageSize},
		})
	}

	return tiles
}

func (p ImageProcessor) processDynamicImage(img image.Image) ([]processedVisionTile, error) {
	bounds := img.Bounds()
	origWidth := bounds.Dx()
	origHeight := bounds.Dy()
	patchesWidth, patchesHeight := p.dynamicPatchGrid(origWidth, origHeight)
	if patchesWidth <= 0 || patchesHeight <= 0 {
		return nil, errors.New("nemotron_h_omni: invalid dynamic image patch grid")
	}

	targetWidth := patchesWidth * p.patchSize
	targetHeight := patchesHeight * p.patchSize
	resized := resizeImageBicubicCHW(img, targetWidth, targetHeight)

	return []processedVisionTile{{
		data: normalizeVisionCHW(resized, p.imageMean, p.imageStd),
		size: image.Point{X: targetWidth, Y: targetHeight},
	}}, nil
}

func (p ImageProcessor) dynamicPatchGrid(origWidth, origHeight int) (int, int) {
	patchesHeight := max(1, int(math.Round(float64(origHeight)/float64(p.patchSize)+0.5)))
	patchesWidth := max(1, int(math.Round(float64(origWidth)/float64(p.patchSize)+0.5)))

	patches := patchesHeight * patchesWidth
	currentNumPatchesAvailable := p.maxNumPatches
	if currentNumPatchesAvailable <= 0 {
		currentNumPatchesAvailable = max(patches, p.minNumPatches)
	}

	factor := math.Min(math.Sqrt(float64(currentNumPatchesAvailable)/float64(patches)), 1.0)
	targetPatchesHeight := max(1, int(math.Floor(factor*float64(patchesHeight))))
	targetPatchesWidth := max(1, int(math.Floor(factor*float64(patchesWidth))))

	if currentNumPatchesAvailable > p.minNumPatches && targetPatchesHeight*targetPatchesWidth < p.minNumPatches {
		upFactor := math.Sqrt(float64(p.minNumPatches) / float64(targetPatchesHeight*targetPatchesWidth))
		targetPatchesHeight = int(math.Ceil(upFactor * float64(targetPatchesHeight)))
		targetPatchesWidth = int(math.Ceil(upFactor * float64(targetPatchesWidth)))
	}

	targetPatchesHeight = roundPatchGridForPixelShuffle(targetPatchesHeight, targetPatchesWidth, currentNumPatchesAvailable, p.projectorScale)
	targetPatchesWidth = roundPatchGridForPixelShuffle(targetPatchesWidth, targetPatchesHeight, currentNumPatchesAvailable, p.projectorScale)

	return targetPatchesWidth, targetPatchesHeight
}

func roundPatchGridForPixelShuffle(v, other, maxPatches, divisor int) int {
	if divisor <= 1 {
		return v
	}
	rem := v % divisor
	if rem == 0 {
		return v
	}

	inc := divisor - rem
	if (v+inc)*other <= maxPatches {
		return v + inc
	}
	return max(divisor, v-rem)
}

type nemotronImageRatio struct {
	width  int
	height int
}

func nemotronTargetRatios(maxTiles int) []nemotronImageRatio {
	targetRatios := make([]nemotronImageRatio, 0, maxTiles*maxTiles)
	for n := 1; n <= maxTiles; n++ {
		for w := 1; w <= n; w++ {
			for h := 1; h <= n; h++ {
				if w*h > maxTiles {
					continue
				}
				targetRatios = append(targetRatios, nemotronImageRatio{width: w, height: h})
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

	slices.SortFunc(unique, func(a, b nemotronImageRatio) int {
		return a.width*a.height - b.width*b.height
	})

	return unique
}

func findClosestAspectRatio(aspectRatio float64, targetRatios []nemotronImageRatio, width, height, imageSize int) (int, int) {
	bestRatio := nemotronImageRatio{width: 1, height: 1}
	bestRatioDiff := math.MaxFloat64
	area := width * height

	for _, ratio := range targetRatios {
		targetAspectRatio := float64(ratio.width) / float64(ratio.height)
		ratioDiff := math.Abs(aspectRatio - targetAspectRatio)
		if ratioDiff < bestRatioDiff {
			bestRatioDiff = ratioDiff
			bestRatio = ratio
			continue
		}

		if ratioDiff == bestRatioDiff && area > int(0.5*float64(imageSize*imageSize*ratio.width*ratio.height)) {
			bestRatio = ratio
		}
	}

	return bestRatio.width, bestRatio.height
}

func resizeImageBicubicCHW(img image.Image, outW, outH int) []float32 {
	bounds := img.Bounds()
	inW := bounds.Dx()
	inH := bounds.Dy()
	src := make([]float32, 3*inW*inH)

	for y := range inH {
		for x := range inW {
			r, g, b, _ := img.At(bounds.Min.X+x, bounds.Min.Y+y).RGBA()
			src[y*inW+x] = float32(r>>8) / 255.0
			src[inW*inH+y*inW+x] = float32(g>>8) / 255.0
			src[2*inW*inH+y*inW+x] = float32(b>>8) / 255.0
		}
	}

	dst := make([]float32, 3*outW*outH)
	scaleX := float64(inW) / float64(outW)
	scaleY := float64(inH) / float64(outH)

	for oy := range outH {
		srcY := scaleY*(float64(oy)+0.5) - 0.5
		yBase := int(math.Floor(srcY))
		yFrac := clampUnit(srcY - float64(yBase))
		wy := torchBicubicWeights(yFrac)

		for ox := range outW {
			srcX := scaleX*(float64(ox)+0.5) - 0.5
			xBase := int(math.Floor(srcX))
			xFrac := clampUnit(srcX - float64(xBase))
			wx := torchBicubicWeights(xFrac)

			for c := range 3 {
				var sum float64
				channelOffset := c * inW * inH
				for ky := range 4 {
					iy := clampIndex(yBase-1+ky, 0, inH-1)
					for kx := range 4 {
						ix := clampIndex(xBase-1+kx, 0, inW-1)
						sum += float64(src[channelOffset+iy*inW+ix]) * wy[ky] * wx[kx]
					}
				}
				dst[c*outW*outH+oy*outW+ox] = float32(sum)
			}
		}
	}

	return dst
}

func cropCHWRegion(values []float32, width, height, channels, left, top, cropW, cropH int) []float32 {
	out := make([]float32, channels*cropW*cropH)
	channelSize := width * height
	cropSize := cropW * cropH
	for c := range channels {
		srcBase := c * channelSize
		dstBase := c * cropSize
		for y := range cropH {
			copy(out[dstBase+y*cropW:dstBase+(y+1)*cropW], values[srcBase+(top+y)*width+left:srcBase+(top+y)*width+left+cropW])
		}
	}
	return out
}

func normalizeVisionCHW(values []float32, mean, std [3]float32) []float32 {
	out := make([]float32, len(values))
	channelSize := len(values) / 3
	for c := range 3 {
		base := c * channelSize
		for i := range channelSize {
			out[base+i] = (values[base+i] - mean[c]) / std[c]
		}
	}
	return out
}

func torchBicubicWeights(t float64) [4]float64 {
	const a = -0.75
	return [4]float64{
		bicubicConvolution2(t+1.0, a),
		bicubicConvolution1(t, a),
		bicubicConvolution1(1.0-t, a),
		bicubicConvolution2(2.0-t, a),
	}
}

func bicubicConvolution1(x, a float64) float64 {
	return ((a+2)*x-(a+3))*x*x + 1
}

func bicubicConvolution2(x, a float64) float64 {
	return ((a*x-5*a)*x+8*a)*x - 4*a
}

func clampUnit(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

func clampIndex(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
