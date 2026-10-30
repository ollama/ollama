package videoclip

import (
	"image"
	"math"
)

type resampleTap struct {
	index  int
	weight float64
}

func cubicWeight(x float64) float64 {
	x = math.Abs(x)
	switch {
	case x < 1:
		return (1.5*x-2.5)*x*x + 1
	case x < 2:
		return ((-0.5*x+2.5)*x-4)*x + 2
	default:
		return 0
	}
}

func bicubicPlan(source, target int) [][]resampleTap {
	plan := make([][]resampleTap, target)
	scale := float64(source) / float64(target)
	filterScale := math.Max(scale, 1)
	for output := range target {
		center := (float64(output) + 0.5) * scale
		first := max(0, int(math.Floor(center-2*filterScale-0.5)))
		last := min(source, int(math.Ceil(center+2*filterScale-0.5))+1)
		var total float64
		for input := first; input < last; input++ {
			weight := cubicWeight((float64(input) + 0.5 - center) / filterScale)
			if weight == 0 {
				continue
			}
			plan[output] = append(plan[output], resampleTap{index: input, weight: weight})
			total += weight
		}
		for i := range plan[output] {
			plan[output][i].weight /= total
		}
	}
	return plan
}

func roundedByte(v float64) uint8 {
	return uint8(max(0, min(255, int(math.Floor(v+0.5)))))
}

// ResizeBicubic is the reference processors' antialiased bicubic resize,
// separable and rounded to bytes after each pass. LFM3 images use it too.
func ResizeBicubic(source *image.NRGBA, width, height int) *image.NRGBA {
	sourceW, sourceH := source.Bounds().Dx(), source.Bounds().Dy()
	horizontal := image.NewNRGBA(image.Rect(0, 0, width, sourceH))
	for x, taps := range bicubicPlan(sourceW, width) {
		for y := range sourceH {
			for channel := range 3 {
				var value float64
				for _, tap := range taps {
					value += float64(source.Pix[source.PixOffset(tap.index, y)+channel]) * tap.weight
				}
				horizontal.Pix[horizontal.PixOffset(x, y)+channel] = roundedByte(value)
			}
			horizontal.Pix[horizontal.PixOffset(x, y)+3] = 255
		}
	}
	output := image.NewNRGBA(image.Rect(0, 0, width, height))
	for y, taps := range bicubicPlan(sourceH, height) {
		for x := range width {
			for channel := range 3 {
				var value float64
				for _, tap := range taps {
					value += float64(horizontal.Pix[horizontal.PixOffset(x, tap.index)+channel]) * tap.weight
				}
				output.Pix[output.PixOffset(x, y)+channel] = roundedByte(value)
			}
			output.Pix[output.PixOffset(x, y)+3] = 255
		}
	}
	return output
}

// PackPatches lays out same-size frames, one per temporal slot, as the
// vision tower's patch rows: merge blocks in raster order, each row holding
// channel, time, y, x, normalized to [-1, 1].
func PackPatches(frames []*image.NRGBA, patch, merge int32) []float32 {
	temporal := int32(len(frames))
	bounds := frames[0].Bounds()
	gridH, gridW := int32(bounds.Dy())/patch, int32(bounds.Dx())/patch
	channels := int32(3) * temporal * patch * patch
	pixels := make([]float32, gridH*gridW*channels)
	row := int32(0)
	for blockH := int32(0); blockH < gridH; blockH += merge {
		for blockW := int32(0); blockW < gridW; blockW += merge {
			for innerH := range merge {
				for innerW := range merge {
					patchH, patchW := blockH+innerH, blockW+innerW
					for channel := range int32(3) {
						for time, frame := range frames {
							for y := range patch {
								for x := range patch {
									pixel := frame.PixOffset(int(patchW*patch+x), int(patchH*patch+y))
									index := row*channels + (((channel*temporal+int32(time))*patch+y)*patch + x)
									pixels[index] = float32(frame.Pix[pixel+int(channel)])/127.5 - 1
								}
							}
						}
					}
					row++
				}
			}
		}
	}
	return pixels
}
