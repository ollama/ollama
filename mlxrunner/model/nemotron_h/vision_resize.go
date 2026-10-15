package nemotron_h

import (
	"image"
	"image/color"
)

// resizeRadioImage matches the reference's float32 bicubic interpolation with
// antialiasing and align_corners=false. Keeping both passes in floating point
// preserves overshoot and avoids rounding/clamping to uint8 before normalization.
// The result is planar RGB in the source's 0..255 scale.
func resizeRadioImage(src image.Image, height, width int) []float32 {
	bounds := src.Bounds()
	srcH, srcW := bounds.Dy(), bounds.Dx()
	pixels := make([]float32, 3*srcH*width)
	row := make([]float32, 3*srcW)
	var horizontal []radioResizeFilter
	if srcW != width {
		horizontal = radioResizeWeights(srcW, width)
	}
	for y := range srcH {
		switch src := src.(type) {
		case *image.NRGBA:
			offset := src.PixOffset(bounds.Min.X, bounds.Min.Y+y)
			pixels := src.Pix[offset : offset+4*srcW]
			for x := range srcW {
				row[x] = float32(pixels[4*x])
				row[srcW+x] = float32(pixels[4*x+1])
				row[2*srcW+x] = float32(pixels[4*x+2])
			}
		case *image.YCbCr:
			offset := src.YOffset(bounds.Min.X, bounds.Min.Y+y)
			for x := range srcW {
				chroma := src.COffset(bounds.Min.X+x, bounds.Min.Y+y)
				r, g, b := color.YCbCrToRGB(src.Y[offset+x], src.Cb[chroma], src.Cr[chroma])
				row[x] = float32(r)
				row[srcW+x] = float32(g)
				row[2*srcW+x] = float32(b)
			}
		default:
			for x := range srcW {
				c := color.NRGBAModel.Convert(src.At(bounds.Min.X+x, bounds.Min.Y+y)).(color.NRGBA)
				row[x] = float32(c.R)
				row[srcW+x] = float32(c.G)
				row[2*srcW+x] = float32(c.B)
			}
		}
		for c := range 3 {
			srcRow := row[c*srcW : (c+1)*srcW]
			dstRow := pixels[(c*srcH+y)*width : (c*srcH+y+1)*width]
			if srcW == width {
				copy(dstRow, srcRow)
				continue
			}
			for x, filter := range horizontal {
				var sum float32
				for j, w := range filter.weights {
					sum += srcRow[filter.start+j] * w
				}
				dstRow[x] = sum
			}
		}
	}
	if srcH != height {
		weights := radioResizeWeights(srcH, height)
		out := make([]float32, 3*height*width)
		for c := range 3 {
			for y, filter := range weights {
				for x := range width {
					var sum float32
					for j, w := range filter.weights {
						sum += pixels[(c*srcH+filter.start+j)*width+x] * w
					}
					out[(c*height+y)*width+x] = sum
				}
			}
		}
		pixels = out
	}
	return pixels
}

type radioResizeFilter struct {
	start   int
	weights []float32
}

func radioResizeWeights(source, target int) []radioResizeFilter {
	scale := float32(source) / float32(target)
	filterScale := max(scale, 1)
	support := 2 * filterScale
	filters := make([]radioResizeFilter, target)
	for i := range target {
		center := (float32(i) + 0.5) * scale
		start := max(int(center-support+0.5), 0)
		end := min(int(center+support+0.5), source)
		weights := make([]float32, end-start)
		var sum float32
		for j := range weights {
			x := (float32(start+j) - center + 0.5) / filterScale
			if x < 0 {
				x = -x
			}
			// Cubic convolution with a=-0.5.
			if x < 1 {
				weights[j] = ((1.5*x-2.5)*x)*x + 1
			} else if x < 2 {
				weights[j] = ((-0.5*x+2.5)*x-4)*x + 2
			}
			sum += weights[j]
		}
		for j := range weights {
			weights[j] /= sum
		}
		filters[i] = radioResizeFilter{start: start, weights: weights}
	}
	return filters
}
