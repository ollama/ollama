package gemma4

import (
	"bytes"
	"fmt"
	"image"
	"math"

	"golang.org/x/image/draw"
)

// ImageGeometry describes a preprocessed image's patch grid.
type ImageGeometry struct {
	PatchesW, PatchesH int32
	NumSoftTokens      int32
}

// preparedImage is gemma4's model-private media state: the patch position
// grid the encoder consumes and the geometry the forward pass derives the
// soft-token run from.
type preparedImage struct {
	positions []int32
	geom      ImageGeometry
}

// visionTargetSize ports the reference resize: sides floored to
// multiples of patchSize*poolingKernel under a soft-token budget, a
// zero-flooring side clamped to one multiple.
func visionTargetSize(height, width int, patchSize, poolingKernel, budget int32) (targetH, targetW int32, err error) {
	if height <= 0 || width <= 0 {
		return 0, 0, fmt.Errorf("invalid image size %dx%d", width, height)
	}

	side := int64(patchSize) * int64(poolingKernel)
	if patchSize <= 0 || poolingKernel <= 0 || budget <= 0 ||
		side > math.MaxInt32/int64(budget) || int64(poolingKernel)*int64(poolingKernel) > math.MaxInt32/int64(budget) {
		return 0, 0, fmt.Errorf("invalid vision geometry: patch %d, pool %d, budget %d", patchSize, poolingKernel, budget)
	}
	sideMult := float64(side)
	targetPx := float64(budget) * sideMult * sideMult
	factor := math.Sqrt(targetPx / (float64(height) * float64(width)))
	h := math.Floor(factor*float64(height)/sideMult) * sideMult
	w := math.Floor(factor*float64(width)/sideMult) * sideMult
	if h == 0 && w == 0 {
		return 0, 0, fmt.Errorf("image %dx%d is too small to process", width, height)
	}

	maxSide := float64(budget) * sideMult
	if h == 0 {
		h = sideMult
		w = min(math.Floor(float64(width)/float64(height))*sideMult, maxSide)
	} else if w == 0 {
		w = sideMult
		h = min(math.Floor(float64(height)/float64(width))*sideMult, maxSide)
	}
	if h*w > targetPx || h > maxSide || w > maxSide {
		return 0, 0, fmt.Errorf("image %dx%d exceeds the patch budget after resize", width, height)
	}
	return int32(h), int32(w), nil
}

// dynamicVisionTargetSize chooses the smallest publisher-supported mode that
// preserves both source dimensions, or the largest mode when none fits.
func dynamicVisionTargetSize(height, width int, patch, pool int32) (int32, int32, error) {
	var targetH, targetW int32
	for _, budget := range [...]int32{70, 140, 280, 560, 1120} {
		var err error
		targetH, targetW, err = visionTargetSize(height, width, patch, pool, budget)
		if err != nil {
			return 0, 0, err
		}
		// Compare the rounded grid: choosing the nearest area can discard
		// detail even when a larger supported mode preserves it.
		if int(targetH) >= height && int(targetW) >= width {
			return targetH, targetW, nil
		}
	}
	return targetH, targetW, nil
}

// preprocessImage decodes and prepares one image: aspect-preserving
// resize, rescale to [0,1], and patchify. The [-1,1] normalization stays
// in the patch embedder, so pixels here match the reference
// pixel_values.
func (m *Model) preprocessImage(data []byte) (pixels []float32, positions []int32, geom ImageGeometry, err error) {
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, nil, ImageGeometry{}, fmt.Errorf("decode image: %w", err)
	}

	patch, pool := m.Vision.PatchSize, m.Vision.PoolingKernelSize
	bounds := img.Bounds()
	targetH, targetW, err := dynamicVisionTargetSize(bounds.Dy(), bounds.Dx(), patch, pool)
	if err != nil {
		return nil, nil, ImageGeometry{}, err
	}
	positionPatch := patch
	if m.Vision.unified() {
		positionPatch *= pool
	}
	if int(max(targetH, targetW)/positionPatch) > m.Vision.positionEmbeddingSize {
		return nil, nil, ImageGeometry{}, fmt.Errorf("image patch grid exceeds vision position embedding size %d", m.Vision.positionEmbeddingSize)
	}

	img = dropAlpha(img, bounds)
	resized := image.NewRGBA(image.Rect(0, 0, int(targetW), int(targetH)))
	draw.CatmullRom.Scale(resized, resized.Bounds(), img, bounds, draw.Src, nil)

	if m.Vision.unified() {
		// One raster patch of pool*patchSize pixels per soft token: the
		// reference merge rearranges its intermediate 16px patches back
		// into this layout, with positions on the merged grid.
		pixels, positions, geom = patchify(resized, targetW, targetH, patch*pool, 1)
	} else {
		pixels, positions, geom = patchify(resized, targetW, targetH, patch, pool)
	}
	return pixels, positions, geom, nil
}

// patchify converts the resized image to the tower's layout: one row per
// patchSize patch, (pixel row, pixel column, RGB) within it.
func patchify(resized *image.RGBA, targetW, targetH, patch, pool int32) ([]float32, []int32, ImageGeometry) {
	pW, pH := targetW/patch, targetH/patch
	numPatches := int(pW * pH)
	patchLen := int(patch * patch * 3)
	pixels := make([]float32, numPatches*patchLen)
	positions := make([]int32, 2*numPatches)
	for p := range numPatches {
		gx, gy := int32(p)%pW, int32(p)/pW
		positions[2*p] = gx
		positions[2*p+1] = gy
		writePatch(pixels[p*patchLen:], resized, int(gx*patch), int(gy*patch), int(patch))
	}

	return pixels, positions, ImageGeometry{PatchesW: pW, PatchesH: pH, NumSoftTokens: pW * pH / (pool * pool)}
}

func writePatch(out []float32, resized *image.RGBA, baseX, baseY, patch int) {
	for py := range patch {
		row := resized.PixOffset(baseX, baseY+py)
		for px := range patch {
			o := (py*patch + px) * 3
			pix := resized.Pix[row+px*4:]
			out[o] = float32(pix[0]) / 255
			out[o+1] = float32(pix[1]) / 255
			out[o+2] = float32(pix[2]) / 255
		}
	}
}

// dropAlpha flattens a non-opaque image to straight RGB: the reference
// drops alpha via RGB conversion before resizing, not by compositing.
func dropAlpha(img image.Image, bounds image.Rectangle) image.Image {
	if o, ok := img.(interface{ Opaque() bool }); ok && o.Opaque() {
		return img
	}
	flat := image.NewNRGBA(bounds)
	draw.Draw(flat, bounds, img, bounds.Min, draw.Src)
	for i := 3; i < len(flat.Pix); i += 4 {
		flat.Pix[i] = 0xff
	}
	return flat
}
