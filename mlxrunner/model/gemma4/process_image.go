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
// multiples of patchSize*poolingKernel under a patch budget, a
// zero-flooring side clamped to one multiple.
func visionTargetSize(height, width, patchSize, poolingKernel, maxPatches int32) (targetH, targetW int32, err error) {
	if height <= 0 || width <= 0 {
		return 0, 0, fmt.Errorf("invalid image size %dx%d", width, height)
	}

	sideMult := patchSize * poolingKernel
	targetPx := float64(maxPatches) * float64(patchSize) * float64(patchSize)
	factor := math.Sqrt(targetPx / (float64(height) * float64(width)))
	targetH = int32(math.Floor(factor*float64(height)/float64(sideMult))) * sideMult
	targetW = int32(math.Floor(factor*float64(width)/float64(sideMult))) * sideMult
	if targetH == 0 && targetW == 0 {
		return 0, 0, fmt.Errorf("image %dx%d is too small to process", width, height)
	}

	maxSide := (maxPatches / (poolingKernel * poolingKernel)) * sideMult
	if targetH == 0 {
		targetH = sideMult
		targetW = min(int32(math.Floor(float64(width)/float64(height)))*sideMult, maxSide)
	} else if targetW == 0 {
		targetW = sideMult
		targetH = min(int32(math.Floor(float64(height)/float64(width)))*sideMult, maxSide)
	}
	if float64(targetH)*float64(targetW) > targetPx {
		return 0, 0, fmt.Errorf("image %dx%d exceeds the patch budget after resize", width, height)
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
	img = dropAlpha(img, bounds)
	targetH, targetW, err := visionTargetSize(int32(bounds.Dy()), int32(bounds.Dx()), patch, pool, m.visionSoftTokenBudget()*pool*pool)
	if err != nil {
		return nil, nil, ImageGeometry{}, err
	}

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
