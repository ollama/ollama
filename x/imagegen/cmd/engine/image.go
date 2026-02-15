//go:build mlx

package main

import (
	"fmt"
	"image"
	"image/png"
	"os"
	"path/filepath"

	"github.com/ollama/ollama/x/imagegen/mlx"
)

// saveImageArray saves an MLX array as a PNG image.
// Expected format: [B, C, H, W] with values in [0, 1] range and C=3 (RGB).
func saveImageArray(arr *mlx.Array, path string) error {
	img, err := arrayToImage(arr)
	if err != nil {
		return err
	}
	return savePNG(img, path)
}

func savePNG(img *image.RGBA, path string) error {
	if filepath.Ext(path) != ".png" {
		path = path + ".png"
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, img)
}

func arrayToImage(arr *mlx.Array) (*image.RGBA, error) {
	shape := arr.Shape()
	if len(shape) != 4 {
		return nil, fmt.Errorf("expected 4D array [B, C, H, W], got %v", shape)
	}

	// Transform to [H, W, C] for image conversion
	img := mlx.Squeeze(arr, 0)
	arr.Free()
	img = mlx.Transpose(img, 1, 2, 0)
	img = mlx.Contiguous(img)
	mlx.Eval(img)

	imgShape := img.Shape()
	H := int(imgShape[0])
	W := int(imgShape[1])
	C := int(imgShape[2])

	if C != 3 {
		img.Free()
		return nil, fmt.Errorf("expected 3 channels (RGB), got %d", C)
	}

	// Copy to CPU and free GPU memory
	data := img.Data()
	img.Free()

	// Write directly to Pix slice (faster than SetRGBA)
	goImg := image.NewRGBA(image.Rect(0, 0, W, H))
	pix := goImg.Pix
	for y := 0; y < H; y++ {
		for x := 0; x < W; x++ {
			srcIdx := (y*W + x) * C
			dstIdx := (y*W + x) * 4
			pix[dstIdx+0] = uint8(clampF(data[srcIdx+0]*255+0.5, 0, 255))
			pix[dstIdx+1] = uint8(clampF(data[srcIdx+1]*255+0.5, 0, 255))
			pix[dstIdx+2] = uint8(clampF(data[srcIdx+2]*255+0.5, 0, 255))
			pix[dstIdx+3] = 255
		}
	}

	return goImg, nil
}

func clampF(v, min, max float32) float32 {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}
