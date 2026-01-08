//go:build mlx

package gemma3

import (
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"os"

	"github.com/ollama/ollama/x/imagegen/mlx"
	"golang.org/x/image/draw"
)

// ProcessImage loads and preprocesses an image for the vision tower
// Returns [1, H, W, C] tensor in NHWC format normalized for SigLIP
func ProcessImage(path string, imageSize int32) (*mlx.Array, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open image: %w", err)
	}
	defer f.Close()

	img, _, err := image.Decode(f)
	if err != nil {
		return nil, fmt.Errorf("decode image: %w", err)
	}

	return ProcessImageData(img, imageSize)
}

// ProcessImageData preprocesses an image.Image for the vision tower
func ProcessImageData(img image.Image, imageSize int32) (*mlx.Array, error) {
	// Resize to target size using bilinear interpolation
	resized := image.NewRGBA(image.Rect(0, 0, int(imageSize), int(imageSize)))
	draw.BiLinear.Scale(resized, resized.Bounds(), img, img.Bounds(), draw.Over, nil)

	// Convert to float32 array [H, W, C] and normalize
	// SigLIP normalization: (pixel / 255.0 - 0.5) / 0.5 = pixel / 127.5 - 1.0
	data := make([]float32, imageSize*imageSize*3)
	idx := 0
	for y := int32(0); y < imageSize; y++ {
		for x := int32(0); x < imageSize; x++ {
			r, g, b, _ := resized.At(int(x), int(y)).RGBA()
			// RGBA returns 16-bit values, convert to 8-bit
			data[idx] = float32(r>>8)/127.5 - 1.0
			data[idx+1] = float32(g>>8)/127.5 - 1.0
			data[idx+2] = float32(b>>8)/127.5 - 1.0
			idx += 3
		}
	}

	// Create MLX array [1, H, W, C] for NHWC layout
	arr := mlx.NewArrayFloat32(data, []int32{1, imageSize, imageSize, 3})
	mlx.Eval(arr) // Materialize to prevent use-after-free
	return arr, nil
}
