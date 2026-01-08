//go:build mlx

package qwen_image_edit

import (
	"fmt"
	"image"
	"image/color"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"os"

	"github.com/ollama/ollama/x/imagegen/mlx"
	"golang.org/x/image/draw"
	_ "golang.org/x/image/webp"
)

// loadImageFile loads an image from disk
func loadImageFile(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open image: %w", err)
	}
	defer f.Close()

	img, _, err := image.Decode(f)
	if err != nil {
		return nil, fmt.Errorf("decode image: %w", err)
	}
	return img, nil
}

// imageToFloat32Pixels converts an image to a float32 pixel array [H, W, C] in [0, 1] range
func imageToFloat32Pixels(img image.Image, width, height int) []float32 {
	pixels := make([]float32, width*height*3)
	idx := 0
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			pixels[idx] = float32(r) / 65535.0
			pixels[idx+1] = float32(g) / 65535.0
			pixels[idx+2] = float32(b) / 65535.0
			idx += 3
		}
	}
	return pixels
}

// normalizeImageNet applies ImageNet normalization to an image tensor
func (p *Processor) normalizeImageNet(arr *mlx.Array) *mlx.Array {
	mean := mlx.NewArray(p.Config.ImageMean, []int32{1, 1, 3})
	std := mlx.NewArray(p.Config.ImageStd, []int32{1, 1, 3})
	return mlx.Div(mlx.Sub(arr, mean), std)
}

// prepareImageTensor transforms [H, W, C] to [B, C, H, W] and converts to bf16
func prepareImageTensor(arr *mlx.Array) *mlx.Array {
	// Transpose to [C, H, W] and make contiguous
	arr = mlx.Contiguous(mlx.Transpose(arr, 2, 0, 1))
	// Add batch dimension [1, C, H, W]
	arr = mlx.ExpandDims(arr, 0)
	// Convert to bf16
	arr = mlx.ToBFloat16(arr)
	mlx.Eval(arr)
	return arr
}

// clampFloat clamps a value to [0, 255] and returns uint8
func clampFloat(v, weightSum float64) uint8 {
	v /= weightSum
	if v < 0 {
		v = 0
	}
	if v > 255 {
		v = 255
	}
	return uint8(math.Round(v))
}

// ImageDims holds dimensions for a preprocessed image
type ImageDims struct {
	// Original image dimensions
	OrigW, OrigH int32
	// Condition image dimensions (for vision encoder)
	CondW, CondH int32
	// VAE image dimensions
	VaeW, VaeH int32
	// Latent dimensions (VAE dims / vae_scale_factor)
	LatentW, LatentH int32
	// Patch dimensions (latent dims / patch_size)
	PatchW, PatchH int32
}

// ProcessorConfig holds image processor configuration
type ProcessorConfig struct {
	// Condition image size (target pixel area for vision encoder input)
	// Python: CONDITION_IMAGE_SIZE = 384 * 384 = 147456
	// Pipeline resizes image to this area before passing to encode_prompt
	ConditionImageSize int32

	// VAE image size (target pixel area)
	// Python: VAE_IMAGE_SIZE = 1024 * 1024 = 1048576
	VAEImageSize int32

	// Image normalization (ImageNet stats for vision encoder)
	ImageMean []float32
	ImageStd  []float32
}

// defaultProcessorConfig returns default processor config
func defaultProcessorConfig() *ProcessorConfig {
	return &ProcessorConfig{
		ConditionImageSize: 384 * 384,   // 147456 - matches Python CONDITION_IMAGE_SIZE
		VAEImageSize:       1024 * 1024, // 1048576 - matches Python VAE_IMAGE_SIZE
		ImageMean:          []float32{0.48145466, 0.4578275, 0.40821073},
		ImageStd:           []float32{0.26862954, 0.26130258, 0.27577711},
	}
}

// Processor handles image preprocessing for Qwen-Image-Edit
type Processor struct {
	Config *ProcessorConfig
}

// Load loads the processor config
func (p *Processor) Load(path string) error {
	p.Config = defaultProcessorConfig()
	return nil
}

// LoadAndPreprocess loads an image and preprocesses it for both paths
// Returns: condImage (for vision encoder), vaeImage (for VAE encoding)
func (p *Processor) LoadAndPreprocess(imagePath string) (*mlx.Array, *mlx.Array, error) {
	img, err := loadImageFile(imagePath)
	if err != nil {
		return nil, nil, err
	}

	bounds := img.Bounds()
	origW := bounds.Dx()
	origH := bounds.Dy()
	ratio := float64(origW) / float64(origH)

	// Calculate dimensions for condition image (vision encoder)
	// Python pipeline does TWO resizes:
	// 1. VaeImageProcessor.resize with Lanczos to CONDITION_IMAGE_SIZE (384x384 area)
	// 2. Qwen2VLProcessor's smart_resize with Bicubic to multiple of 28
	intermediateW, intermediateH := calculateDimensions(p.Config.ConditionImageSize, ratio, 32)
	finalH, finalW := smartResize(intermediateH, intermediateW, 28, 56*56, 28*28*1280)

	// Calculate dimensions for VAE image (1024x1024 area)
	// Use multiple of 32 (vae_scale_factor * patch_size * 2 = 8 * 2 * 2 = 32)
	vaeW, vaeH := calculateDimensions(p.Config.VAEImageSize, ratio, 32)

	// Preprocess for condition (vision encoder) - two-step resize
	condImage := p.preprocessImageTwoStep(img, intermediateW, intermediateH, finalW, finalH)

	// Preprocess for VAE ([-1, 1] range, 5D tensor)
	vaeImage := p.preprocessImageForVAE(img, vaeW, vaeH)

	return condImage, vaeImage, nil
}

// preprocessImageLanczos does single-step Lanczos resize for vision encoder
// Matches Python VaeImageProcessor.resize with resample='lanczos' (the default)
// Used by edit_plus pipeline for multi-image input
// Returns: [B, C, H, W] normalized tensor
func (p *Processor) preprocessImageLanczos(img image.Image, width, height int32) *mlx.Array {
	resized := resizeImageLanczos(img, int(width), int(height))
	pixels := imageToFloat32Pixels(resized, int(width), int(height))
	arr := mlx.NewArray(pixels, []int32{height, width, 3})
	arr = p.normalizeImageNet(arr)
	return prepareImageTensor(arr)
}

// preprocessImageTwoStep does two-step resize for vision encoder to match Python pipeline
// Step 1: Lanczos resize from original to intermediate size (VaeImageProcessor.resize)
// Step 2: Bicubic resize from intermediate to final size (Qwen2VLProcessor smart_resize)
// Returns: [B, C, H, W] normalized tensor
func (p *Processor) preprocessImageTwoStep(img image.Image, intermediateW, intermediateH, finalW, finalH int32) *mlx.Array {
	intermediate := resizeImageLanczos(img, int(intermediateW), int(intermediateH))
	resized := resizeImageBicubic(intermediate, int(finalW), int(finalH))
	pixels := imageToFloat32Pixels(resized, int(finalW), int(finalH))
	arr := mlx.NewArray(pixels, []int32{finalH, finalW, 3})
	arr = p.normalizeImageNet(arr)
	return prepareImageTensor(arr)
}

// preprocessImage converts image to tensor for vision encoder
// Returns: [B, C, H, W] normalized tensor
func (p *Processor) preprocessImage(img image.Image, width, height int32, normalize bool) *mlx.Array {
	resized := resizeImageBicubic(img, int(width), int(height))
	pixels := imageToFloat32Pixels(resized, int(width), int(height))
	arr := mlx.NewArray(pixels, []int32{height, width, 3})
	if normalize {
		arr = p.normalizeImageNet(arr)
	}
	return prepareImageTensor(arr)
}

// preprocessImageForVAE converts image to tensor for VAE encoding
// Returns: [B, C, T, H, W] tensor in [-1, 1] range
func (p *Processor) preprocessImageForVAE(img image.Image, width, height int32) *mlx.Array {
	resized := resizeImageLanczos(img, int(width), int(height))
	pixels := imageToFloat32Pixels(resized, int(width), int(height))
	arr := mlx.NewArray(pixels, []int32{height, width, 3})

	// Scale to [-1, 1]: arr * 2 - 1
	arr = mlx.MulScalar(arr, 2.0)
	arr = mlx.AddScalar(arr, -1.0)

	// Transpose to [C, H, W] and make contiguous
	arr = mlx.Contiguous(mlx.Transpose(arr, 2, 0, 1))

	// Add batch and temporal dimensions [1, C, 1, H, W]
	arr = mlx.ExpandDims(arr, 0) // [1, C, H, W]
	arr = mlx.ExpandDims(arr, 2) // [1, C, 1, H, W]

	arr = mlx.ToBFloat16(arr)
	mlx.Eval(arr)
	return arr
}

// smartResize implements Python Qwen2VL processor's smart_resize logic
// Returns (resizedHeight, resizedWidth) that fit within min/max pixel constraints
func smartResize(height, width, factor, minPixels, maxPixels int32) (int32, int32) {
	// Round to factor
	hBar := int32(math.Round(float64(height)/float64(factor))) * factor
	wBar := int32(math.Round(float64(width)/float64(factor))) * factor

	// Ensure minimum factor size
	if hBar < factor {
		hBar = factor
	}
	if wBar < factor {
		wBar = factor
	}

	// Check pixel constraints
	total := hBar * wBar
	if total > maxPixels {
		// Scale down
		beta := math.Sqrt(float64(maxPixels) / float64(total))
		hBar = int32(math.Floor(float64(height)*beta/float64(factor))) * factor
		wBar = int32(math.Floor(float64(width)*beta/float64(factor))) * factor
	} else if total < minPixels {
		// Scale up
		beta := math.Sqrt(float64(minPixels) / float64(total))
		hBar = int32(math.Ceil(float64(height)*beta/float64(factor))) * factor
		wBar = int32(math.Ceil(float64(width)*beta/float64(factor))) * factor
	}

	return hBar, wBar
}

// calculateDimensions calculates width and height for a target area while maintaining ratio
// multiple: the value to round dimensions to (e.g., 28 for vision encoder with patch 14 and 2x2 merge)
func calculateDimensions(targetArea int32, ratio float64, multiple int32) (int32, int32) {
	width := math.Sqrt(float64(targetArea) * ratio)
	height := width / ratio

	m := float64(multiple)
	width = math.Round(width/m) * m
	height = math.Round(height/m) * m

	// Ensure minimum dimensions
	if width < m {
		width = m
	}
	if height < m {
		height = m
	}

	return int32(width), int32(height)
}

// resizeImageLanczos resizes an image using Lanczos3 interpolation (matches PIL.LANCZOS)
func resizeImageLanczos(img image.Image, width, height int) image.Image {
	bounds := img.Bounds()
	dst := image.NewRGBA(image.Rect(0, 0, width, height))

	// Lanczos3 kernel (a=3) to match PIL.LANCZOS
	lanczos3 := &draw.Kernel{
		Support: 3.0,
		At: func(t float64) float64 {
			if t == 0 {
				return 1.0
			}
			if t < 0 {
				t = -t
			}
			if t >= 3.0 {
				return 0.0
			}
			// sinc(t) * sinc(t/3)
			piT := math.Pi * t
			return (math.Sin(piT) / piT) * (math.Sin(piT/3) / (piT / 3))
		},
	}
	lanczos3.Scale(dst, dst.Bounds(), img, bounds, draw.Over, nil)

	return dst
}

// resizeImageBicubic resizes an image using bicubic interpolation (matches PIL.BICUBIC)
// Uses separable interpolation with PIL's coordinate mapping for exact match
func resizeImageBicubic(img image.Image, width, height int) image.Image {
	bounds := img.Bounds()
	srcW := bounds.Dx()
	srcH := bounds.Dy()

	// Convert to RGBA if needed
	var src *image.RGBA
	if rgba, ok := img.(*image.RGBA); ok {
		src = rgba
	} else {
		src = image.NewRGBA(bounds)
		for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
			for x := bounds.Min.X; x < bounds.Max.X; x++ {
				src.Set(x, y, img.At(x, y))
			}
		}
	}

	// Keys cubic with a=-0.5 (PIL BICUBIC)
	cubic := func(x float64) float64 {
		if x < 0 {
			x = -x
		}
		if x < 1 {
			return 1.5*x*x*x - 2.5*x*x + 1
		}
		if x < 2 {
			return -0.5*x*x*x + 2.5*x*x - 4*x + 2
		}
		return 0
	}

	// Horizontal pass: srcW -> width, keep srcH rows
	temp := image.NewRGBA(image.Rect(0, 0, width, srcH))
	for y := 0; y < srcH; y++ {
		for dstX := 0; dstX < width; dstX++ {
			// PIL coordinate mapping: center-to-center
			srcXf := (float64(dstX)+0.5)*(float64(srcW)/float64(width)) - 0.5
			baseX := int(math.Floor(srcXf))

			var sumR, sumG, sumB, sumA, weightSum float64
			for i := -1; i <= 2; i++ {
				sx := baseX + i
				if sx < 0 {
					sx = 0
				}
				if sx >= srcW {
					sx = srcW - 1
				}

				w := cubic(math.Abs(srcXf - float64(baseX+i)))
				c := src.RGBAAt(sx, y)
				sumR += float64(c.R) * w
				sumG += float64(c.G) * w
				sumB += float64(c.B) * w
				sumA += float64(c.A) * w
				weightSum += w
			}

			temp.SetRGBA(dstX, y, color.RGBA{
				clampFloat(sumR, weightSum),
				clampFloat(sumG, weightSum),
				clampFloat(sumB, weightSum),
				clampFloat(sumA, weightSum),
			})
		}
	}

	// Vertical pass: srcH -> height
	dst := image.NewRGBA(image.Rect(0, 0, width, height))
	for x := 0; x < width; x++ {
		for dstY := 0; dstY < height; dstY++ {
			srcYf := (float64(dstY)+0.5)*(float64(srcH)/float64(height)) - 0.5
			baseY := int(math.Floor(srcYf))

			var sumR, sumG, sumB, sumA, weightSum float64
			for j := -1; j <= 2; j++ {
				sy := baseY + j
				if sy < 0 {
					sy = 0
				}
				if sy >= srcH {
					sy = srcH - 1
				}

				w := cubic(math.Abs(srcYf - float64(baseY+j)))
				c := temp.RGBAAt(x, sy)
				sumR += float64(c.R) * w
				sumG += float64(c.G) * w
				sumB += float64(c.B) * w
				sumA += float64(c.A) * w
				weightSum += w
			}

			dst.SetRGBA(x, dstY, color.RGBA{
				clampFloat(sumR, weightSum),
				clampFloat(sumG, weightSum),
				clampFloat(sumB, weightSum),
				clampFloat(sumA, weightSum),
			})
		}
	}

	return dst
}

// LoadAndPreprocessMultiple loads multiple images and preprocesses them
// Returns: condImages (for vision encoder), vaeImages (for VAE encoding), dims (per-image dimensions)
func (p *Processor) LoadAndPreprocessMultiple(imagePaths []string) ([]*mlx.Array, []*mlx.Array, []ImageDims, error) {
	const vaeScaleFactor int32 = 8
	const patchSize int32 = 2

	condImages := make([]*mlx.Array, len(imagePaths))
	vaeImages := make([]*mlx.Array, len(imagePaths))
	dims := make([]ImageDims, len(imagePaths))

	for i, imagePath := range imagePaths {
		img, err := loadImageFile(imagePath)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("image %d: %w", i, err)
		}

		bounds := img.Bounds()
		origW := int32(bounds.Dx())
		origH := int32(bounds.Dy())
		ratio := float64(origW) / float64(origH)

		// Calculate dimensions for condition image (vision encoder)
		// Python pipeline does TWO resizes:
		// 1. VaeImageProcessor.resize with Lanczos to CONDITION_IMAGE_SIZE (384x384 area)
		// 2. Qwen2VLProcessor's smart_resize with Bicubic to multiple of 28
		intermediateW, intermediateH := calculateDimensions(p.Config.ConditionImageSize, ratio, 32)
		condH, condW := smartResize(intermediateH, intermediateW, 28, 56*56, 28*28*1280)

		// Calculate dimensions for VAE image (1024x1024 area)
		vaeW, vaeH := calculateDimensions(p.Config.VAEImageSize, ratio, 32)

		// Calculate derived dimensions
		latentW := vaeW / vaeScaleFactor
		latentH := vaeH / vaeScaleFactor
		patchW := latentW / patchSize
		patchH := latentH / patchSize

		dims[i] = ImageDims{
			OrigW:   origW,
			OrigH:   origH,
			CondW:   condW,
			CondH:   condH,
			VaeW:    vaeW,
			VaeH:    vaeH,
			LatentW: latentW,
			LatentH: latentH,
			PatchW:  patchW,
			PatchH:  patchH,
		}

		fmt.Printf("  Image %d: orig=%dx%d, cond=%dx%d, vae=%dx%d, latent=%dx%d, patch=%dx%d\n",
			i+1, origW, origH, condW, condH, vaeW, vaeH, latentW, latentH, patchW, patchH)

		// Preprocess for condition (vision encoder) - two-step resize to match Python pipeline
		condImages[i] = p.preprocessImageTwoStep(img, intermediateW, intermediateH, condW, condH)

		// Preprocess for VAE ([-1, 1] range, 5D tensor)
		vaeImages[i] = p.preprocessImageForVAE(img, vaeW, vaeH)
	}

	return condImages, vaeImages, dims, nil
}
