//go:build mlx

package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"log"
	"os"
	"path/filepath"
	"runtime/pprof"

	"github.com/ollama/ollama/x/imagegen"
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/models/flux2"
	"github.com/ollama/ollama/x/imagegen/models/gemma3"
	"github.com/ollama/ollama/x/imagegen/models/gpt_oss"
	"github.com/ollama/ollama/x/imagegen/models/llama"
	"github.com/ollama/ollama/x/imagegen/models/qwen_image"
	"github.com/ollama/ollama/x/imagegen/models/qwen_image_edit"
	"github.com/ollama/ollama/x/imagegen/models/zimage"
	"github.com/ollama/ollama/x/imagegen/safetensors"
)

// stringSlice is a flag type that accumulates multiple values
type stringSlice []string

func (s *stringSlice) String() string {
	return fmt.Sprintf("%v", *s)
}

func (s *stringSlice) Set(value string) error {
	*s = append(*s, value)
	return nil
}

func main() {
	modelPath := flag.String("model", "", "Model directory")
	prompt := flag.String("prompt", "Hello", "Prompt")

	// Text generation params
	maxTokens := flag.Int("max-tokens", 100, "Max tokens")
	temperature := flag.Float64("temperature", 0.7, "Temperature")
	topP := flag.Float64("top-p", 0.9, "Top-p sampling")
	topK := flag.Int("top-k", 40, "Top-k sampling")
	imagePath := flag.String("image", "", "Image path for multimodal models")

	// Image generation params
	width := flag.Int("width", 0, "Image width (0 = auto from input or 1024)")
	height := flag.Int("height", 0, "Image height (0 = auto from input or 1024)")
	steps := flag.Int("steps", 0, "Denoising steps (0 = model default)")
	seed := flag.Int64("seed", 42, "Random seed")
	out := flag.String("output", "output.png", "Output path")

	// Utility flags
	listTensors := flag.Bool("list", false, "List tensors only")
	cpuProfile := flag.String("cpuprofile", "", "Write CPU profile to file")
	gpuCapture := flag.String("gpu-capture", "", "Capture GPU trace to .gputrace file (run with MTL_CAPTURE_ENABLED=1)")
	layerCache := flag.Bool("layer-cache", false, "Enable layer caching for faster diffusion (Z-Image, Qwen-Image). Not compatible with CFG/negative prompts.")
	wiredLimitGB := flag.Int("wired-limit", 32, "Metal wired memory limit in GB")

	// Legacy mode flags
	zimageFlag := flag.Bool("zimage", false, "Z-Image generation")
	flux2Flag := flag.Bool("flux2", false, "FLUX.2 Klein generation")
	qwenImage := flag.Bool("qwen-image", false, "Qwen-Image text-to-image generation")
	qwenImageEdit := flag.Bool("qwen-image-edit", false, "Qwen-Image-Edit image editing")
	var inputImages stringSlice
	flag.Var(&inputImages, "input-image", "Input image for image editing (can be specified multiple times)")
	negativePrompt := flag.String("negative-prompt", "", "Negative prompt for CFG (empty = no CFG, matching Python)")
	cfgScale := flag.Float64("cfg-scale", 4.0, "CFG scale for image editing")
	teaCache := flag.Bool("teacache", false, "Enable TeaCache for faster inference")
	teaCacheThreshold := flag.Float64("teacache-threshold", 0.1, "TeaCache threshold (lower = more aggressive caching)")
	fusedQKV := flag.Bool("fused-qkv", false, "Enable fused QKV projection for faster attention")

	flag.Parse()

	if *modelPath == "" {
		flag.Usage()
		return
	}

	// Check if MLX initialized successfully
	if !mlx.IsMLXAvailable() {
		log.Fatalf("MLX initialization failed: %v", mlx.GetMLXInitError())
	}

	// CPU profiling
	if *cpuProfile != "" {
		f, err := os.Create(*cpuProfile)
		if err != nil {
			log.Fatal(err)
		}
		defer f.Close()
		if err := pprof.StartCPUProfile(f); err != nil {
			log.Fatal(err)
		}
		defer pprof.StopCPUProfile()
	}

	var err error

	// Handle legacy mode flags that aren't unified yet
	switch {
	case *zimageFlag:
		m := &zimage.Model{}
		if loadErr := m.Load(*modelPath); loadErr != nil {
			log.Fatal(loadErr)
		}
		var img *mlx.Array
		img, err = m.GenerateFromConfig(context.Background(), &zimage.GenerateConfig{
			Prompt:            *prompt,
			NegativePrompt:    *negativePrompt,
			CFGScale:          float32(*cfgScale),
			Width:             int32(*width),
			Height:            int32(*height),
			Steps:             *steps,
			Seed:              *seed,
			CapturePath:       *gpuCapture,
			TeaCache:          *teaCache,
			TeaCacheThreshold: float32(*teaCacheThreshold),
			FusedQKV:          *fusedQKV,
		})
		if err == nil {
			err = saveImageArray(img, *out)
		}
	case *flux2Flag:
		m := &flux2.Model{}
		if loadErr := m.Load(*modelPath); loadErr != nil {
			log.Fatal(loadErr)
		}
		// Load input images with EXIF orientation correction
		var loadedImages []image.Image
		for _, path := range inputImages {
			img, loadErr := loadImageWithEXIF(path)
			if loadErr != nil {
				log.Fatalf("Failed to load image %s: %v", path, loadErr)
			}
			loadedImages = append(loadedImages, img)
		}
		// When input images provided and user didn't override dimensions, use 0 to match input
		fluxWidth := int32(*width)
		fluxHeight := int32(*height)
		if len(loadedImages) > 0 && *width == 0 && *height == 0 {
			// Both unset, will auto-detect from input
		} else if len(loadedImages) > 0 && *width == 0 {
			fluxWidth = 0 // Compute from height + aspect ratio
		} else if len(loadedImages) > 0 && *height == 0 {
			fluxHeight = 0 // Compute from width + aspect ratio
		}
		var img *mlx.Array
		img, err = m.GenerateFromConfig(context.Background(), &flux2.GenerateConfig{
			Prompt:        *prompt,
			Width:         fluxWidth,
			Height:        fluxHeight,
			Steps:         *steps,
			GuidanceScale: float32(*cfgScale),
			Seed:          *seed,
			CapturePath:   *gpuCapture,
			InputImages:   loadedImages,
		})
		if err == nil {
			err = saveImageArray(img, *out)
		}
	case *qwenImage:
		m, loadErr := qwen_image.LoadPersistent(*modelPath)
		if loadErr != nil {
			log.Fatal(loadErr)
		}
		var img *mlx.Array
		img, err = m.GenerateFromConfig(&qwen_image.GenerateConfig{
			Prompt:         *prompt,
			NegativePrompt: *negativePrompt,
			CFGScale:       float32(*cfgScale),
			Width:          int32(*width),
			Height:         int32(*height),
			Steps:          *steps,
			Seed:           *seed,
			LayerCache:     *layerCache,
		})
		if err == nil {
			err = saveImageArray(img, *out)
		}
	case *qwenImageEdit:
		if len(inputImages) == 0 {
			log.Fatal("qwen-image-edit requires at least one -input-image")
		}

		m, loadErr := qwen_image_edit.LoadPersistent(*modelPath)
		if loadErr != nil {
			log.Fatal(loadErr)
		}
		// For image editing, use 0 for dimensions to auto-detect from input image
		// unless explicitly overridden from defaults
		editWidth := int32(0)
		editHeight := int32(0)
		if *width != 1024 {
			editWidth = int32(*width)
		}
		if *height != 1024 {
			editHeight = int32(*height)
		}

		cfg := &qwen_image_edit.GenerateConfig{
			Prompt:         *prompt,
			NegativePrompt: *negativePrompt,
			CFGScale:       float32(*cfgScale),
			Width:          editWidth,
			Height:         editHeight,
			Steps:          *steps,
			Seed:           *seed,
		}

		var img *mlx.Array
		img, err = m.EditFromConfig(inputImages, cfg)
		if err == nil {
			err = saveImageArray(img, *out)
		}
	case *listTensors:
		err = listModelTensors(*modelPath)
	default:
		// llm path
		m, err := load(*modelPath)
		if err != nil {
			log.Fatal(err)
		}

		// Load image if provided and model supports it
		var image *mlx.Array
		if *imagePath != "" {
			if mm, ok := m.(interface{ ImageSize() int32 }); ok {
				image, err = gemma3.ProcessImage(*imagePath, mm.ImageSize())
				if err != nil {
					log.Fatal("load image:", err)
				}
			} else {
				log.Fatal("model does not support image input")
			}
		}

		err = generate(context.Background(), m, input{
			Prompt:       *prompt,
			Image:        image,
			MaxTokens:    *maxTokens,
			Temperature:  float32(*temperature),
			TopP:         float32(*topP),
			TopK:         *topK,
			WiredLimitGB: *wiredLimitGB,
		}, func(out output) {
			if out.Text != "" {
				fmt.Print(out.Text)
			}
			if out.Done {
				fmt.Printf("\n\n[prefill: %.1f tok/s, gen: %.1f tok/s]\n", out.PrefillTokSec, out.GenTokSec)
			}
		})
	}

	if err != nil {
		log.Fatal(err)
	}
}

func listModelTensors(modelPath string) error {
	weights, err := safetensors.LoadModelWeights(modelPath)
	if err != nil {
		return err
	}
	for _, name := range weights.ListTensors() {
		info, _ := weights.GetTensorInfo(name)
		fmt.Printf("%s: %v (%s)\n", name, info.Shape, info.Dtype)
	}
	return nil
}

// loadModel builds and evaluates a model using the common load pattern.
// Release safetensors BEFORE eval - lazy arrays have captured their data,
// and this reduces peak memory by ~6GB (matches mlx-lm behavior).
func loadModel[T Model](build func() T, cleanup func()) T {
	m := build()
	weights := mlx.Collect(m)
	cleanup()
	mlx.Eval(weights...)
	return m
}

func load(modelPath string) (Model, error) {
	kind, err := detectModelKind(modelPath)
	if err != nil {
		return nil, fmt.Errorf("detect model kind: %w", err)
	}

	switch kind {
	case "gpt_oss":
		return gpt_oss.Load(modelPath)
	case "gemma3":
		return gemma3.Load(modelPath)
	case "gemma3_text":
		return gemma3.LoadText(modelPath)
	default:
		return llama.Load(modelPath)
	}
}

func detectModelKind(modelPath string) (string, error) {
	indexPath := filepath.Join(modelPath, "model_index.json")
	if _, err := os.Stat(indexPath); err == nil {
		data, err := os.ReadFile(indexPath)
		if err != nil {
			return "zimage", nil
		}
		var index struct {
			ClassName string `json:"_class_name"`
		}
		if err := json.Unmarshal(data, &index); err == nil {
			switch index.ClassName {
			case "FluxPipeline", "ZImagePipeline":
				return "zimage", nil
			case "Flux2KleinPipeline":
				return "flux2", nil
			}
		}
		return "zimage", nil
	}

	configPath := filepath.Join(modelPath, "config.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		return "", fmt.Errorf("no config.json or model_index.json found: %w", err)
	}

	var cfg struct {
		ModelType string `json:"model_type"`
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return "", fmt.Errorf("parse config.json: %w", err)
	}

	return cfg.ModelType, nil
}

// loadImageWithEXIF loads an image from a file path with EXIF orientation correction.
func loadImageWithEXIF(path string) (image.Image, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read file: %w", err)
	}
	return imagegen.DecodeImage(data)
}
