package qwen3vl

import (
	"bytes"
	"fmt"
	"image"
	"image/jpeg"
	_ "image/png"
	"math"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/imageproc"
)

// ImageProcessor contains configuration for the Qwen 3 VL image processing
type ImageProcessor struct {
	numChannels       int
	patchSize         int
	temporalPatchSize int
	mergeSize         int
	shortestEdge      int
	longestEdge       int
	factor            int
	rescaleFactor     float32
	imageMean         []float32
	imageStd          []float32
}

// newImageProcessor creates a new image processor with default values
func newImageProcessor(c fs.Config) ImageProcessor {
	patchSize := int(c.Uint("vision.patch_size", 14))
	mergeSize := int(c.Uint("vision.spatial_merge_size", 2))

	return ImageProcessor{
		numChannels:       int(c.Uint("vision.num_channels", 3)), // not set
		patchSize:         patchSize,
		temporalPatchSize: 2,
		mergeSize:         mergeSize,
		shortestEdge:      int(c.Uint("vision.shortest_edge", 64<<10)),
		// FIXME(mxyng): the model defined longest edge (16M) is too large for the default
		// context length of 8K and will panic. Adjusting to 2M for now.
		// longestEdge:   int(c.Uint("vision.longest_edge", 16<<20)),
		longestEdge:   2 << 20,
		factor:        patchSize * mergeSize,
		rescaleFactor: 1.0 / 255.0,
		imageMean:     c.Floats("vision.image_mean", imageproc.ImageNetStandardMean[:]),
		imageStd:      c.Floats("vision.image_std", imageproc.ImageNetStandardSTD[:]),
	}
}

// SmartResize implements the smart resize algorithm
func (p *ImageProcessor) SmartResize(height, width int) (int, int) {
	factor := p.factor

	if height < factor || width < factor {
		panic(fmt.Sprintf("height:%d or width:%d must be larger than factor:%d", height, width, factor))
	} else if aspectRatio := max(height, width) / min(height, width); aspectRatio > 200 {
		panic(fmt.Sprintf("absolute aspect ratio must be smaller than 200, got %v", aspectRatio))
	}

	round := func(x float64) int { return int(math.RoundToEven(x)) }

	hBar := round(float64(height)/float64(factor)) * factor
	wBar := round(float64(width)/float64(factor)) * factor

	if hBar*wBar > p.longestEdge {
		beta := math.Sqrt(float64(height*width) / float64(p.longestEdge))

		hBar = int(math.Floor(float64(height)/beta/float64(factor))) * factor
		wBar = int(math.Floor(float64(width)/beta/float64(factor))) * factor
	} else if hBar*wBar < p.shortestEdge {
		beta := math.Sqrt(float64(p.shortestEdge) / float64(height*width))

		hBar = int(math.Ceil(float64(height)*beta/float64(factor))) * factor
		wBar = int(math.Ceil(float64(width)*beta/float64(factor))) * factor
	}

	return hBar, wBar
}

// Grid represents the spatial and temporal dimensions of processed vision inputs.
//
// For images:
//   - Height, Width: Spatial patch grid dimensions (in units of merged patches)
//   - Temporal: Always 1 for static images
//
// For videos:
//   - Height, Width: Spatial patch grid dimensions (in units of merged patches)
//   - Temporal: Number of temporal patch groups (ceil(numFrames / temporalPatchSize))
//
// The total number of patches is Temporal × Height × Width.
// This structure is used by the vision model to apply position embeddings correctly.
type Grid struct {
	Height   int // Spatial height in patch units
	Width    int // Spatial width in patch units
	Temporal int // Temporal dimension (1 for images, >1 for videos)
}

func (p *ImageProcessor) ProcessImage(ctx ml.Context, img image.Image) (ml.Tensor, *Grid, error) {
	img = imageproc.Composite(img)

	origWidth := img.Bounds().Dx()
	origHeight := img.Bounds().Dy()

	// Calculate smart resize dimensions
	resizedHeight, resizedWidth := p.SmartResize(origHeight, origWidth)

	// Resize image using existing functions
	resizedImg := imageproc.Resize(img, image.Point{X: resizedWidth, Y: resizedHeight}, imageproc.ResizeBilinear)

	normalizedPixels := imageproc.Normalize(
		resizedImg,
		[3]float32{p.imageMean[0], p.imageMean[1], p.imageMean[2]},
		[3]float32{p.imageStd[0], p.imageStd[1], p.imageStd[2]},
		true, // rescale
		true, // channelFirst
	)

	// Calculate grid dimensions
	grid := &Grid{
		Height:   resizedHeight / p.patchSize,
		Width:    resizedWidth / p.patchSize,
		Temporal: 1, // For single images, temporal dimension is 1
	}

	patches, err := p.createPatches(normalizedPixels, resizedHeight, resizedWidth, grid)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create patches: %v", err)
	}

	patchDim := p.numChannels * p.temporalPatchSize *
		p.patchSize * p.patchSize
	numPatches := grid.Temporal * grid.Height * grid.Width

	pixelValues := ctx.Input().FromFloats(patches, patchDim, numPatches)

	// Return patches and grid dimensions
	return pixelValues, grid, nil
}

// ProcessVideoFrames processes multiple video frames with temporal awareness for Qwen3-VL.
//
// This method implements the video processing pipeline for Qwen3-VL's temporal understanding:
//
//  1. Frame Preprocessing:
//     - Composites each frame (removes alpha channel)
//     - Resizes frames to optimal dimensions using SmartResize
//     - Normalizes pixel values using ImageNet statistics
//
//  2. Temporal Patch Creation:
//     - Groups frames into temporal patches using temporalPatchSize (typically 2)
//     - Creates 3D patches (Time × Height × Width) from the video
//     - Applies 2×2 spatial merging for efficiency
//
//  3. Grid Calculation:
//     - Temporal dimension: ceil(numFrames / temporalPatchSize)
//     - Spatial dimensions: (H/patchSize/mergeSize) × (W/patchSize/mergeSize)
//
// The output tensor has shape [patchDim, numPatches] where:
//   - patchDim = channels × temporalPatchSize × patchSize × patchSize
//   - numPatches = Temporal × Height × Width
//
// Parameters:
//   - ctx: ML context for tensor creation
//   - frames: Slice of video frames as image.Image objects
//
// Returns:
//   - ml.Tensor: Processed video patches ready for the vision model
//   - *Grid: Grid structure containing Temporal, Height, Width dimensions
//   - error: Any error encountered during processing
func (p *ImageProcessor) ProcessVideoFrames(ctx ml.Context, frames []image.Image) (ml.Tensor, *Grid, error) {
	if len(frames) == 0 {
		return nil, nil, fmt.Errorf("no frames provided")
	}

	// Composite first frame to get dimensions
	firstFrame := imageproc.Composite(frames[0])
	origWidth := firstFrame.Bounds().Dx()
	origHeight := firstFrame.Bounds().Dy()

	// Calculate smart resize dimensions
	resizedHeight, resizedWidth := p.SmartResize(origHeight, origWidth)

	// Calculate grid dimensions with temporal component
	numFrames := len(frames)
	grid := &Grid{
		Height:   resizedHeight / p.patchSize / p.mergeSize,
		Width:    resizedWidth / p.patchSize / p.mergeSize,
		Temporal: (numFrames + p.temporalPatchSize - 1) / p.temporalPatchSize, // Ceiling division
	}

	// Process all frames and collect pixels
	allPixels := make([]float32, 0, numFrames*3*resizedHeight*resizedWidth)

	for _, frame := range frames {
		// Composite frame
		frame = imageproc.Composite(frame)

		// Resize frame
		resizedImg := imageproc.Resize(frame, image.Point{X: resizedWidth, Y: resizedHeight}, imageproc.ResizeBilinear)

		// Normalize
		normalizedPixels := imageproc.Normalize(
			resizedImg,
			[3]float32{p.imageMean[0], p.imageMean[1], p.imageMean[2]},
			[3]float32{p.imageStd[0], p.imageStd[1], p.imageStd[2]},
			true, // rescale
			true, // channelFirst
		)

		allPixels = append(allPixels, normalizedPixels...)
	}

	// Create patches with temporal dimension
	patches, err := p.createPatchesWithTemporal(allPixels, resizedHeight, resizedWidth, numFrames, grid)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create temporal patches: %v", err)
	}

	patchDim := p.numChannels * p.temporalPatchSize * p.patchSize * p.patchSize
	numPatches := grid.Temporal * grid.Height * grid.Width

	pixelValues := ctx.Input().FromFloats(patches, patchDim, numPatches)

	return pixelValues, grid, nil
}

func (p *ImageProcessor) createPatches(pixels []float32, height, width int, grid *Grid) ([]float32, error) {
	channels := p.numChannels
	patchSize := p.patchSize
	mergeSize := p.mergeSize
	temporalPatchSize := p.temporalPatchSize

	// Calculate output dimensions
	numPatches := grid.Temporal * grid.Height * grid.Width
	patchDim := channels * temporalPatchSize * patchSize * patchSize

	result := make([]float32, numPatches*patchDim)
	patchIndex := 0

	// Single temporal frame handling (copies to all frames)
	for range grid.Temporal {
		for h := 0; h < grid.Height; h += mergeSize {
			for w := 0; w < grid.Width; w += mergeSize {
				// Handle the 2x2 merged patches
				for mh := range mergeSize {
					for mw := range mergeSize {
						baseOffset := patchIndex * patchDim

						// Extract patch data for first temporal frame
						for c := range channels {
							channelOffset := baseOffset + (c * temporalPatchSize * patchSize * patchSize)

							for py := range patchSize {
								for px := range patchSize {
									// Calculate source pixel coordinates
									y := (h+mh)*patchSize + py
									x := (w+mw)*patchSize + px

									// Source index in input tensor (CHW format)
									srcIdx := c*height*width + y*width + x

									// Destination index in first temporal frame
									dstIdx := channelOffset + (py * patchSize) + px

									if srcIdx < len(pixels) && dstIdx < len(result) {
										result[dstIdx] = pixels[srcIdx]
									}
								}
							}
						}

						// Copy first temporal frame to all other frames
						if temporalPatchSize > 1 {
							for c := range channels {
								channelOffset := baseOffset + (c * temporalPatchSize * patchSize * patchSize)
								firstFrameOffset := channelOffset
								frameSize := patchSize * patchSize

								// Copy first frame to all other frames
								for tp := 1; tp < temporalPatchSize; tp++ {
									currentFrameOffset := channelOffset + (tp * frameSize)
									copy(result[currentFrameOffset:currentFrameOffset+frameSize],
										result[firstFrameOffset:firstFrameOffset+frameSize])
								}
							}
						}

						patchIndex++
					}
				}
			}
		}
	}

	return result, nil
}

// createPatchesWithTemporal creates 3D video patches with temporal and spatial dimensions.
//
// This function implements Qwen3-VL's video patching strategy:
//
//  1. Temporal Grouping:
//     - Groups consecutive frames into temporal patches of size temporalPatchSize (typically 2)
//     - Each temporal patch contains pixel data from multiple frames
//     - If the last group has fewer frames, it's padded by repeating the last frame
//
//  2. Spatial Patching:
//     - Divides each frame spatially into patches of size patchSize × patchSize
//     - Applies 2×2 merge: combines 4 adjacent patches into a single unit
//     - This reduces the spatial resolution by 4× (2×2 merging)
//
//  3. Patch Organization:
//     - Output format: [Temporal][Height][Width][MergeH][MergeW][Channel][TemporalIdx][PatchH][PatchW]
//     - Flattened into a 1D array with shape: [numPatches × patchDim]
//     - numPatches = Temporal × Height × Width × mergeSize × mergeSize
//     - patchDim = channels × temporalPatchSize × patchSize × patchSize
//
// Parameters:
//   - pixels: Flattened pixel data for all frames in CHW format (channels, height, width) per frame
//   - height, width: Spatial dimensions of each frame
//   - numFrames: Total number of frames
//   - grid: Grid structure specifying output dimensions
//
// Returns:
//   - []float32: Flattened patch data ready for Conv3D processing
//   - error: Any error encountered during patch creation
func (p *ImageProcessor) createPatchesWithTemporal(pixels []float32, height, width, numFrames int, grid *Grid) ([]float32, error) {
	channels := p.numChannels
	patchSize := p.patchSize
	mergeSize := p.mergeSize
	temporalPatchSize := p.temporalPatchSize

	// Calculate output dimensions
	numPatches := grid.Temporal * grid.Height * grid.Width
	patchDim := channels * temporalPatchSize * patchSize * patchSize

	result := make([]float32, numPatches*patchDim)
	patchIndex := 0

	// Iterate over temporal groups
	for t := range grid.Temporal {
		// Get frames for this temporal patch
		frameStart := t * temporalPatchSize
		frameEnd := frameStart + temporalPatchSize
		if frameEnd > numFrames {
			frameEnd = numFrames
		}

		// Iterate over spatial grid with 2x2 merging
		for h := range grid.Height {
			for w := range grid.Width {
				// Handle the 2x2 merged patches
				for mh := 0; mh < mergeSize; mh++ {
					for mw := 0; mw < mergeSize; mw++ {
						baseOffset := patchIndex * patchDim

						// Extract patch data for each temporal frame in this group
						for tf := frameStart; tf < frameEnd; tf++ {
							temporalIdx := tf - frameStart
							frameOffset := tf * channels * height * width

							for c := 0; c < channels; c++ {
								channelOffset := baseOffset +
									(c * temporalPatchSize * patchSize * patchSize) +
									(temporalIdx * patchSize * patchSize)

								for py := 0; py < patchSize; py++ {
									for px := 0; px < patchSize; px++ {
										// Calculate source pixel coordinates
										y := (h*mergeSize+mh)*patchSize + py
										x := (w*mergeSize+mw)*patchSize + px

										// Source index in input tensor (CHW format per frame)
										srcIdx := frameOffset + c*height*width + y*width + x

										// Destination index in patch
										dstIdx := channelOffset + (py * patchSize) + px

										if srcIdx < len(pixels) && dstIdx < len(result) {
											result[dstIdx] = pixels[srcIdx]
										}
									}
								}
							}
						}

						// If we have fewer frames than temporalPatchSize, pad with last frame
						if frameEnd-frameStart < temporalPatchSize {
							lastFrame := frameEnd - 1
							lastFrameOffset := lastFrame * channels * height * width

							for tf := frameEnd - frameStart; tf < temporalPatchSize; tf++ {
								temporalIdx := tf

								for c := 0; c < channels; c++ {
									channelOffset := baseOffset +
										(c * temporalPatchSize * patchSize * patchSize) +
										(temporalIdx * patchSize * patchSize)

									for py := 0; py < patchSize; py++ {
										for px := 0; px < patchSize; px++ {
											y := (h*mergeSize+mh)*patchSize + py
											x := (w*mergeSize+mw)*patchSize + px

											srcIdx := lastFrameOffset + c*height*width + y*width + x
											dstIdx := channelOffset + (py * patchSize) + px

											if srcIdx < len(pixels) && dstIdx < len(result) {
												result[dstIdx] = pixels[srcIdx]
											}
										}
									}
								}
							}
						}

						patchIndex++
					}
				}
			}
		}
	}

	return result, nil
}

// ExtractVideoFrames extracts frames from video data using ffmpeg.
// This is a convenience wrapper around imageproc.ExtractVideoFrames with default settings.
// For more control over extraction parameters, use imageproc.ExtractVideoFrames directly.
func ExtractVideoFrames(videoData []byte, fps float64) ([]image.Image, error) {
	config := imageproc.DefaultVideoConfig()
	config.FPS = fps
	return imageproc.ExtractVideoFrames(videoData, config)
}

// ProcessVideo extracts frames from video and processes them as images
// Returns a slice of processed frame data
func (p *ImageProcessor) ProcessVideo(ctx ml.Context, videoData []byte, fps float64) ([][]byte, error) {
	// Extract frames from video
	frames, err := ExtractVideoFrames(videoData, fps)
	if err != nil {
		return nil, fmt.Errorf("failed to extract video frames: %w", err)
	}

	if len(frames) == 0 {
		return nil, fmt.Errorf("no frames extracted from video")
	}

	// Convert each frame to image data (same format as regular images)
	// This allows reusing the existing image processing pipeline
	frameData := make([][]byte, 0, len(frames))
	for _, frame := range frames {
		// Encode frame back to JPEG
		var buf bytes.Buffer
		if err := jpeg.Encode(&buf, frame, &jpeg.Options{Quality: 90}); err != nil {
			return nil, fmt.Errorf("failed to encode frame: %w", err)
		}
		frameData = append(frameData, buf.Bytes())
	}

	return frameData, nil
}
