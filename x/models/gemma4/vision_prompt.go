package gemma4

import (
	"bytes"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"log/slog"
	"math"
	"regexp"
	"strconv"

	"golang.org/x/image/draw"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	"github.com/ollama/ollama/x/models/nn"
)

var imageTagPattern = regexp.MustCompile(`\[img-(\d+)\]`)

// Image fidelity is bought with memory, so we take the highest budget the device
// can hold rather than defaulting low: a machine with room to spare reads tables
// and small text, and a small machine still answers at lower fidelity.
//
// Measured with gemma4:e4b-nvfp4 (8.37 GiB resident, the more expensive of the
// two towers) transcribing the OCR document from vision_ocr_test.go on an
// M5 Max, reading peak memory at the end of prefill:
//
//	budget  prompt tokens      peak   above the resident model
//	   280            304  8.85 GiB                   0.48 GiB
//	   560            578  9.31 GiB                   0.94 GiB
//	  1120           1102 10.21 GiB                   1.84 GiB
//
// Cost is linear in soft tokens - about 1.6 MiB each - because an image token
// costs roughly what a text token costs; the tower's own activations are
// near-noise beside the KV cache for the tokens it produces. The headroom
// thresholds carry about a 2x margin over the measured cost of the budget they
// unlock. A model with a larger KV footprint per token than e4b will cost
// proportionally more at the same budget.
const (
	// gemma4MinImageTokens floors the resize so a tiny image is still upscaled
	// to something the tower can read.
	gemma4MinImageTokens = 40

	gemma4FallbackImageTokens = 280
	gemma4StandardImageTokens = 560
	gemma4MaxImageTokens      = 1120

	gemma4StandardResHeadroom = 2 << 30 // 2 GiB, ~2x the measured 0.94 GiB
	gemma4HighResHeadroom     = 4 << 30 // 4 GiB, ~2x the measured 1.84 GiB
)

type visionInput struct {
	pixels    *mlx.Array
	numTokens int
}

type visionSegment struct {
	start      int
	end        int
	numTokens  int
	input      *visionInput
	featureKey base.MediaFeatureKey
	features   *base.MediaFeatureLease
}

type preparedMultimodalPrompt struct {
	model        *Model
	tokens       []int32
	maskedTokens []int32
	segments     []visionSegment
	closed       bool
}

func (p *preparedMultimodalPrompt) Tokens() []int32 {
	return p.tokens
}

func (p *preparedMultimodalPrompt) Forward(b *batch.Batch, caches []cache.Cache) *mlx.Array {
	chunkStart := int(b.SeqOffsets[0])
	chunkEnd := chunkStart + b.InputIDs.Dim(1)

	mlx.Pin(b.InputIDs)
	defer mlx.Unpin(b.InputIDs)
	for i := range p.segments {
		segment := &p.segments[i]
		overlapStart := max(chunkStart, segment.start)
		overlapEnd := min(chunkEnd, segment.end)
		if overlapStart >= overlapEnd {
			continue
		}

		if segment.features == nil {
			if features, ok := p.model.mediaFeatures.Acquire(segment.featureKey); ok {
				if features.TokenCount() != segment.numTokens {
					// Request-goroutine panic, kept only because Forward has no
					// error return in the pipeline's current shape. It belongs
					// with the stopgap media cache and should become a returned
					// error once multimodal inputs use the normal cache
					// lifecycle. See base.MediaFeatureCache.Store.
					panic("gemma4 cached vision token count changed")
				}
				segment.features = features
				mlx.Unpin(segment.input.pixels)
				segment.input = nil
			} else {
				features := p.model.ForwardVision(segment.input.pixels)
				mlx.Eval(features)
				segment.features = p.model.mediaFeatures.Store(
					segment.featureKey,
					features,
					segment.numTokens,
				)
			}
		}
	}

	h := p.model.embed(b.InputIDs)
	for i := range p.segments {
		segment := &p.segments[i]
		overlapStart := max(chunkStart, segment.start)
		overlapEnd := min(chunkEnd, segment.end)
		if overlapStart >= overlapEnd {
			continue
		}

		featureStart := overlapStart - segment.start
		featureEnd := featureStart + overlapEnd - overlapStart
		replacement := segment.features.Features().Slice(
			mlx.Slice(),
			mlx.Slice(featureStart, featureEnd),
			mlx.Slice(),
		).AsType(h.DType())
		h = h.SliceUpdate(
			replacement,
			mlx.Slice(),
			mlx.Slice(overlapStart-chunkStart, overlapEnd-chunkStart),
			mlx.Slice(),
		)
	}

	var pleTokens *mlx.Array
	if p.model.HiddenSizePerLayer > 0 {
		masked := p.maskedTokens[chunkStart:chunkEnd]
		pleTokens = mlx.FromValues(masked, 1, len(masked))
	}

	attentionMask := nn.CausalMask()
	if p.model.UseBidirectionalAttn == "vision" {
		for _, segment := range p.segments {
			attentionMask = attentionMask.Relax(0, segment.start, segment.end, segment.start, segment.end)
		}
	}
	return p.model.forwardEmbeddings(h, b, caches, pleTokens, attentionMask)
}

func (p *preparedMultimodalPrompt) Close() {
	if p.closed {
		return
	}
	p.closed = true
	for i := range p.segments {
		if p.segments[i].input != nil {
			mlx.Unpin(p.segments[i].input.pixels)
		}
		if p.segments[i].features != nil {
			p.segments[i].features.Release()
		}
	}
}

func (m *Model) PrepareMultimodalPrompt(prompt string, media []base.MediaInput, headroomBytes int) (base.PreparedMultimodalPrompt, error) {
	if m.VisionEncoder == nil || m.MultimodalEmbed == nil {
		return nil, fmt.Errorf("gemma4 model has no supported vision encoder")
	}

	byID := make(map[int]base.MediaInput, len(media))
	for _, item := range media {
		if item.Kind != "" && item.Kind != "image" {
			return nil, fmt.Errorf("gemma4 does not support %s input", item.Kind)
		}
		if _, exists := byID[item.ID]; exists {
			return nil, fmt.Errorf("duplicate media ID %d", item.ID)
		}
		byID[item.ID] = item
	}

	parts := imageTagPattern.Split(prompt, -1)
	matches := imageTagPattern.FindAllStringSubmatch(prompt, -1)
	if len(matches) != len(media) {
		return nil, fmt.Errorf("prompt contains %d image markers for %d images", len(matches), len(media))
	}

	prepared := &preparedMultimodalPrompt{model: m}
	used := make(map[int]bool, len(media))
	imageTokenLimit := gemma4ImageTokenLimit(headroomBytes)
	slog.Debug("gemma4 vision token limit",
		"tokens", imageTokenLimit,
		"headroom", mlx.PrettyBytes(headroomBytes),
	)
	featureVariant := "gemma4:image:tokens=" + strconv.Itoa(imageTokenLimit)
	cleanup := func(err error) (base.PreparedMultimodalPrompt, error) {
		prepared.Close()
		return nil, err
	}

	for i, part := range parts {
		prepared.tokens = append(prepared.tokens, m.tok.Encode(part, i == 0 && m.tok.AddBOS())...)
		if i >= len(matches) {
			continue
		}

		id, err := strconv.Atoi(matches[i][1])
		if err != nil {
			return cleanup(fmt.Errorf("invalid image marker %q: %w", matches[i][0], err))
		}
		item, ok := byID[id]
		if !ok {
			return cleanup(fmt.Errorf("image marker references missing media ID %d", id))
		}
		if used[id] {
			return cleanup(fmt.Errorf("image media ID %d is referenced more than once", id))
		}
		used[id] = true

		featureKey := base.NewMediaFeatureKey(item.Data, featureVariant)
		features, cached := m.mediaFeatures.Acquire(featureKey)
		var input *visionInput
		var numTokens int
		if cached {
			numTokens = features.TokenCount()
		} else {
			input, err = m.preprocessImage(item.Data, imageTokenLimit)
			if err != nil {
				return cleanup(fmt.Errorf("preprocess image %d: %w", id, err))
			}
			mlx.Eval(input.pixels)
			mlx.Pin(input.pixels)
			numTokens = input.numTokens
		}
		slog.Debug("gemma4 vision feature cache", "hit", cached, "tokens", numTokens)

		prepared.tokens = append(prepared.tokens, m.imageStartTokenID)
		start := len(prepared.tokens)
		for range numTokens {
			prepared.tokens = append(prepared.tokens, m.imageTokenID)
		}
		prepared.segments = append(prepared.segments, visionSegment{
			start:      start,
			end:        len(prepared.tokens),
			numTokens:  numTokens,
			input:      input,
			featureKey: featureKey,
			features:   features,
		})
		prepared.tokens = append(prepared.tokens, m.imageEndTokenID)
	}

	prepared.maskedTokens = append([]int32(nil), prepared.tokens...)
	for _, segment := range prepared.segments {
		clear(prepared.maskedTokens[segment.start:segment.end])
	}
	return prepared, nil
}

func (m *Model) preprocessImage(data []byte, maxTokens int) (*visionInput, error) {
	cfg := m.VisionConfig
	if cfg == nil {
		return nil, fmt.Errorf("model has no vision config")
	}

	src, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("decode image: %w", err)
	}

	bounds := src.Bounds()
	srcW, srcH := bounds.Dx(), bounds.Dy()
	if srcW <= 0 || srcH <= 0 {
		return nil, fmt.Errorf("invalid image dimensions %dx%d", srcW, srcH)
	}

	patchSize := int(cfg.PatchSize)
	alignSize := int(cfg.PoolingKernelSize) * patchSize
	pixelsPerToken := alignSize * alignSize
	minTokens := min(maxTokens, max(gemma4MinImageTokens, int(cfg.ImageSeqLength)))
	targetH, targetW := gemma4ImageSize(
		srcH,
		srcW,
		alignSize,
		minTokens*pixelsPerToken,
		maxTokens*pixelsPerToken,
	)

	dst := image.NewRGBA(image.Rect(0, 0, targetW, targetH))
	draw.CatmullRom.Scale(dst, dst.Bounds(), src, bounds, draw.Over, nil)

	pixels := make([]float32, 3*targetH*targetW)
	for y := range targetH {
		for x := range targetW {
			offset := y*dst.Stride + x*4
			pixels[y*targetW+x] = float32(dst.Pix[offset]) / 255
			pixels[targetH*targetW+y*targetW+x] = float32(dst.Pix[offset+1]) / 255
			pixels[2*targetH*targetW+y*targetW+x] = float32(dst.Pix[offset+2]) / 255
		}
	}

	return &visionInput{
		pixels:    mlx.FromValues(pixels, 1, 3, targetH, targetW),
		numTokens: m.VisionTokenCount(targetH, targetW),
	}, nil
}

// gemma4ImageTokenLimit picks the highest image fidelity the memory left over
// after loading the model can hold. See the budget constants for the
// measurements behind each step.
func gemma4ImageTokenLimit(headroom int) int {
	switch {
	case headroom >= gemma4HighResHeadroom:
		return gemma4MaxImageTokens
	case headroom >= gemma4StandardResHeadroom:
		return gemma4StandardImageTokens
	default:
		return gemma4FallbackImageTokens
	}
}

func gemma4ImageSize(height, width, alignment, minPixels, maxPixels int) (int, int) {
	roundTo := func(value float64) int {
		return int(math.Round(value/float64(alignment))) * alignment
	}

	targetH := max(alignment, roundTo(float64(height)))
	targetW := max(alignment, roundTo(float64(width)))

	switch {
	case float64(targetH)*float64(targetW) > float64(maxPixels):
		targetH, targetW = gemma4FitImageSize(height, width, alignment, maxPixels)
	case float64(targetH)*float64(targetW) < float64(minPixels):
		targetH, targetW = gemma4FitImageSize(height, width, alignment, minPixels)
	}

	return targetH, targetW
}

func gemma4FitImageSize(height, width, alignment, pixelBudget int) (int, int) {
	scale := math.Sqrt(float64(pixelBudget) / (float64(height) * float64(width)))
	targetH := int(math.Floor(scale*float64(height)/float64(alignment))) * alignment
	targetW := int(math.Floor(scale*float64(width)/float64(alignment))) * alignment

	maxSide := pixelBudget / (alignment * alignment) * alignment
	switch {
	case targetH == 0 && targetW == 0:
		return alignment, alignment
	case targetH == 0:
		targetH = alignment
		targetW = min(int(math.Floor(float64(width)/float64(height)))*alignment, maxSide)
	case targetW == 0:
		targetW = alignment
		targetH = min(int(math.Floor(float64(height)/float64(width)))*alignment, maxSide)
	}

	return targetH, targetW
}
