package qwen3_5

import (
	"fmt"
	"math"
	"regexp"
	"strconv"

	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
)

var imageTagRE = regexp.MustCompile(`\[img-(\d+)\]`)

type promptVisionSpan struct {
	Start int32
	End   int32

	Main *mlx.Array
	Grid *VisionGrid
}

type promptVisionState struct {
	Spans         []promptVisionSpan
	PositionCache []int32
}

func promptStartPosFromCaches(caches []cache.Cache) int32 {
	offset := -1
	for _, c := range caches {
		if c == nil {
			continue
		}
		off := c.Offset()
		if offset < 0 || off < offset {
			offset = off
		}
	}
	if offset < 0 {
		return 0
	}
	return int32(offset)
}

func promptVisionStateFromState(state any) *promptVisionState {
	typed, _ := state.(*promptVisionState)
	return typed
}

func overlapRange(chunkStart, chunkLen, spanStart, spanEnd int32) (int32, int32, int32, int32, bool) {
	chunkEnd := chunkStart + chunkLen
	overlapStart := max(chunkStart, spanStart)
	overlapEnd := min(chunkEnd, spanEnd)
	if overlapStart >= overlapEnd {
		return 0, 0, 0, 0, false
	}

	chunkLo := overlapStart - chunkStart
	chunkHi := overlapEnd - chunkStart
	spanLo := overlapStart - spanStart
	spanHi := overlapEnd - spanStart
	return chunkLo, chunkHi, spanLo, spanHi, true
}

func (m *Model) applyPromptVisionEmbeddings(h *mlx.Array, startPos int32, state *promptVisionState) *mlx.Array {
	if m == nil || h == nil || state == nil || len(state.Spans) == 0 {
		return h
	}

	dims := h.Dims()
	if len(dims) != 3 {
		return h
	}

	L := int32(dims[1])
	for _, span := range state.Spans {
		chunkLo, chunkHi, spanLo, spanHi, ok := overlapRange(startPos, L, span.Start, span.End)
		if !ok || span.Main == nil || !span.Main.Valid() {
			continue
		}

		repl := span.Main.Slice(
			mlx.Slice(),
			mlx.Slice(int(spanLo), int(spanHi)),
			mlx.Slice(),
		)
		repl = repl.AsType(h.DType())
		h = h.SliceUpdate(
			repl,
			mlx.Slice(),
			mlx.Slice(int(chunkLo), int(chunkHi)),
			mlx.Slice(),
		)
	}

	return h
}

func findImageByID(images []base.ImageInput, id int) (base.ImageInput, bool) {
	for i := range images {
		if images[i].ID == id {
			return images[i], true
		}
	}
	return base.ImageInput{}, false
}

func mapPromptPosition(state *promptVisionState, id int32) int32 {
	if state == nil {
		return id
	}
	if id < int32(len(state.PositionCache)) {
		return state.PositionCache[id]
	}
	if len(state.PositionCache) > 0 {
		return id - int32(len(state.PositionCache)) + state.PositionCache[len(state.PositionCache)-1] + 1
	}
	return id
}

func promptVisionGridSpan(grid *VisionGrid, merge int32, fallback int32) int32 {
	if fallback <= 0 {
		fallback = 1
	}
	if grid == nil {
		return fallback
	}
	if merge <= 0 {
		merge = 1
	}
	return max(max(int32(1), grid.Width/merge), max(int32(1), grid.Height/merge))
}

func normalizeMRoPESections(sections []int32) [4]int32 {
	var out [4]int32
	for i := range min(4, len(sections)) {
		if sections[i] > 0 {
			out[i] = sections[i]
		}
	}
	return out
}

func mropePairComponent(pair int32, sections [4]int32, interleaved bool) int {
	if interleaved {
		if pair%3 == 1 && pair < 1+3*sections[1] {
			return 1
		}
		if pair%3 == 2 && pair < 2+3*sections[2] {
			return 2
		}
		if pair%3 == 0 && pair < 3*sections[0] {
			return 0
		}
		return 3
	}

	secW := sections[0] + sections[1]
	secE := secW + sections[2]
	switch {
	case pair < sections[0]:
		return 0
	case pair < secW:
		return 1
	case pair < secE:
		return 2
	default:
		return 3
	}
}

func (m *Model) buildPromptMRoPEPositions(state *promptVisionState, startPos, chunkLen int32) [4][]int32 {
	var positions [4][]int32
	for i := range positions {
		positions[i] = make([]int32, chunkLen)
	}

	// positions[3] stays zero — it covers RoPE dims beyond the 3 MRoPE sections.
	for i := range chunkLen {
		p := mapPromptPosition(state, startPos+i)
		positions[0][i] = p
		positions[1][i] = p
		positions[2][i] = p
	}

	merge := int32(1)
	if m != nil && m.Config != nil && m.Config.Vision != nil {
		merge = m.Config.Vision.SpatialMergeSize
	}
	for _, span := range state.Spans {
		if span.Grid == nil {
			continue
		}

		chunkLo, chunkHi, spanLo, _, ok := overlapRange(startPos, chunkLen, span.Start, span.End)
		if !ok {
			continue
		}

		w := max(int32(1), span.Grid.Width/merge)
		for i := chunkLo; i < chunkHi; i++ {
			rel := spanLo + (i - chunkLo)
			positions[1][i] += rel / w
			positions[2][i] += rel % w
		}
	}

	return positions
}

func (m *Model) buildPromptMRoPECosSin(state *promptVisionState, startPos, chunkLen int32, dtype mlx.DType) (*mlx.Array, *mlx.Array) {
	if m == nil || m.Config == nil || state == nil || chunkLen <= 0 || len(m.Config.MRoPESections) == 0 {
		return nil, nil
	}

	ropeDim := m.Config.RopeDim
	if ropeDim%2 != 0 {
		ropeDim--
	}
	if ropeDim <= 0 {
		return nil, nil
	}

	half := ropeDim / 2
	positions := m.buildPromptMRoPEPositions(state, startPos, chunkLen)
	sections := normalizeMRoPESections(m.Config.MRoPESections)
	theta := m.Config.RopeTheta
	if theta <= 0 {
		theta = 100000.0
	}

	freqs := make([]float64, half)
	for j := range half {
		freqs[j] = math.Pow(float64(theta), -2.0*float64(j)/float64(ropeDim))
	}

	angles := make([]float32, chunkLen*ropeDim)
	for i := range chunkLen {
		base := i * ropeDim
		for j := range half {
			component := mropePairComponent(j, sections, m.Config.MRoPEInterleaved)
			angle := float32(float64(positions[component][i]) * freqs[j])
			angles[base+j] = angle
			angles[base+half+j] = angle
		}
	}

	arr := mlx.FromValues(angles, 1, 1, int(chunkLen), int(ropeDim))
	cos := mlx.Cos(arr)
	sin := mlx.Sin(arr)
	if dtype != 0 {
		cos = cos.AsType(dtype)
		sin = sin.AsType(dtype)
	}
	return cos, sin
}

func (m *Model) tokenizePromptWithResolvedImages(
	prompt string,
	images []base.ImageInput,
	resolve func([]byte) (*VisionEmbeddings, error),
) ([]int32, *promptVisionState, error) {
	if m == nil || m.tok == nil {
		return nil, nil, fmt.Errorf("qwen3_5: tokenizer not initialized")
	}

	if m.Vision == nil || m.ImageProcessor == nil || resolve == nil {
		return m.tok.Encode(prompt, true), nil, nil
	}

	parts := imageTagRE.Split(prompt, -1)
	matches := imageTagRE.FindAllStringSubmatch(prompt, -1)

	resolved := make(map[int]*VisionEmbeddings, len(images))
	var out []int32
	state := &promptVisionState{}
	var p int32
	appendToken := func(tok, pos int32) {
		out = append(out, tok)
		state.PositionCache = append(state.PositionCache, pos)
	}
	for i, part := range parts {
		for _, tok := range m.tok.Encode(part, i == 0) {
			appendToken(tok, p)
			p++
		}

		if i >= len(matches) {
			continue
		}

		imageID, err := strconv.Atoi(matches[i][1])
		if err != nil {
			return nil, nil, fmt.Errorf("qwen3_5: invalid image tag %q: %w", matches[i][0], err)
		}

		img, ok := findImageByID(images, imageID)
		if !ok {
			return nil, nil, fmt.Errorf("invalid image index: %d", imageID)
		}

		embeds := resolved[imageID]
		if embeds == nil {
			embeds, err = resolve(img.Data)
			if err != nil {
				return nil, nil, err
			}
			resolved[imageID] = embeds
		}
		if embeds == nil || embeds.Main == nil || !embeds.Main.Valid() || embeds.Main.NumDims() < 2 {
			return nil, nil, fmt.Errorf("qwen3_5: invalid vision embeddings")
		}

		tokensPerImage := int32(embeds.Main.Dim(1))
		if tokensPerImage <= 0 {
			return nil, nil, fmt.Errorf("qwen3_5: invalid image token count: %d", tokensPerImage)
		}

		appendToken(m.VisionStartToken, p)
		p++
		basePos := p
		spanStart := int32(len(out))
		for range tokensPerImage {
			appendToken(m.ImageTokenID, basePos)
		}
		spanEnd := int32(len(out))
		merge := int32(1)
		if m.Config != nil && m.Config.Vision != nil {
			merge = m.Config.Vision.SpatialMergeSize
		}
		gridSpan := promptVisionGridSpan(embeds.Grid, merge, tokensPerImage)
		p += gridSpan
		appendToken(m.VisionEndToken, p)
		p++

		state.Spans = append(state.Spans, promptVisionSpan{
			Start: spanStart,
			End:   spanEnd,
			Main:  embeds.Main,
			Grid:  embeds.Grid,
		})
	}

	return out, state, nil
}

func (m *Model) TokenizePromptWithImagesState(prompt string, images []base.ImageInput) (*base.PromptTokenization, error) {
	tokens, state, err := m.tokenizePromptWithResolvedImages(prompt, images, m.EncodeVisionImage)
	if err != nil {
		return nil, err
	}
	return &base.PromptTokenization{Tokens: tokens, State: state}, nil
}
