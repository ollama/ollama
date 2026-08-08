package mlxrunner

// media.go expands the renderer's [img-N] markers into model-specific image
// token runs. Text segments tokenize independently — boi/image/eoi are
// special tokens, which break merges at exactly these boundaries, so
// per-segment encoding matches whole-string encoding.

import (
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"regexp"
	"strconv"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
)

var imgMarker = regexp.MustCompile(`\[img-(\d+)\]`)

// expandedPrompt is a tokenized prompt with media expanded in place.
type expandedPrompt struct {
	Tokens []int32
	Spans  [][2]int32 // [start, end) of each image's soft-token block
	Salts  []uint32   // per-token prefix-cache salt; 0 for text positions
	Inputs []base.VisionInput
}

// mediaSalts derives one nonzero cache-salt word per soft token from the
// payload digest, position-mixed so distinct images diverge at their first
// soft token and identical images share every key.
func mediaSalts(data []byte, n int) []uint32 {
	sum := sha256.Sum256(data)
	var words [8]uint32
	for i := range words {
		words[i] = binary.BigEndian.Uint32(sum[i*4:])
	}
	salts := make([]uint32, n)
	for i := range salts {
		salts[i] = (words[i%8] ^ (uint32(i) * 0x9E3779B9)) | 1
	}
	return salts
}

// expandMedia tokenizes prompt, replacing each [img-N] marker with
// boi + SoftTokens×image + eoi and preprocessing the matching payload.
// encode is segment tokenization; addBOS applies to the first emitted
// segment only, mirroring the text-only path.
func expandMedia(prompt string, media []llm.MediaData, vm base.VisionModel, opts api.Options,
	encode func(text string, addBOS bool) []int32, addBOS bool) (*expandedPrompt, error) {

	byID := make(map[int]llm.MediaData, len(media))
	for _, m := range media {
		byID[m.ID] = m
	}
	boi, imageToken, eoi := vm.VisionTokens()

	out := &expandedPrompt{}
	first := true
	appendText := func(s string) {
		if s == "" {
			return
		}
		toks := encode(s, addBOS && first)
		first = false
		out.Tokens = append(out.Tokens, toks...)
		out.Salts = append(out.Salts, make([]uint32, len(toks))...)
	}

	rest := prompt
	for {
		loc := imgMarker.FindStringSubmatchIndex(rest)
		if loc == nil {
			appendText(rest)
			break
		}
		appendText(rest[:loc[0]])

		id, err := strconv.Atoi(rest[loc[2]:loc[3]])
		if err != nil {
			return nil, fmt.Errorf("malformed image marker %q", rest[loc[0]:loc[1]])
		}
		m, ok := byID[id]
		if !ok {
			return nil, fmt.Errorf("prompt references [img-%d] but no media with that id was provided", id)
		}
		if m.Kind == llm.MediaKindAudio {
			return nil, errors.New("audio input is not supported on the MLX runner")
		}

		in, err := vm.NewVisionInput(m.Data, opts)
		if err != nil {
			return nil, err
		}
		n := in.SoftTokens()

		first = false
		start := int32(len(out.Tokens) + 1) // soft tokens begin after boi
		out.Tokens = append(out.Tokens, boi)
		out.Salts = append(out.Salts, 0)
		for range n {
			out.Tokens = append(out.Tokens, imageToken)
		}
		out.Salts = append(out.Salts, mediaSalts(m.Data, n)...)
		out.Tokens = append(out.Tokens, eoi)
		out.Salts = append(out.Salts, 0)

		out.Spans = append(out.Spans, [2]int32{start, start + int32(n)})
		out.Inputs = append(out.Inputs, in)
		rest = rest[loc[1]:]
	}
	return out, nil
}
