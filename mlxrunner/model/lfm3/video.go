package lfm3

import (
	"context"
	"errors"
	"fmt"
	"math"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/model/audio"
	"github.com/ollama/ollama/mlxrunner/model/lfm3/lfm3audio"
	"github.com/ollama/ollama/mlxrunner/model/videoclip"
)

// maxSoundtrackSeconds is the audio input limit: the whole soundtrack is
// decoded and encoded at once.
const maxSoundtrackSeconds = 600

// soundtrackSpans splits a soundtrack of audioTokens encoder outputs over a
// clip's groups: each group is followed by the tokens from its first frame
// up to the next group's.
func soundtrackSpans(plan videoclip.Plan, audioTokens int, secondsPerAudioToken float64) [][2]int {
	if audioTokens == 0 {
		return nil
	}
	spans := make([][2]int, len(plan.Groups))
	bound := 0
	for g := range plan.Groups {
		end := audioTokens
		if g+1 < len(plan.Groups) {
			start := float64(plan.Groups[g+1].Frames[0]) / plan.FPS
			end = min(audioTokens, int(math.RoundToEven(start/secondsPerAudioToken)))
		}
		end = max(end, bound)
		spans[g] = [2]int{bound, end}
		bound = end
	}
	return spans
}

// videoClip is one video in a request: the shared clip and the soundtrack
// its groups share. Only the goroutine running the request uses it.
type videoClip struct {
	*videoclip.Clip
	audio [][2]int // each group's soundtrack span; nil without one

	samples []float32 // soundtrack at the model rate; nil without one
	fe      *lfm3audio.FeatureExtractor
	mel     *lfm3audio.Features
	scope   *mlx.Scope
	encoded *mlx.Array // the whole soundtrack, encoded once, held until release
}

// preparedSoundtrack is one group's span of its clip's soundtrack.
type preparedSoundtrack struct {
	clip   *videoClip
	Start  int
	Tokens int32
}

// soundtrackFeatures extracts the soundtrack's features once, off the
// prepare path: a request served from the prefix cache never needs them.
func (c *videoClip) soundtrackFeatures() {
	if c.mel == nil && c.encoded == nil {
		f := c.fe.Extract(c.samples)
		c.mel = &f
	}
}

// soundtrack returns rows [start, start+n) of the encoded soundtrack,
// building the encoder graph on first use and holding it until release.
func (c *videoClip) soundtrack(enc *lfm3audio.Encoder, start, n int) *mlx.Array {
	if c.encoded == nil {
		c.encoded = enc.Forward(lfm3audio.FeatureArray(*c.mel))
		c.scope.Attach(c.encoded)
		c.mel = nil
	}
	return c.encoded.Slice(mlx.Slice(start, start+n), mlx.Slice())
}

func (c *videoClip) release() {
	c.Close()
	c.scope.Close()
}

// prepareVideo plans a video segment and appends its expansion to tokens:
// one item per frame group and, with a soundtrack, one per group's audio
// span, all slicing a single soundtrack encode.
func (m *VLModel) prepareVideo(segment model.Segment, source int, tokens []int32) ([]int32, []model.PreparedItem, func(), error) {
	v := m.Vision
	if v == nil || v.Video == nil {
		return nil, nil, nil, errors.New("this LFM3 model does not support video input")
	}
	c, err := videoclip.Open(segment.Data, *v.Video, v.Config.PatchSize, v.Config.SpatialMergeSize)
	if err != nil {
		return nil, nil, nil, err
	}
	clip := &videoClip{Clip: c, scope: mlx.NewScope()}
	fail := func(err error) ([]int32, []model.PreparedItem, func(), error) {
		clip.release()
		return nil, nil, nil, err
	}

	if m.Audio != nil && c.File.Meta().Audio {
		// The whole soundtrack is held in memory, so its decode is bounded.
		pcm, rate, err := c.File.PCM(maxSoundtrackSeconds)
		if err != nil {
			return fail(fmt.Errorf("decode soundtrack: %w", err))
		}
		cfg := m.Audio.cfg
		clip.samples = audio.Resample(pcm, rate, cfg.SampleRate)
		clip.fe = m.Audio.fe
		if len(clip.samples) > 0 {
			secondsPerToken := float64(cfg.HopLength*cfg.EncoderStride) / float64(cfg.SampleRate)
			clip.audio = soundtrackSpans(c.Plan, cfg.TokenCount(len(clip.samples)), secondsPerToken)
		}
	}
	tokens, items := m.expandVideo(clip, source, tokens)
	return tokens, items, clip.release, nil
}

// expandVideo appends a planned clip's expansion to tokens: the clip and
// each frame group inside vision markers, each group led by its timestamp
// and followed by its soundtrack span. It returns one item per group and
// one per non-empty soundtrack span.
func (m *VLModel) expandVideo(clip *videoClip, source int, tokens []int32) ([]int32, []model.PreparedItem) {
	gridH, gridW := clip.Grid()
	perGroup := clip.Tokens()
	var items []model.PreparedItem
	tokens = append(tokens, visionStartTokenID)
	for g, group := range clip.Plan.Groups {
		tokens = append(tokens, m.tok.Encode(group.Label(), false)...)
		start := len(tokens)
		tokens = append(tokens, visionStartTokenID)
		for range perGroup {
			tokens = append(tokens, m.Vision.VideoTokenID)
		}
		tokens = append(tokens, visionEndTokenID)
		items = append(items, clip.Item(g, source, start, len(tokens), preparedImage{Height: gridH, Width: gridW, Tokens: int32(perGroup)}))
		if clip.audio == nil {
			continue
		}
		if span := clip.audio[g]; span[1] > span[0] {
			n := span[1] - span[0]
			start := len(tokens)
			for range n {
				tokens = append(tokens, audioTokenID)
			}
			items = append(items, model.PreparedItem{
				Range:  [2]int{start, len(tokens)},
				Source: source,
				Dims:   []int{span[0], n}, // identity only: the encode is shared
				// Extracts the shared soundtrack's features; each span encodes from them.
				Pixels: func(context.Context) ([]float32, error) { clip.soundtrackFeatures(); return nil, nil },
				Opaque: preparedSoundtrack{clip: clip, Start: span[0], Tokens: int32(n)},
				Causal: true,
			})
		}
	}
	return append(tokens, visionEndTokenID), items
}
