// Package videoclip plans and decodes video for vision towers that take
// clips as pairs of frames (temporal patches), the layout the Qwen-VL video
// processor defines and LFM3 shares: frames sampled at a fixed rate, one
// resize for the whole clip, and each pair led by its timestamp.
package videoclip

import (
	"context"
	"errors"
	"fmt"
	"image"
	"math"
	"net/http"
	"os"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/video"
)

// maxAspectRatio is the reference resize rule's limit.
const maxAspectRatio = 200

// Config is a reference video processor's sampling and resize policy.
// MinPixels and MaxPixels bound the whole clip (frames x height x width).
type Config struct {
	FPS       float64
	MinFrames int
	MaxFrames int
	MinPixels int64
	MaxPixels int64
}

// Group is one temporal patch: two source frames (the last one repeated
// when the sample count is odd) and the midpoint timestamp the prompt shows.
type Group struct {
	Frames    [2]int
	Timestamp float64
}

// Label is the text that leads a group in the prompt.
func (g Group) Label() string { return fmt.Sprintf("<%.1f seconds>", g.Timestamp) }

type Plan struct {
	Height int32 // resized frame size
	Width  int32
	FPS    float64 // the source rate timestamps are computed at
	Groups []Group
}

// sampleFrames picks the frames the reference samples: the configured rate
// over the clip, clamped to [MinFrames, MaxFrames] and to the frames there
// are, spread evenly from the first frame to the last.
func (c Config) sampleFrames(meta video.Meta) ([]int, error) {
	if meta.Frames <= 0 || !(meta.FPS > 0) {
		return nil, fmt.Errorf("video has no frames")
	}
	n := int(float64(meta.Frames) / meta.FPS * c.FPS)
	n = min(max(n, c.MinFrames), c.MaxFrames, meta.Frames)
	indices := make([]int, n)
	for i := range indices {
		if n > 1 {
			indices[i] = int(math.RoundToEven(float64(i) * float64(meta.Frames-1) / float64(n-1)))
		}
	}
	return indices, nil
}

// resizeGrid sizes every frame of a clip of n frames: the image rule with
// the pixel budget shared by all frames.
func resizeGrid(n int, height, width, factor int32, minPixels, maxPixels int64) (int32, int32, error) {
	const temporal = 2
	if n < temporal {
		return 0, 0, fmt.Errorf("video needs at least %d frames, got %d", temporal, n)
	}
	h, w := float64(height), float64(width)
	f := float64(factor)
	if h < f || w < f {
		scale := max(f/h, f/w)
		h, w = math.Floor(h*scale), math.Floor(w*scale)
	}
	if max(h, w)/min(h, w) > maxAspectRatio {
		return 0, 0, fmt.Errorf("video aspect ratio exceeds %d", maxAspectRatio)
	}
	hBar := math.RoundToEven(h/f) * f
	wBar := math.RoundToEven(w/f) * f
	tBar := math.RoundToEven(float64(n)/temporal) * temporal
	switch {
	case tBar*hBar*wBar > float64(maxPixels):
		beta := math.Sqrt(float64(n) * h * w / float64(maxPixels))
		hBar = max(f, math.Floor(h/beta/f)*f)
		wBar = max(f, math.Floor(w/beta/f)*f)
	case tBar*hBar*wBar < float64(minPixels):
		beta := math.Sqrt(float64(minPixels) / (float64(n) * h * w))
		hBar = math.Ceil(h*beta/f) * f
		wBar = math.Ceil(w*beta/f) * f
	}
	return int32(hBar), int32(wBar), nil
}

// NewPlan lays a clip out the way the reference processor does, with frames
// resized to multiples of factor (patch size times merge size).
func NewPlan(meta video.Meta, cfg Config, factor int32) (Plan, error) {
	indices, err := cfg.sampleFrames(meta)
	if err != nil {
		return Plan{}, err
	}
	height, width, err := resizeGrid(len(indices), int32(meta.Height), int32(meta.Width), factor, cfg.MinPixels, cfg.MaxPixels)
	if err != nil {
		return Plan{}, err
	}
	if len(indices)%2 == 1 {
		indices = append(indices, indices[len(indices)-1])
	}
	// Timestamps use the decoder's average rate at the reference's precision.
	fps := math.Round(meta.FPS*1e6) / 1e6
	if !(fps > 0) || math.IsInf(fps, 0) {
		return Plan{}, fmt.Errorf("video frame rate %v is not usable", meta.FPS)
	}
	plan := Plan{Height: height, Width: width, FPS: fps, Groups: make([]Group, len(indices)/2)}
	for g := range plan.Groups {
		a, b := indices[2*g], indices[2*g+1]
		plan.Groups[g] = Group{Frames: [2]int{a, b}, Timestamp: float64(a+b) / (2 * fps)}
	}
	return plan, nil
}

// readAhead bounds how many groups the decoder prepares ahead of prefill.
const readAhead = 4

// Clip is one video in a request: the opened file, its plan, and the
// read-ahead decoder for its frame groups. Only the goroutine running the
// request uses it.
type Clip struct {
	File  video.File
	Plan  Plan
	patch int32
	merge int32
	path  string // the temporary copy decoders read, deleted on Close

	next    int // the group the decoder delivers next
	groups  chan groupPixels
	stop    context.CancelFunc
	stopped chan struct{}
}

type groupPixels struct {
	pixels []float32
	err    error
}

// Open spools a video to a temporary file, since the system decoders read
// files, and plans it for a tower with the given patch and merge sizes.
func Open(data []byte, cfg Config, patch, merge int32) (*Clip, error) {
	f, err := os.CreateTemp("", "ollama-video-*")
	if err != nil {
		return nil, err
	}
	_, err = f.Write(data)
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		os.Remove(f.Name())
		return nil, err
	}
	file, err := video.Open(f.Name())
	if err != nil {
		os.Remove(f.Name())
		return nil, fmt.Errorf("open video: %w", err)
	}
	c, err := newClip(file, cfg, patch, merge)
	if err != nil {
		os.Remove(f.Name())
		return nil, err
	}
	c.path = f.Name()
	return c, nil
}

// newClip plans an opened video for a tower with the given patch and merge
// sizes. The clip owns file, closing it on error too.
func newClip(file video.File, cfg Config, patch, merge int32) (*Clip, error) {
	plan, err := NewPlan(file.Meta(), cfg, patch*merge)
	if err != nil {
		file.Close()
		return nil, err
	}
	return &Clip{File: file, Plan: plan, patch: patch, merge: merge}, nil
}

// Grid is the pre-merge patch grid of every group.
func (c *Clip) Grid() (height, width int32) {
	return c.Plan.Height / c.patch, c.Plan.Width / c.patch
}

// Tokens is the number of merged tokens each group expands to.
func (c *Clip) Tokens() int {
	h, w := c.Grid()
	return int(h * w / (c.merge * c.merge))
}

// Item returns group g's prepared item over the token range [start, end).
// Its pixels come from the read-ahead decoder when prefill reaches it.
func (c *Clip) Item(g, source, start, end int, opaque any) model.PreparedItem {
	h, w := c.Grid()
	return model.PreparedItem{
		Range:  [2]int{start, end},
		Source: source,
		Dims:   []int{int(h * w), 3 * 2 * int(c.patch*c.patch)},
		Pixels: func(ctx context.Context) ([]float32, error) { return c.groupPixels(ctx, g) },
		Opaque: opaque,
		Causal: true,
	}
}

// groupPixels returns group g's packed frames from the read-ahead decoder,
// restarting it when prefill resumes somewhere else.
func (c *Clip) groupPixels(ctx context.Context, g int) ([]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if c.groups == nil || c.next != g {
		c.startDecoder(g)
	}
	select {
	case p, ok := <-c.groups:
		if !ok {
			return nil, errors.New("video decoder stopped")
		}
		if p.err != nil {
			return nil, p.err
		}
		c.next++
		return p.pixels, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (c *Clip) startDecoder(from int) {
	c.stopDecoder()
	ctx, stop := context.WithCancel(context.Background())
	groups := make(chan groupPixels, readAhead)
	stopped := make(chan struct{})
	c.next, c.groups, c.stop, c.stopped = from, groups, stop, stopped
	go func() {
		defer close(stopped)
		defer close(groups)
		var indices []int
		for _, g := range c.Plan.Groups[from:] {
			indices = append(indices, g.Frames[0], g.Frames[1])
		}
		var pair []*image.NRGBA
		err := c.File.Frames(ctx, indices, func(f video.Frame) error {
			pair = append(pair, ResizeBicubic(f.Image, int(c.Plan.Width), int(c.Plan.Height)))
			if len(pair) < 2 {
				return nil
			}
			p := groupPixels{pixels: PackPatches(pair, c.patch, c.merge)}
			pair = nil
			select {
			case groups <- p:
				return nil
			case <-ctx.Done():
				return ctx.Err()
			}
		})
		if err != nil && ctx.Err() == nil {
			select {
			// The header opened, so a failure here is the client's file.
			case groups <- groupPixels{err: api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: "decode video: " + err.Error()}}:
			case <-ctx.Done():
			}
		}
	}()
}

func (c *Clip) stopDecoder() {
	if c.stop != nil {
		c.stop()
		<-c.stopped
		c.stop, c.groups = nil, nil
	}
}

// Close stops the decoder and deletes the temporary copy.
func (c *Clip) Close() {
	c.stopDecoder()
	c.File.Close()
	if c.path != "" {
		os.Remove(c.path)
	}
}
