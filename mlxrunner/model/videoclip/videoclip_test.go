package videoclip

import (
	"context"
	"encoding/binary"
	"errors"
	"image"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/ollama/ollama/video"
)

// clipConfig is the shipped video_preprocessor_config.json of LFM3-Omni;
// Qwen3.5 differs only in MaxFrames (768).
var clipConfig = Config{FPS: 2, MinFrames: 4, MaxFrames: 384, MinPixels: 4096, MaxPixels: 25165824}

func TestSampleFrames(t *testing.T) {
	for _, tc := range []struct {
		name  string
		meta  video.Meta
		first []int
		n     int
	}{
		// 300 frames at 30 fps: 2 fps gives 20, spread over 0..299.
		{"10s", video.Meta{Frames: 300, FPS: 30}, []int{0, 16, 31, 47, 63, 79}, 20},
		// Fewer frames than MinFrames at 2 fps are raised to MinFrames.
		{"short", video.Meta{Frames: 33, FPS: 25}, []int{0, 11, 21, 32}, 4},
		// Long clips cap at MaxFrames.
		{"long", video.Meta{Frames: 2000, FPS: 10}, []int{0, 5, 10, 16, 21, 26}, 384},
		// Never more frames than the clip has.
		{"tiny", video.Meta{Frames: 3, FPS: 1}, []int{0, 1, 2}, 3},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := clipConfig.sampleFrames(tc.meta)
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != tc.n || !slices.Equal(got[:len(tc.first)], tc.first) || got[len(got)-1] != tc.meta.Frames-1 {
				t.Fatalf("got %d frames %v..., want %d starting %v", len(got), got[:min(len(got), 6)], tc.n, tc.first)
			}
		})
	}
	if _, err := clipConfig.sampleFrames(video.Meta{}); err == nil {
		t.Error("sampled a video with no frames")
	}
}

// The expected sizes are the reference processor's video_grid_thw times the
// 16-pixel patch for clips of these shapes.
func TestResizeGrid(t *testing.T) {
	for _, tc := range []struct {
		name    string
		n, h, w int
		wantH   int32
		wantW   int32
	}{
		{"within budget", 20, 360, 640, 352, 640},
		{"half rounds to even", 384, 144, 256, 128, 256}, // 144/32 = 4.5
		{"few frames", 4, 270, 480, 256, 480},
		{"rounds up to the factor", 14, 240, 320, 256, 320},
		{"over budget scales down", 20, 1080, 1920, 832, 1472},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h, w, err := resizeGrid(tc.n, int32(tc.h), int32(tc.w), 32, clipConfig.MinPixels, clipConfig.MaxPixels)
			if err != nil {
				t.Fatal(err)
			}
			if h != tc.wantH || w != tc.wantW {
				t.Errorf("got %dx%d, want %dx%d", h, w, tc.wantH, tc.wantW)
			}
			if got := int64(tc.n) * int64(h) * int64(w); got > clipConfig.MaxPixels {
				t.Errorf("got %d pixels, want at most %d", got, clipConfig.MaxPixels)
			}
		})
	}
	for _, tc := range []struct {
		name    string
		n, h, w int
	}{
		{"one frame", 1, 360, 640},
		{"extreme aspect ratio", 4, 32, 7000},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, _, err := resizeGrid(tc.n, int32(tc.h), int32(tc.w), 32, clipConfig.MinPixels, clipConfig.MaxPixels); err == nil {
				t.Error("got no error")
			}
		})
	}
}

// TestNewPlan checks a plan against the reference processor's prompt for a
// 10 s, 30 fps, 640x360 clip.
func TestNewPlan(t *testing.T) {
	plan, err := NewPlan(video.Meta{Frames: 300, FPS: 30, Width: 640, Height: 360}, clipConfig, 32)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Height != 352 || plan.Width != 640 {
		t.Fatalf("got %dx%d, want 352x640", plan.Height, plan.Width)
	}
	var labels []string
	for _, g := range plan.Groups {
		labels = append(labels, g.Label())
	}
	want := []string{
		"<0.3 seconds>", "<1.3 seconds>", "<2.4 seconds>", "<3.4 seconds>", "<4.5 seconds>",
		"<5.5 seconds>", "<6.6 seconds>", "<7.6 seconds>", "<8.7 seconds>", "<9.7 seconds>",
	}
	if !slices.Equal(labels, want) {
		t.Errorf("got labels %v, want %v", labels, want)
	}

	// An odd sample count repeats the last frame.
	plan, err = NewPlan(video.Meta{Frames: 3, FPS: 1, Width: 64, Height: 64}, clipConfig, 32)
	if err != nil {
		t.Fatal(err)
	}
	if got := plan.Groups[len(plan.Groups)-1].Frames; got != [2]int{2, 2} {
		t.Errorf("got last group %v, want [2 2]", got)
	}
}

// fakeFile is a video whose frame i is filled with gray level 2i, so packed
// pixels show which frames were decoded. It counts decoder starts.
type fakeFile struct {
	meta   video.Meta
	starts *atomic.Int32
}

func (f fakeFile) Meta() video.Meta { return f.meta }
func (f fakeFile) Frames(ctx context.Context, indices []int, fn func(video.Frame) error) error {
	f.starts.Add(1)
	for _, i := range indices {
		if err := ctx.Err(); err != nil {
			return err
		}
		img := image.NewNRGBA(image.Rect(0, 0, f.meta.Width, f.meta.Height))
		for p := 0; p < len(img.Pix); p += 4 {
			img.Pix[p], img.Pix[p+1], img.Pix[p+2], img.Pix[p+3] = uint8(2*i), uint8(2*i), uint8(2*i), 255
		}
		if err := fn(video.Frame{Index: i, Image: img}); err != nil {
			return err
		}
	}
	return nil
}
func (fakeFile) PCM(int) ([]float32, int, error) { return nil, 0, nil }
func (fakeFile) Close() error                    { return nil }

func TestNewLimits(t *testing.T) {
	var starts atomic.Int32
	if _, err := NewPlan(video.Meta{Frames: 4, FPS: 1e-9, Width: 64, Height: 64}, clipConfig, 32); err == nil {
		t.Error("planned a clip whose frame rate rounds to zero")
	}

	// A cancelled request does not start the decoder.
	clip, err := newClip(fakeFile{video.Meta{Frames: 20, FPS: 10, Duration: 2, Width: 64, Height: 64}, &starts}, clipConfig, 16, 2)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = clip.groupPixels(ctx, 0)
	clip.Close()
	if !errors.Is(err, context.Canceled) || starts.Load() != 0 {
		t.Errorf("groupPixels on a cancelled context = %v after %d decoder starts", err, starts.Load())
	}
}

// TestGroupPixelsResumes checks that prefill resuming at another group, as
// after a partial prefix-cache hit, gets that group's frames.
func TestGroupPixelsResumes(t *testing.T) {
	var starts atomic.Int32
	meta := video.Meta{Frames: 100, FPS: 10, Duration: 10, Width: 64, Height: 64}
	pixels := func(order []int) map[int][]float32 {
		clip, err := newClip(fakeFile{meta, &starts}, clipConfig, 16, 2)
		if err != nil {
			t.Fatal(err)
		}
		defer clip.Close()
		got := map[int][]float32{}
		for _, g := range order {
			px, err := clip.groupPixels(context.Background(), g)
			if err != nil {
				t.Fatal(err)
			}
			got[g] = px
		}
		return got
	}
	sequential := pixels([]int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
	resumed := pixels([]int{5, 6, 2, 3, 9})
	for g, px := range resumed {
		if !slices.Equal(px, sequential[g]) {
			t.Errorf("group %d resumed out of order differs from a sequential decode", g)
		}
	}
	if slices.Equal(sequential[0], sequential[5]) {
		t.Fatal("groups are indistinguishable; the check proves nothing")
	}
}

// TestPixelGoldens compares our decoded, resized and packed frame groups
// with the reference processor's pixel_values_videos. OLLAMA_VIDEO_FIXTURES
// names a directory with fixtures/ and goldens/.
func TestPixelGoldens(t *testing.T) {
	dir := os.Getenv("OLLAMA_VIDEO_FIXTURES")
	if dir == "" {
		t.Skip("OLLAMA_VIDEO_FIXTURES not set")
	}
	for _, name := range []string{"clip10s.mp4", "short1.3s.mp4", "rot90.mov"} {
		t.Run(name, func(t *testing.T) {
			stem := strings.TrimSuffix(name, filepath.Ext(name))
			want, shape := readNPY(t, filepath.Join(dir, "goldens", stem+".v.pixels.npy"))
			f, err := video.Open(filepath.Join(dir, "fixtures", name))
			if err != nil {
				t.Fatal(err)
			}
			clip, err := newClip(f, clipConfig, 16, 2)
			if err != nil {
				t.Fatal(err)
			}
			defer clip.Close()
			var got []float32
			for g := range clip.Plan.Groups {
				px, err := clip.groupPixels(context.Background(), g)
				if err != nil {
					t.Fatal(err)
				}
				got = append(got, px...)
			}
			if len(got) != len(want) || shape[1] != 1536 {
				t.Fatalf("got %d values, reference %v", len(got), shape)
			}
			var sum, bias float64
			for i := range got {
				d := float64(got[i] - want[i])
				sum += math.Abs(d)
				bias += d
			}
			n := float64(len(got))
			t.Logf("mean abs %.4f (%.2f levels), bias %.4f", sum/n, sum/n*127.5, bias/n)
			// Decoders differ by about 2 levels in chroma at sharp edges.
			if sum/n*127.5 > 6 || math.Abs(bias/n*127.5) > 1.5 {
				t.Errorf("pixels differ from the reference: mean abs %.2f levels, bias %.2f", sum/n*127.5, bias/n*127.5)
			}
		})
	}
}

// readNPY reads a little-endian float32 .npy file.
func readNPY(t *testing.T, path string) ([]float32, []int) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	headerLen := int(binary.LittleEndian.Uint16(data[8:10]))
	header := string(data[10 : 10+headerLen])
	if !strings.Contains(header, "'<f4'") || strings.Contains(header, "True") {
		t.Fatalf("unsupported npy header %q", header)
	}
	var shape []int
	dims := header[strings.Index(header, "(")+1 : strings.Index(header, ")")]
	for d := range strings.SplitSeq(dims, ",") {
		if d = strings.TrimSpace(d); d != "" {
			n, err := strconv.Atoi(d)
			if err != nil {
				t.Fatal(err)
			}
			shape = append(shape, n)
		}
	}
	body := data[10+headerLen:]
	out := make([]float32, len(body)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(body[4*i:]))
	}
	return out, shape
}
