package video

import (
	"bytes"
	"context"
	"encoding/binary"
	"image"
	"image/color"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/video/videotest"
)

// boxClip is 2 s of 96x64 at 10 fps with a 16 kHz tone.
var boxClip = videotest.Clip{Width: 96, Height: 64, Frames: 20, FPS: 10, AudioRate: 16000}

// clipBytes encodes c.
func clipBytes(t testing.TB, c videotest.Clip) []byte {
	t.Helper()
	data, err := c.Bytes()
	if err != nil {
		t.Fatal(err)
	}
	return data
}

// writeClip saves c under a temporary directory with no extension, like the
// runner's spooled copies.
func writeClip(t testing.TB, c videotest.Clip) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "clip")
	if err := os.WriteFile(path, clipBytes(t, c), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

// rotated90 displays a clip rotated 90° counterclockwise: stored (x, y)
// shows at (y, width-x).
func rotated90(width int) [9]uint32 {
	minusOne := uint32(0xffff0000)
	return [9]uint32{0, minusOne, 0, 1 << 16, 0, 0, 0, uint32(width) << 16, 1 << 30}
}

func TestOpen(t *testing.T) {
	red := color.NRGBA{255, 0, 0, 255}
	black := color.NRGBA{0, 0, 0, 255}
	silent := boxClip
	silent.AudioRate = 0
	rotated := silent
	rotated.Matrix = rotated90(silent.Width)
	for _, tc := range []struct {
		name string
		clip videotest.Clip
		meta Meta
		// display maps a stored pixel to where it is displayed
		display func(image.Point) image.Point
	}{
		{"plain", boxClip, Meta{Frames: 20, FPS: 10, Duration: 2, Width: 96, Height: 64, Audio: true}, func(p image.Point) image.Point { return p }},
		{"silent", silent, Meta{Frames: 20, FPS: 10, Duration: 2, Width: 96, Height: 64}, func(p image.Point) image.Point { return p }},
		{"rotated", rotated, Meta{Frames: 20, FPS: 10, Duration: 2, Width: 64, Height: 96}, func(p image.Point) image.Point {
			return image.Pt(p.Y, silent.Width-1-p.X)
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f, err := Open(writeClip(t, tc.clip))
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()
			if got := f.Meta(); got != tc.meta {
				t.Fatalf("got meta %+v, want %+v", got, tc.meta)
			}

			var got []int
			err = f.Frames(context.Background(), []int{0, 7, 19, 19}, func(fr Frame) error {
				got = append(got, fr.Index)
				if b := fr.Image.Bounds(); b.Dx() != tc.meta.Width || b.Dy() != tc.meta.Height {
					t.Errorf("frame %d is %v, want %dx%d", fr.Index, b, tc.meta.Width, tc.meta.Height)
				}
				// The box's center, below it, and beside it pin the frame.
				x, boxSize := tc.clip.BoxAt(fr.Index), tc.clip.BoxSize()
				beside := image.Pt(x+boxSize+8, boxSize/2)
				if x >= 16 {
					beside = image.Pt(x-8, boxSize/2)
				}
				for _, want := range []struct {
					at image.Point
					c  color.NRGBA
				}{
					{image.Pt(x+boxSize/2, boxSize/2), red},
					{image.Pt(x+boxSize/2, boxSize+16), black},
					{beside, black},
				} {
					at := tc.display(want.at)
					if c := fr.Image.NRGBAAt(at.X, at.Y); !near(c, want.c) {
						t.Errorf("frame %d pixel %v = %v, want about %v", fr.Index, at, c, want.c)
					}
				}
				return nil
			})
			if err != nil {
				t.Fatal(err)
			}
			if want := []int{0, 7, 19, 19}; !slices.Equal(got, want) {
				t.Fatalf("got frames %v, want %v", got, want)
			}

			pcm, rate, err := f.PCM(600)
			if err != nil {
				t.Fatal(err)
			}
			if !tc.meta.Audio {
				if pcm != nil {
					t.Fatalf("got %d samples from a clip without audio", len(pcm))
				}
				return
			}
			if d := float64(len(pcm)) / float64(rate); math.Abs(d-tc.meta.Duration) > 0.05 {
				t.Errorf("got %.3fs of audio, want about %.0fs", d, tc.meta.Duration)
			}
		})
	}
}

func TestFramesRejectsUnorderedIndices(t *testing.T) {
	f, err := Open(writeClip(t, boxClip))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	for _, indices := range [][]int{{1, 0}, {-1}, {25}} {
		if err := f.Frames(context.Background(), indices, func(Frame) error { return nil }); err == nil {
			t.Errorf("Frames(%v) succeeded", indices)
		}
	}
}

// TestPCMLimit checks that the soundtrack decode stops at the limit: the
// audio track can run longer than the video the metadata describes.
func TestPCMLimit(t *testing.T) {
	f, err := Open(writeClip(t, boxClip))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if _, _, err := f.PCM(1); err == nil || !strings.Contains(err.Error(), "longer than 1 seconds") {
		t.Fatalf("PCM of a 2 s soundtrack with a 1 s limit = %v", err)
	}
	if pcm, _, err := f.PCM(3); err != nil || len(pcm) == 0 {
		t.Fatalf("PCM within the limit = %d samples, %v", len(pcm), err)
	}
}

// TestPCMDownmix checks that a stereo soundtrack comes back as the mean of
// its channels: the tone is on the left only, so mono carries half of it.
func TestPCMDownmix(t *testing.T) {
	peak := func(c videotest.Clip) (float32, int) {
		f, err := Open(writeClip(t, c))
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		pcm, _, err := f.PCM(3)
		if err != nil {
			t.Fatal(err)
		}
		var p float32
		for _, v := range pcm {
			p = max(p, v)
		}
		return p, len(pcm)
	}
	stereo := boxClip
	stereo.Stereo = true
	monoPeak, monoLen := peak(boxClip)
	stereoPeak, stereoLen := peak(stereo)
	if stereoLen != monoLen || math.Abs(float64(stereoPeak)-float64(monoPeak)/2) > 1e-3 {
		t.Fatalf("stereo downmix: %d samples peaking at %.4f; mono %d samples peaking at %.4f", stereoLen, stereoPeak, monoLen, monoPeak)
	}
}

// TestOpenRejectsMirroredTracks checks that only quarter-turn display
// transforms are applied; a mirror is not a rotation.
func TestOpenRejectsMirroredTracks(t *testing.T) {
	mirrored := boxClip
	mirrored.Matrix = videotest.Identity
	mirrored.Matrix[0] = 0xffff0000 // a = -1
	if f, err := Open(writeClip(t, mirrored)); err == nil {
		f.Close()
		t.Fatal("opened a horizontally mirrored track")
	}
}

// FuzzOpen feeds the system decoder files that pass the container sniff.
// Besides the clean clips, the seeds corrupt headers the decoder reads: a
// perspective term in the track matrix and zero timescales.
func FuzzOpen(f *testing.F) {
	rotated := boxClip
	rotated.AudioRate = 0
	rotated.Matrix = rotated90(rotated.Width)
	f.Add(clipBytes(f, boxClip))
	f.Add(clipBytes(f, rotated))

	perspective := boxClip
	perspective.Matrix = videotest.Identity
	perspective.Matrix[8] = 0x43000000
	f.Add(clipBytes(f, perspective))
	for _, header := range []string{"mvhd", "mdhd"} {
		data := clipBytes(f, boxClip)
		// The timescale follows the version, flags and two timestamps.
		at := bytes.Index(data, []byte(header)) + 4 + 12
		binary.BigEndian.PutUint32(data[at:], 0)
		f.Add(data)
	}

	f.Fuzz(func(t *testing.T, data []byte) {
		if !IsVideo(data) {
			return
		}
		path := filepath.Join(t.TempDir(), "clip")
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatal(err)
		}
		v, err := Open(path)
		if err != nil {
			return
		}
		defer v.Close()
		m := v.Meta()
		if m.Frames <= 0 || m.Width <= 0 || m.Height <= 0 || !(m.FPS > 0) || !(m.Duration > 0) {
			t.Fatalf("opened with meta %+v", m)
		}
		_ = v.Frames(t.Context(), []int{0, m.Frames - 1}, func(fr Frame) error {
			if b := fr.Image.Bounds(); b.Dx() != m.Width || b.Dy() != m.Height {
				t.Fatalf("frame %d is %v, meta says %dx%d", fr.Index, b, m.Width, m.Height)
			}
			return nil
		})
		_, _, _ = v.PCM(600)
	})
}

// TestBGRAToNRGBA checks each display rotation by where two source corners
// land: top-left (red) and top-right (green).
func TestBGRAToNRGBA(t *testing.T) {
	const w, h = 3, 2
	src := make([]byte, 4*w*h)
	src[2] = 255         // top-left red (BGRA)
	src[4*(w-1)+1] = 255 // top-right green
	for _, tc := range []struct {
		degrees           int
		topLeft, topRight image.Point
	}{
		{0, image.Pt(0, 0), image.Pt(w-1, 0)},
		{90, image.Pt(h-1, 0), image.Pt(h-1, w-1)},
		{180, image.Pt(w-1, h-1), image.Pt(0, h-1)},
		{270, image.Pt(0, w-1), image.Pt(0, 0)},
	} {
		img := bgraToNRGBA(src, w, h, 4*w, tc.degrees)
		if c := img.NRGBAAt(tc.topLeft.X, tc.topLeft.Y); c.R != 255 || c.G != 0 {
			t.Errorf("%d°: top-left landed elsewhere, %v at %v", tc.degrees, c, tc.topLeft)
		}
		if c := img.NRGBAAt(tc.topRight.X, tc.topRight.Y); c.G != 255 || c.R != 0 {
			t.Errorf("%d°: top-right landed elsewhere, %v at %v", tc.degrees, c, tc.topRight)
		}
	}
}

func near(a, b color.NRGBA) bool {
	d := func(x, y uint8) bool { return max(x, y)-min(x, y) < 24 }
	return d(a.R, b.R) && d(a.G, b.G) && d(a.B, b.B)
}
