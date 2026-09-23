package gemma4

import (
	"bytes"
	"encoding/binary"
	"math"
	"strings"
	"testing"
)

func wavPCM16(samples int) []byte {
	var b bytes.Buffer
	b.WriteString("RIFF")
	binary.Write(&b, binary.LittleEndian, uint32(36+2*samples))
	b.WriteString("WAVE")
	b.WriteString("fmt ")
	binary.Write(&b, binary.LittleEndian, uint32(16))
	binary.Write(&b, binary.LittleEndian, uint16(1))
	binary.Write(&b, binary.LittleEndian, uint16(1))
	binary.Write(&b, binary.LittleEndian, uint32(audioSampleRate))
	binary.Write(&b, binary.LittleEndian, uint32(2*audioSampleRate))
	binary.Write(&b, binary.LittleEndian, uint16(2))
	binary.Write(&b, binary.LittleEndian, uint16(16))
	b.WriteString("data")
	binary.Write(&b, binary.LittleEndian, uint32(2*samples))
	b.Write(make([]byte, 2*samples))
	return b.Bytes()
}

// Golden log-mel values from the reference Gemma4AudioFeatureExtractor over
// 1664 samples of 0.5*sin(2*pi*440*t/16000): bins 0, 1, 64, 127 per frame.
var melGolden = [][4]float32{
	{-6.907755, 0.808556, -1.241259, -2.153034},
	{-6.907755, -4.019087, -6.397553, -6.901781},
	{-6.907755, -4.861755, -6.335430, -6.902662},
	{-6.907755, -4.861755, -6.335430, -6.902662},
	{-6.907755, -4.019087, -6.397553, -6.901781},
	{-6.907755, -3.820103, -6.459379, -6.901654},
	{-6.907755, -4.019087, -6.397553, -6.901781},
	{-6.907755, -4.861755, -6.335430, -6.902662},
	{-6.907755, -4.861755, -6.335430, -6.902662},
	{-6.907755, -4.019087, -6.397553, -6.901781},
}

func TestMelSpectrogramGolden(t *testing.T) {
	samples := make([]float32, 1664)
	for i := range samples {
		samples[i] = float32(0.5 * math.Sin(2*math.Pi*440*float64(i)/audioSampleRate))
	}

	mel, frames := melSpectrogram(samples)
	if frames != len(melGolden) {
		t.Fatalf("%d frames, want %d", frames, len(melGolden))
	}
	for f, want := range melGolden {
		for i, bin := range []int{0, 1, 64, 127} {
			got := mel[f*audioMelBins+bin]
			if diff := float64(got - want[i]); math.Abs(diff) > 2e-6 {
				t.Errorf("frame %d bin %d: %v, want %v", f, bin, got, want[i])
			}
		}
	}
}

func TestProcessAudioTokenCounts(t *testing.T) {
	cases := []struct {
		name    string
		samples int
		frames  []int
		tokens  []int
	}{
		{"minimum", 161, []int{1}, []int{1}},
		{"short", 1000, []int{6}, []int{2}},
		{"thirty seconds", 480000, []int{2999}, []int{750}},
		// Over the limit the clip splits evenly; each chunk must still fit.
		{"just over the limit", 480001, []int{-1, -1}, []int{-1, -1}},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			chunks, err := processAudio(wavPCM16(tt.samples))
			if err != nil {
				t.Fatal(err)
			}
			if len(chunks) != len(tt.frames) {
				t.Fatalf("%d chunks, want %d", len(chunks), len(tt.frames))
			}
			for i, c := range chunks {
				if tt.frames[i] < 0 {
					if c.frames <= 0 || c.frames > 2999 || c.numTokens > 750 {
						t.Errorf("chunk %d: frames %d tokens %d exceed one 30 s chunk", i, c.frames, c.numTokens)
					}
				} else if c.frames != tt.frames[i] || c.numTokens != tt.tokens[i] {
					t.Errorf("chunk %d: frames %d tokens %d, want %d %d",
						i, c.frames, c.numTokens, tt.frames[i], tt.tokens[i])
				}
				if len(c.data) != c.frames*audioMelBins {
					t.Errorf("chunk %d: %d mel values for %d frames", i, len(c.data), c.frames)
				}
			}
		})
	}
}

func TestProcessAudioTooShort(t *testing.T) {
	_, err := processAudio(wavPCM16(160))
	if err == nil || !strings.Contains(err.Error(), "audio too short") {
		t.Fatalf("error %v, want audio too short", err)
	}
}
