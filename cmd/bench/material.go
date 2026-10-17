package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"math/rand/v2"
	"strings"
)

var synthNouns = []string{
	"ledger", "packet", "vector", "window", "record", "buffer", "matrix", "signal",
	"cursor", "bucket", "sensor", "tensor", "stream", "parcel", "socket", "filter",
	"queue", "shard", "token", "frame", "batch", "route", "index", "chunk",
}

var synthVerbs = []string{
	"merge", "split", "score", "rank", "scale", "clamp", "fold", "shift",
	"trim", "pack", "sort", "blend", "probe", "count", "match", "align",
}

// synthCode returns roughly words words of generated Python, deterministic for
// a seed. Every function has its own name, constants and docstring, so the text
// does not repeat the way a tiled corpus would.
func synthCode(words int, seed uint64) string {
	r := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	var b strings.Builder
	used := 0
	for i := 0; used < words; i++ {
		verb := synthVerbs[r.IntN(len(synthVerbs))]
		noun := synthNouns[r.IntN(len(synthNouns))]
		other := synthNouns[r.IntN(len(synthNouns))]
		a, c, k := r.IntN(97)+2, r.IntN(1000), r.IntN(31)+1
		fn := fmt.Sprintf(`def %s_%s_%d(%s, %s, limit=%d):
    """%s each %s in %s against %s, keeping values below %d.

    >>> %s_%s_%d([%d, %d], [%d], %d)
    """
    total = 0
    for item in %s:
        if item %% %d == 0:
            total += item * %d
        elif item > limit:
            total -= %d
    return [x + total for x in %s if x < limit]


`, verb, noun, i, noun, other, c, strings.ToUpper(verb[:1])+verb[1:], noun, noun, other, c,
			verb, noun, i, r.IntN(50), r.IntN(50), r.IntN(50), c,
			noun, k, a, r.IntN(9)+1, other)
		b.WriteString(fn)
		used += len(strings.Fields(fn))
	}
	return b.String()
}

// longCodeBody packs HumanEval up to words, then continues with generated code
// once the problem set runs out.
func longCodeBody(words, variation int) string {
	full := fullCodePromptWords()
	if words <= full {
		return codePromptBody(words, variation)
	}
	return codePromptBody(full, variation) + "\n\n\n" + synthCode(words-full, uint64(variation)+1)
}

// scenarioWAV returns one second of 16 kHz mono PCM whose tone depends on seed.
func scenarioWAV(seed int) []byte {
	const rate, seconds = 16000, 1
	freq := 220 + float64(seed%17)*40
	samples := make([]byte, 0, 2*rate*seconds)
	for i := range rate * seconds {
		v := int16(8000 * math.Sin(2*math.Pi*freq*float64(i)/rate))
		samples = binary.LittleEndian.AppendUint16(samples, uint16(v))
	}
	var b bytes.Buffer
	b.WriteString("RIFF")
	_ = binary.Write(&b, binary.LittleEndian, uint32(36+len(samples)))
	b.WriteString("WAVEfmt ")
	for _, v := range []any{uint32(16), uint16(1), uint16(1), uint32(rate), uint32(2 * rate), uint16(2), uint16(16)} {
		_ = binary.Write(&b, binary.LittleEndian, v)
	}
	b.WriteString("data")
	_ = binary.Write(&b, binary.LittleEndian, uint32(len(samples)))
	b.Write(samples)
	return b.Bytes()
}

// scenarioPNG returns a small image whose content depends on seed, so each
// epoch's image is distinct.
func scenarioPNG(seed int) []byte {
	const size = 224
	r := rand.New(rand.NewPCG(uint64(seed), 7))
	base := color.RGBA{uint8(r.IntN(256)), uint8(r.IntN(256)), uint8(r.IntN(256)), 255}
	img := image.NewRGBA(image.Rect(0, 0, size, size))
	for y := range size {
		for x := range size {
			img.Set(x, y, color.RGBA{base.R + uint8(x), base.G + uint8(y), base.B + uint8(x^y), 255})
		}
	}
	var buf bytes.Buffer
	_ = png.Encode(&buf, img)
	return buf.Bytes()
}
