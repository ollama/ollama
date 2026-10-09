package llm

import (
	"bytes"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	"image/png"
	"math"
	"net/http"
	"strings"

	"golang.org/x/image/draw"
)

func NewMediaData(id int, data []byte) MediaData {
	return MediaData{
		Data: data,
		ID:   id,
		Kind: DetectMediaKind(data),
	}
}

func DetectMediaKind(data []byte) MediaKind {
	if _, ok := AudioFormat(data); ok {
		return MediaKindAudio
	}
	if strings.HasPrefix(http.DetectContentType(data), "image/") {
		return MediaKindImage
	}
	return MediaKindUnknown
}

func AudioFormat(data []byte) (string, bool) {
	if len(data) >= 12 && bytes.Equal(data[:4], []byte("RIFF")) && bytes.Equal(data[8:12], []byte("WAVE")) {
		return "wav", true
	}
	if len(data) >= 3 && bytes.Equal(data[:3], []byte("ID3")) {
		return "mp3", true
	}
	if len(data) >= 2 && data[0] == 0xff && data[1]&0xe0 == 0xe0 {
		return "mp3", true
	}
	return "", false
}

// resizeImageToTokenBudget stretches an image onto the largest grid of
// cell-pixel squares, at most budget of them, that keeps its aspect ratio,
// as the Gemma 4 reference processor does, and returns it as PNG.
func resizeImageToTokenBudget(data []byte, cell, budget int) ([]byte, error) {
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("decode image: %w", err)
	}

	bounds := img.Bounds()
	h, w, side := float64(bounds.Dy()), float64(bounds.Dx()), float64(cell)
	factor := math.Sqrt(float64(budget) * side * side / (h * w))
	th := math.Floor(factor*h/side) * side
	tw := math.Floor(factor*w/side) * side
	switch maxSide := float64(budget) * side; {
	case th == 0 && tw == 0:
		return nil, fmt.Errorf("image %dx%d is too small to process", bounds.Dx(), bounds.Dy())
	case th == 0:
		th, tw = side, min(math.Floor(w/h)*side, maxSide)
	case tw == 0:
		th, tw = min(math.Floor(h/w)*side, maxSide), side
	}

	// The reference drops alpha rather than compositing it.
	if o, ok := img.(interface{ Opaque() bool }); !ok || !o.Opaque() {
		flat := image.NewNRGBA(bounds)
		draw.Draw(flat, bounds, img, bounds.Min, draw.Src)
		for i := 3; i < len(flat.Pix); i += 4 {
			flat.Pix[i] = 0xff
		}
		img = flat
	}

	resized := image.NewRGBA(image.Rect(0, 0, int(tw), int(th)))
	draw.CatmullRom.Scale(resized, resized.Bounds(), img, bounds, draw.Src, nil)

	// The PNG only crosses loopback, and compressing it costs more than
	// encoding the image does.
	var buf bytes.Buffer
	if err := (&png.Encoder{CompressionLevel: png.NoCompression}).Encode(&buf, resized); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}
