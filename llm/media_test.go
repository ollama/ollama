package llm

import (
	"bytes"
	"encoding/base64"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"net/http"
	"strings"
	"testing"
)

func TestNewMediaDataNormalizesJPEGEXIFOrientation(t *testing.T) {
	jpegData := testJPEG(t)
	source, err := jpeg.Decode(bytes.NewReader(jpegData))
	if err != nil {
		t.Fatal(err)
	}

	for orientation := uint16(2); orientation <= 8; orientation++ {
		t.Run(exifOrientationName(orientation), func(t *testing.T) {
			input := jpegWithEXIFOrientation(t, jpegData, orientation, false)
			data := NewMediaData(0, input)
			if data.Kind != MediaKindImage {
				t.Fatalf("media kind = %q, want image", data.Kind)
			}
			if got := http.DetectContentType(data.Data); got != "image/png" {
				t.Fatalf("content type = %q, want image/png", got)
			}

			upright, err := png.Decode(bytes.NewReader(data.Data))
			if err != nil {
				t.Fatal(err)
			}
			assertEXIFOrientation(t, source, upright, orientation)
		})
	}

	t.Run("big endian", func(t *testing.T) {
		data := NewMediaData(0, jpegWithEXIFOrientation(t, jpegData, 6, true))
		upright, err := png.Decode(bytes.NewReader(data.Data))
		if err != nil {
			t.Fatal(err)
		}
		assertEXIFOrientation(t, source, upright, 6)
	})
}

func TestNewMediaDataKeepsUnorientedJPEGBytes(t *testing.T) {
	jpegData := testJPEG(t)
	for _, data := range [][]byte{
		jpegData,
		jpegWithEXIFOrientation(t, jpegData, 1, false),
		{0xff, 0xd8, 0xff, 0xe1, 0x00, 0x08, 'E', 'x', 'i', 'f'},
		{0xff, 0xd8, 0xff, 0xe1, 0x00, 0x10, 'E', 'x', 'i', 'f', 0x00, 0x00, 'I', 'I', 0x2a, 0x00},
	} {
		media := NewMediaData(0, data)
		if !bytes.Equal(media.Data, data) {
			t.Fatal("unoriented or malformed JPEG bytes changed")
		}
	}
}

func TestLlamaServerChatMessageUsesNormalizedJPEG(t *testing.T) {
	jpegData := testJPEG(t)
	source, err := jpeg.Decode(bytes.NewReader(jpegData))
	if err != nil {
		t.Fatal(err)
	}

	message, err := llamaServerChatMessage(Message{
		Role:  "user",
		Media: []MediaData{NewMediaData(0, jpegWithEXIFOrientation(t, jpegData, 6, false))},
	})
	if err != nil {
		t.Fatal(err)
	}
	parts, ok := message["content"].([]map[string]any)
	if !ok || len(parts) != 1 {
		t.Fatalf("content = %#v, want one image part", message["content"])
	}
	url := parts[0]["image_url"].(map[string]any)["url"].(string)
	const prefix = "data:image/png;base64,"
	if !strings.HasPrefix(url, prefix) {
		t.Fatalf("image URL = %q, want PNG data URL", url)
	}
	payload, err := base64.StdEncoding.DecodeString(strings.TrimPrefix(url, prefix))
	if err != nil {
		t.Fatal(err)
	}
	upright, err := png.Decode(bytes.NewReader(payload))
	if err != nil {
		t.Fatal(err)
	}
	assertEXIFOrientation(t, source, upright, 6)
}

func testJPEG(t *testing.T) []byte {
	t.Helper()
	img := image.NewGray(image.Rect(0, 0, 2, 3))
	for y := range 3 {
		for x := range 2 {
			img.SetGray(x, y, color.Gray{Y: uint8(20 + 35*y + 70*x)})
		}
	}

	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: 100}); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func jpegWithEXIFOrientation(t *testing.T, jpegData []byte, orientation uint16, bigEndian bool) []byte {
	t.Helper()
	if len(jpegData) < 2 || jpegData[0] != 0xff || jpegData[1] != 0xd8 {
		t.Fatal("test input is not a JPEG")
	}

	tiff := []byte{'I', 'I', 0x2a, 0x00, 0x08, 0x00, 0x00, 0x00, 0x01, 0x00,
		0x12, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00,
		byte(orientation), byte(orientation >> 8), 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00}
	if bigEndian {
		tiff = []byte{'M', 'M', 0x00, 0x2a, 0x00, 0x00, 0x00, 0x08, 0x00, 0x01,
			0x01, 0x12, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01,
			byte(orientation >> 8), byte(orientation), 0x00, 0x00,
			0x00, 0x00, 0x00, 0x00}
	}
	payload := append([]byte("Exif\x00\x00"), tiff...)
	length := len(payload) + 2
	segment := []byte{0xff, 0xe1, byte(length >> 8), byte(length)}
	segment = append(segment, payload...)

	result := make([]byte, 0, len(jpegData)+len(segment))
	result = append(result, jpegData[:2]...)
	result = append(result, segment...)
	return append(result, jpegData[2:]...)
}

func assertEXIFOrientation(t *testing.T, source, upright image.Image, orientation uint16) {
	t.Helper()
	want, ok := exifPixelSources[orientation]
	if !ok {
		t.Fatalf("unexpected orientation %d", orientation)
	}
	if got := upright.Bounds(); got.Dx() != len(want[0]) || got.Dy() != len(want) {
		t.Fatalf("upright bounds = %v, want %dx%d", got, len(want[0]), len(want))
	}

	for y, row := range want {
		for x, point := range row {
			wantR, wantG, wantB, wantA := source.At(point.X, point.Y).RGBA()
			gotR, gotG, gotB, gotA := upright.At(x, y).RGBA()
			if gotR != wantR || gotG != wantG || gotB != wantB || gotA != wantA {
				t.Fatalf("pixel (%d, %d) = %#04x %#04x %#04x %#04x, want source (%d, %d) %#04x %#04x %#04x %#04x", x, y, gotR, gotG, gotB, gotA, point.X, point.Y, wantR, wantG, wantB, wantA)
			}
		}
	}
}

var exifPixelSources = map[uint16][][]image.Point{
	2: {{image.Pt(1, 0), image.Pt(0, 0)}, {image.Pt(1, 1), image.Pt(0, 1)}, {image.Pt(1, 2), image.Pt(0, 2)}},
	3: {{image.Pt(1, 2), image.Pt(0, 2)}, {image.Pt(1, 1), image.Pt(0, 1)}, {image.Pt(1, 0), image.Pt(0, 0)}},
	4: {{image.Pt(0, 2), image.Pt(1, 2)}, {image.Pt(0, 1), image.Pt(1, 1)}, {image.Pt(0, 0), image.Pt(1, 0)}},
	5: {{image.Pt(0, 0), image.Pt(0, 1), image.Pt(0, 2)}, {image.Pt(1, 0), image.Pt(1, 1), image.Pt(1, 2)}},
	6: {{image.Pt(0, 2), image.Pt(0, 1), image.Pt(0, 0)}, {image.Pt(1, 2), image.Pt(1, 1), image.Pt(1, 0)}},
	7: {{image.Pt(1, 2), image.Pt(1, 1), image.Pt(1, 0)}, {image.Pt(0, 2), image.Pt(0, 1), image.Pt(0, 0)}},
	8: {{image.Pt(1, 0), image.Pt(1, 1), image.Pt(1, 2)}, {image.Pt(0, 0), image.Pt(0, 1), image.Pt(0, 2)}},
}

func exifOrientationName(orientation uint16) string {
	return string(rune('0' + orientation))
}
