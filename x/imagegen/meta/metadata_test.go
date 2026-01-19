package meta

import (
	"bytes"
	"encoding/binary"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"testing"
)

func TestRead_JPEG(t *testing.T) {
	// Create a simple JPEG
	img := image.NewRGBA(image.Rect(0, 0, 100, 50))
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, nil); err != nil {
		t.Fatal(err)
	}

	m := Read(buf.Bytes())
	if m == nil {
		t.Fatal("expected metadata, got nil")
	}
	if m.Format != "jpeg" {
		t.Errorf("format = %q, want jpeg", m.Format)
	}
	if m.Width != 100 {
		t.Errorf("width = %d, want 100", m.Width)
	}
	if m.Height != 50 {
		t.Errorf("height = %d, want 50", m.Height)
	}
	if m.Orientation != 1 {
		t.Errorf("orientation = %d, want 1", m.Orientation)
	}
}

func TestRead_PNG(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 200, 100))
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}

	m := Read(buf.Bytes())
	if m == nil {
		t.Fatal("expected metadata, got nil")
	}
	if m.Format != "png" {
		t.Errorf("format = %q, want png", m.Format)
	}
	if m.Width != 200 {
		t.Errorf("width = %d, want 200", m.Width)
	}
	if m.Height != 100 {
		t.Errorf("height = %d, want 100", m.Height)
	}
	if m.Orientation != 1 {
		t.Errorf("orientation = %d, want 1 (PNG has no EXIF)", m.Orientation)
	}
}

func TestRead_JPEGWithEXIF(t *testing.T) {
	tests := []struct {
		name        string
		orientation int
		wantW, wantH int // after orientation correction
	}{
		{"normal", 1, 100, 50},
		{"mirror_h", 2, 100, 50},
		{"rotate_180", 3, 100, 50},
		{"mirror_v", 4, 100, 50},
		{"mirror_h_rot270", 5, 50, 100}, // swapped
		{"rotate_90", 6, 50, 100},       // swapped
		{"mirror_h_rot90", 7, 50, 100},  // swapped
		{"rotate_270", 8, 50, 100},      // swapped
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data := makeJPEGWithOrientation(100, 50, tt.orientation)
			m := Read(data)
			if m == nil {
				t.Fatal("expected metadata")
			}
			if m.Orientation != tt.orientation {
				t.Errorf("orientation = %d, want %d", m.Orientation, tt.orientation)
			}
			if m.Width != tt.wantW || m.Height != tt.wantH {
				t.Errorf("size = %dx%d, want %dx%d", m.Width, m.Height, tt.wantW, tt.wantH)
			}
		})
	}
}

func TestRead_Invalid(t *testing.T) {
	tests := []struct {
		name string
		data []byte
	}{
		{"empty", nil},
		{"too_short", []byte{0xFF}},
		{"random", []byte{0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := Read(tt.data)
			if m != nil {
				t.Errorf("expected nil for invalid data, got %+v", m)
			}
		})
	}
}

func TestApplyOrientation(t *testing.T) {
	// Create a 4x2 image with distinct pixels to verify transformations
	// Layout: [R G B Y]
	//         [C M W K]
	img := image.NewRGBA(image.Rect(0, 0, 4, 2))
	colors := []color.RGBA{
		{255, 0, 0, 255},     // R
		{0, 255, 0, 255},     // G
		{0, 0, 255, 255},     // B
		{255, 255, 0, 255},   // Y
		{0, 255, 255, 255},   // C
		{255, 0, 255, 255},   // M
		{255, 255, 255, 255}, // W
		{0, 0, 0, 255},       // K
	}
	for i, c := range colors {
		img.Set(i%4, i/4, c)
	}

	tests := []struct {
		orientation int
		wantW, wantH int
		topLeft      color.RGBA // what should be at (0,0) after transform
	}{
		{1, 4, 2, colors[0]}, // R - no change
		{2, 4, 2, colors[3]}, // Y - mirror horizontal
		{3, 4, 2, colors[7]}, // K - rotate 180
		{4, 4, 2, colors[4]}, // C - mirror vertical
		{5, 2, 4, colors[0]}, // R - transpose
		{6, 2, 4, colors[4]}, // C - rotate 90 CW
		{7, 2, 4, colors[7]}, // K - transverse
		{8, 2, 4, colors[3]}, // Y - rotate 270 CW
	}

	for _, tt := range tests {
		t.Run(string(rune('0'+tt.orientation)), func(t *testing.T) {
			result := ApplyOrientation(img, tt.orientation)
			bounds := result.Bounds()

			if bounds.Dx() != tt.wantW || bounds.Dy() != tt.wantH {
				t.Errorf("size = %dx%d, want %dx%d", bounds.Dx(), bounds.Dy(), tt.wantW, tt.wantH)
			}

			got := result.At(0, 0).(color.RGBA)
			if got != tt.topLeft {
				t.Errorf("top-left = %v, want %v", got, tt.topLeft)
			}
		})
	}
}

func TestDecode(t *testing.T) {
	// Test with orientation 6 (90° CW rotation)
	data := makeJPEGWithOrientation(100, 50, 6)

	img, format, err := Decode(data)
	if err != nil {
		t.Fatal(err)
	}
	if format != "jpeg" {
		t.Errorf("format = %q, want jpeg", format)
	}

	bounds := img.Bounds()
	// After 90° rotation, 100x50 becomes 50x100
	if bounds.Dx() != 50 || bounds.Dy() != 100 {
		t.Errorf("decoded size = %dx%d, want 50x100", bounds.Dx(), bounds.Dy())
	}
}

// makeJPEGWithOrientation creates a minimal JPEG with EXIF orientation.
func makeJPEGWithOrientation(w, h, orientation int) []byte {
	// Build EXIF APP1 segment with orientation
	exif := buildEXIF(orientation)

	// Create base JPEG
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	var jpegBuf bytes.Buffer
	jpeg.Encode(&jpegBuf, img, nil)
	jpegData := jpegBuf.Bytes()

	// Insert EXIF after SOI marker (first 2 bytes)
	var result bytes.Buffer
	result.Write(jpegData[:2]) // SOI
	result.Write(exif)         // APP1 with EXIF
	result.Write(jpegData[2:]) // Rest of JPEG

	return result.Bytes()
}

// buildEXIF creates a minimal EXIF APP1 segment with orientation tag.
func buildEXIF(orientation int) []byte {
	var buf bytes.Buffer

	// APP1 marker
	buf.WriteByte(0xFF)
	buf.WriteByte(0xE1)

	// Build TIFF/EXIF data
	var tiff bytes.Buffer

	// TIFF header (little endian)
	tiff.WriteString("II")                                 // Little endian
	binary.Write(&tiff, binary.LittleEndian, uint16(42))   // TIFF magic
	binary.Write(&tiff, binary.LittleEndian, uint32(8))    // IFD0 offset

	// IFD0
	binary.Write(&tiff, binary.LittleEndian, uint16(1))    // 1 entry

	// Orientation tag entry (12 bytes)
	binary.Write(&tiff, binary.LittleEndian, uint16(0x0112)) // Tag: Orientation
	binary.Write(&tiff, binary.LittleEndian, uint16(3))      // Type: SHORT
	binary.Write(&tiff, binary.LittleEndian, uint32(1))      // Count: 1
	binary.Write(&tiff, binary.LittleEndian, uint16(orientation))
	binary.Write(&tiff, binary.LittleEndian, uint16(0))      // Padding

	// Next IFD offset (0 = none)
	binary.Write(&tiff, binary.LittleEndian, uint32(0))

	// Build APP1 segment
	exifData := append([]byte("Exif\x00\x00"), tiff.Bytes()...)

	// Write length (includes length bytes but not marker)
	binary.Write(&buf, binary.BigEndian, uint16(len(exifData)+2))
	buf.Write(exifData)

	return buf.Bytes()
}
