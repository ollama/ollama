// Package meta provides image metadata reading and transformation utilities.
package meta

import (
	"bytes"
	"encoding/binary"
	"image"
	_ "image/jpeg"
	_ "image/png"
)

// Metadata contains image metadata extracted from raw bytes.
type Metadata struct {
	Width       int    // Image width in pixels (after orientation correction)
	Height      int    // Image height in pixels (after orientation correction)
	Orientation int    // EXIF orientation (1-8, 1=normal)
	Format      string // Image format ("jpeg", "png", etc.)
}

// Read extracts metadata from image bytes without fully decoding pixels.
// Returns nil if the format is not recognized.
func Read(data []byte) *Metadata {
	if len(data) < 8 {
		return nil
	}

	// Detect format and read metadata
	if data[0] == 0xFF && data[1] == 0xD8 {
		return readJPEGMetadata(data)
	}
	if string(data[:8]) == "\x89PNG\r\n\x1a\n" {
		return readPNGMetadata(data)
	}

	return nil
}

// readJPEGMetadata reads metadata from JPEG bytes.
func readJPEGMetadata(data []byte) *Metadata {
	m := &Metadata{
		Format:      "jpeg",
		Orientation: 1,
	}

	r := bytes.NewReader(data[2:]) // Skip SOI marker

	for {
		var marker [2]byte
		if _, err := r.Read(marker[:]); err != nil {
			break
		}
		if marker[0] != 0xFF {
			break
		}

		switch marker[1] {
		case 0xE1: // APP1 (EXIF)
			m.Orientation = parseAPP1(r)

		case 0xC0, 0xC1, 0xC2: // SOF0, SOF1, SOF2 (Start of Frame)
			var lenBytes [2]byte
			if _, err := r.Read(lenBytes[:]); err != nil {
				break
			}
			var sof [5]byte
			if _, err := r.Read(sof[:]); err != nil {
				break
			}
			m.Height = int(binary.BigEndian.Uint16(sof[1:3]))
			m.Width = int(binary.BigEndian.Uint16(sof[3:5]))

			// Swap dimensions for 90°/270° rotations
			if m.Orientation >= 5 {
				m.Width, m.Height = m.Height, m.Width
			}
			return m

		case 0xD9, 0xDA: // EOI or SOS - stop scanning
			return m

		default:
			// Skip marker
			if marker[1] >= 0xD0 && marker[1] <= 0xD7 {
				continue // RST markers have no length
			}
			var lenBytes [2]byte
			if _, err := r.Read(lenBytes[:]); err != nil {
				break
			}
			segLen := int(binary.BigEndian.Uint16(lenBytes[:])) - 2
			if segLen > 0 {
				r.Seek(int64(segLen), 1)
			}
		}
	}

	return m
}

// parseAPP1 parses an APP1 segment for EXIF orientation.
func parseAPP1(r *bytes.Reader) int {
	var lenBytes [2]byte
	if _, err := r.Read(lenBytes[:]); err != nil {
		return 1
	}
	segLen := int(binary.BigEndian.Uint16(lenBytes[:])) - 2
	if segLen < 14 {
		r.Seek(int64(segLen), 1)
		return 1
	}

	data := make([]byte, segLen)
	if _, err := r.Read(data); err != nil {
		return 1
	}

	// Check for "Exif\0\0" header
	if string(data[:4]) != "Exif" || data[4] != 0 || data[5] != 0 {
		return 1
	}

	return parseTIFFOrientation(data[6:])
}

// parseTIFFOrientation extracts orientation from TIFF header.
func parseTIFFOrientation(tiff []byte) int {
	if len(tiff) < 8 {
		return 1
	}

	var byteOrder binary.ByteOrder
	switch string(tiff[:2]) {
	case "MM":
		byteOrder = binary.BigEndian
	case "II":
		byteOrder = binary.LittleEndian
	default:
		return 1
	}

	if byteOrder.Uint16(tiff[2:4]) != 42 {
		return 1
	}

	ifdOffset := byteOrder.Uint32(tiff[4:8])
	if int(ifdOffset)+2 > len(tiff) {
		return 1
	}

	numEntries := byteOrder.Uint16(tiff[ifdOffset : ifdOffset+2])
	entryStart := ifdOffset + 2

	for i := range int(numEntries) {
		offset := entryStart + uint32(i)*12
		if int(offset)+12 > len(tiff) {
			break
		}
		if byteOrder.Uint16(tiff[offset:offset+2]) == 0x0112 {
			orientation := int(byteOrder.Uint16(tiff[offset+8 : offset+10]))
			if orientation >= 1 && orientation <= 8 {
				return orientation
			}
			return 1
		}
	}

	return 1
}

// readPNGMetadata reads metadata from PNG bytes.
func readPNGMetadata(data []byte) *Metadata {
	m := &Metadata{
		Format:      "png",
		Orientation: 1, // PNG has no EXIF orientation
	}

	// IHDR chunk starts at offset 8 (after signature)
	// Structure: length(4) + type(4) + data + crc(4)
	if len(data) < 24 {
		return m
	}

	// Verify IHDR chunk type
	if string(data[12:16]) != "IHDR" {
		return m
	}

	// IHDR data: width(4) + height(4) + bit_depth(1) + ...
	m.Width = int(binary.BigEndian.Uint32(data[16:20]))
	m.Height = int(binary.BigEndian.Uint32(data[20:24]))

	return m
}

// Decode decodes an image from bytes with EXIF orientation applied.
func Decode(data []byte) (image.Image, string, error) {
	meta := Read(data)
	orientation := 1
	if meta != nil {
		orientation = meta.Orientation
	}

	img, format, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, "", err
	}

	return ApplyOrientation(img, orientation), format, nil
}

// ApplyOrientation transforms an image according to EXIF orientation.
// Returns the original image if orientation is 1 (normal) or invalid.
func ApplyOrientation(img image.Image, orientation int) image.Image {
	if orientation <= 1 || orientation > 8 {
		return img
	}

	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()

	outW, outH := w, h
	if orientation >= 5 {
		outW, outH = h, w
	}

	out := image.NewRGBA(image.Rect(0, 0, outW, outH))

	for y := range h {
		for x := range w {
			var dx, dy int
			switch orientation {
			case 2: // Mirror horizontal
				dx, dy = w-1-x, y
			case 3: // Rotate 180
				dx, dy = w-1-x, h-1-y
			case 4: // Mirror vertical
				dx, dy = x, h-1-y
			case 5: // Mirror horizontal + rotate 270
				dx, dy = y, x
			case 6: // Rotate 90 CW
				dx, dy = h-1-y, x
			case 7: // Mirror horizontal + rotate 90
				dx, dy = h-1-y, w-1-x
			case 8: // Rotate 270 CW
				dx, dy = y, w-1-x
			}
			out.Set(dx, dy, img.At(x+bounds.Min.X, y+bounds.Min.Y))
		}
	}

	return out
}
