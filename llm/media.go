package llm

import (
	"bytes"
	"encoding/binary"
	"image"
	"image/jpeg"
	"image/png"
	"net/http"
	"strings"
)

func NewMediaData(id int, data []byte) MediaData {
	data = normalizeJPEGOrientation(data)

	return MediaData{
		Data: data,
		ID:   id,
		Kind: DetectMediaKind(data),
	}
}

// normalizeJPEGOrientation transcodes JPEGs that carry a non-default Exif
// orientation into upright PNGs. Go's JPEG decoder, like llama.cpp's image
// loader, decodes the stored pixel order without applying the Exif transform.
// Leave malformed metadata and unrotated images untouched so media handling
// retains its existing behavior for unsupported inputs.
func normalizeJPEGOrientation(data []byte) []byte {
	orientation := jpegOrientation(data)
	if orientation == 1 {
		return data
	}

	img, err := jpeg.Decode(bytes.NewReader(data))
	if err != nil {
		return data
	}

	var buf bytes.Buffer
	if err := png.Encode(&buf, applyEXIFOrientation(img, orientation)); err != nil {
		return data
	}
	return buf.Bytes()
}

// jpegOrientation returns the first valid Exif orientation tag in a JPEG's
// APP1 segments. It deliberately treats malformed metadata as the default
// orientation: parsing media metadata must not make an otherwise pass-through
// media request fail.
func jpegOrientation(data []byte) uint16 {
	if len(data) < 4 || data[0] != 0xff || data[1] != 0xd8 {
		return 1
	}

	for i := 2; i < len(data); {
		if data[i] != 0xff {
			return 1
		}
		for i < len(data) && data[i] == 0xff {
			i++
		}
		if i == len(data) {
			return 1
		}

		marker := data[i]
		i++
		switch {
		case marker == 0xd9 || marker == 0xda:
			return 1
		case marker == 0x01 || marker >= 0xd0 && marker <= 0xd7:
			continue
		}

		if i+2 > len(data) {
			return 1
		}
		length := int(binary.BigEndian.Uint16(data[i : i+2]))
		if length < 2 || i+length > len(data) {
			return 1
		}
		if marker == 0xe1 {
			if orientation := exifOrientation(data[i+2 : i+length]); orientation != 1 {
				return orientation
			}
		}
		i += length
	}

	return 1
}

func exifOrientation(data []byte) uint16 {
	if len(data) < 14 || !bytes.Equal(data[:6], []byte("Exif\x00\x00")) {
		return 1
	}

	tiff := data[6:]
	var order binary.ByteOrder
	switch string(tiff[:2]) {
	case "II":
		order = binary.LittleEndian
	case "MM":
		order = binary.BigEndian
	default:
		return 1
	}
	if order.Uint16(tiff[2:4]) != 42 {
		return 1
	}

	ifdOffset := order.Uint32(tiff[4:8])
	if ifdOffset < 8 || uint64(ifdOffset)+2 > uint64(len(tiff)) {
		return 1
	}
	entryCount := int(order.Uint16(tiff[int(ifdOffset) : int(ifdOffset)+2]))
	entries := int(ifdOffset) + 2
	if entryCount > (len(tiff)-entries)/12 {
		return 1
	}

	for i := range entryCount {
		entry := tiff[entries+i*12 : entries+(i+1)*12]
		if order.Uint16(entry[:2]) != 0x0112 {
			continue
		}
		if order.Uint16(entry[2:4]) != 3 || order.Uint32(entry[4:8]) != 1 {
			return 1
		}
		orientation := order.Uint16(entry[8:10])
		if orientation >= 2 && orientation <= 8 {
			return orientation
		}
		return 1
	}

	return 1
}

func applyEXIFOrientation(src image.Image, orientation uint16) image.Image {
	bounds := src.Bounds()
	width, height := bounds.Dx(), bounds.Dy()
	destination := image.Rect(0, 0, width, height)
	if orientation >= 5 {
		destination = image.Rect(0, 0, height, width)
	}
	dst := image.NewRGBA(destination)

	for y := range height {
		for x := range width {
			dx, dy := x, y
			switch orientation {
			case 2:
				dx = width - 1 - x
			case 3:
				dx, dy = width-1-x, height-1-y
			case 4:
				dy = height - 1 - y
			case 5:
				dx, dy = y, x
			case 6:
				dx, dy = height-1-y, x
			case 7:
				dx, dy = height-1-y, width-1-x
			case 8:
				dx, dy = y, width-1-x
			}
			dst.Set(dx, dy, src.At(bounds.Min.X+x, bounds.Min.Y+y))
		}
	}

	return dst
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
