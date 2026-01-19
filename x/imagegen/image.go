//go:build mlx

package imagegen

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"image"
	_ "image/jpeg"
	"image/png"
	"os"
	"path/filepath"

	"github.com/ollama/ollama/x/imagegen/mlx"
)

// SaveImage saves an MLX array as a PNG image file.
// Expected format: [B, C, H, W] with values in [0, 1] range and C=3 (RGB).
func SaveImage(arr *mlx.Array, path string) error {
	img, err := ArrayToImage(arr)
	if err != nil {
		return err
	}

	if filepath.Ext(path) != ".png" {
		path = path + ".png"
	}

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	return png.Encode(f, img)
}

// EncodeImageBase64 encodes an MLX array as a base64-encoded PNG.
// Expected format: [B, C, H, W] with values in [0, 1] range and C=3 (RGB).
func EncodeImageBase64(arr *mlx.Array) (string, error) {
	img, err := ArrayToImage(arr)
	if err != nil {
		return "", err
	}

	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return "", err
	}

	return base64.StdEncoding.EncodeToString(buf.Bytes()), nil
}

// ArrayToImage converts an MLX array to a Go image.RGBA.
// Expected format: [B, C, H, W] with values in [0, 1] range and C=3 (RGB).
func ArrayToImage(arr *mlx.Array) (*image.RGBA, error) {
	shape := arr.Shape()
	if len(shape) != 4 {
		return nil, fmt.Errorf("expected 4D array [B, C, H, W], got %v", shape)
	}

	// Transform to [H, W, C] for image conversion
	// Free intermediate arrays to avoid memory leak
	squeezed := mlx.Squeeze(arr, 0)
	transposed := mlx.Transpose(squeezed, 1, 2, 0)
	squeezed.Free()
	img := mlx.Contiguous(transposed)
	transposed.Free()
	mlx.Eval(img)

	imgShape := img.Shape()
	H := int(imgShape[0])
	W := int(imgShape[1])
	C := int(imgShape[2])

	if C != 3 {
		img.Free()
		return nil, fmt.Errorf("expected 3 channels (RGB), got %d", C)
	}

	// Copy to CPU and free GPU memory
	data := img.Data()
	img.Free()

	// Write directly to Pix slice (faster than SetRGBA)
	goImg := image.NewRGBA(image.Rect(0, 0, W, H))
	pix := goImg.Pix
	for y := 0; y < H; y++ {
		for x := 0; x < W; x++ {
			srcIdx := (y*W + x) * C
			dstIdx := (y*W + x) * 4
			pix[dstIdx+0] = uint8(clampF(data[srcIdx+0]*255+0.5, 0, 255))
			pix[dstIdx+1] = uint8(clampF(data[srcIdx+1]*255+0.5, 0, 255))
			pix[dstIdx+2] = uint8(clampF(data[srcIdx+2]*255+0.5, 0, 255))
			pix[dstIdx+3] = 255
		}
	}

	return goImg, nil
}

func clampF(v, min, max float32) float32 {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

// DecodeImage decodes image bytes with EXIF orientation applied.
func DecodeImage(data []byte) (image.Image, error) {
	orientation := readJPEGOrientation(data)

	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}

	return applyOrientation(img, orientation), nil
}

// readJPEGOrientation extracts EXIF orientation from JPEG bytes.
// Returns 1 (normal) for non-JPEG or if orientation not found.
func readJPEGOrientation(data []byte) int {
	if len(data) < 2 || data[0] != 0xFF || data[1] != 0xD8 {
		return 1 // Not JPEG
	}

	r := bytes.NewReader(data[2:])
	for {
		var marker [2]byte
		if _, err := r.Read(marker[:]); err != nil || marker[0] != 0xFF {
			return 1
		}

		if marker[1] == 0xE1 { // APP1 (EXIF)
			var lenBytes [2]byte
			if _, err := r.Read(lenBytes[:]); err != nil {
				return 1
			}
			segLen := int(uint16(lenBytes[0])<<8|uint16(lenBytes[1])) - 2
			if segLen < 14 {
				r.Seek(int64(segLen), 1)
				continue
			}
			seg := make([]byte, segLen)
			if _, err := r.Read(seg); err != nil {
				return 1
			}
			if string(seg[:4]) == "Exif" && seg[4] == 0 && seg[5] == 0 {
				return parseTIFFOrientation(seg[6:])
			}
			continue
		}

		if marker[1] == 0xD9 || marker[1] == 0xDA {
			return 1 // EOI or SOS
		}
		if marker[1] >= 0xD0 && marker[1] <= 0xD7 {
			continue // RST markers
		}

		var lenBytes [2]byte
		if _, err := r.Read(lenBytes[:]); err != nil {
			return 1
		}
		segLen := int(uint16(lenBytes[0])<<8|uint16(lenBytes[1])) - 2
		if segLen > 0 {
			r.Seek(int64(segLen), 1)
		}
	}
}

func parseTIFFOrientation(tiff []byte) int {
	if len(tiff) < 8 {
		return 1
	}

	var big bool
	switch string(tiff[:2]) {
	case "MM":
		big = true
	case "II":
		big = false
	default:
		return 1
	}

	u16 := func(b []byte) uint16 {
		if big {
			return uint16(b[0])<<8 | uint16(b[1])
		}
		return uint16(b[1])<<8 | uint16(b[0])
	}
	u32 := func(b []byte) uint32 {
		if big {
			return uint32(b[0])<<24 | uint32(b[1])<<16 | uint32(b[2])<<8 | uint32(b[3])
		}
		return uint32(b[3])<<24 | uint32(b[2])<<16 | uint32(b[1])<<8 | uint32(b[0])
	}

	if u16(tiff[2:4]) != 42 {
		return 1
	}

	ifdOffset := u32(tiff[4:8])
	if int(ifdOffset)+2 > len(tiff) {
		return 1
	}

	numEntries := u16(tiff[ifdOffset : ifdOffset+2])
	for i := range int(numEntries) {
		offset := ifdOffset + 2 + uint32(i)*12
		if int(offset)+12 > len(tiff) {
			break
		}
		if u16(tiff[offset:offset+2]) == 0x0112 { // Orientation tag
			o := int(u16(tiff[offset+8 : offset+10]))
			if o >= 1 && o <= 8 {
				return o
			}
			return 1
		}
	}
	return 1
}

func applyOrientation(img image.Image, orientation int) image.Image {
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
			case 2:
				dx, dy = w-1-x, y
			case 3:
				dx, dy = w-1-x, h-1-y
			case 4:
				dx, dy = x, h-1-y
			case 5:
				dx, dy = y, x
			case 6:
				dx, dy = h-1-y, x
			case 7:
				dx, dy = h-1-y, w-1-x
			case 8:
				dx, dy = y, w-1-x
			}
			out.Set(dx, dy, img.At(x+bounds.Min.X, y+bounds.Min.Y))
		}
	}
	return out
}
