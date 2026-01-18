//go:build mlx

package imagegen

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"image"
	_ "image/jpeg"
	"image/png"
	"io"
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

// LoadImageFromBytes loads an image from bytes, applying EXIF orientation.
// Supports JPEG and PNG formats.
func LoadImageFromBytes(data []byte) (image.Image, error) {
	// Read EXIF orientation before decoding (JPEG only)
	orientation := readJPEGOrientation(bytes.NewReader(data))

	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("decode image: %w", err)
	}

	// Apply EXIF orientation
	return applyOrientation(img, orientation), nil
}

// applyOrientation rotates/flips an image based on EXIF orientation.
func applyOrientation(img image.Image, orientation int) image.Image {
	if orientation <= 1 || orientation > 8 {
		return img // Normal or unknown
	}

	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()

	// Determine output dimensions (swap for 90° rotations)
	outW, outH := w, h
	if orientation >= 5 {
		outW, outH = h, w
	}

	out := image.NewRGBA(image.Rect(0, 0, outW, outH))

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
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
			case 8: // Rotate 270 CW (90 CCW)
				dx, dy = y, w-1-x
			}
			out.Set(dx, dy, img.At(x+bounds.Min.X, y+bounds.Min.Y))
		}
	}

	return out
}

// readJPEGOrientation reads EXIF orientation from a JPEG file.
// Returns 1 (normal) if not found or not a JPEG.
func readJPEGOrientation(r io.Reader) int {
	// Read JPEG header
	var header [2]byte
	if _, err := io.ReadFull(r, header[:]); err != nil || header[0] != 0xFF || header[1] != 0xD8 {
		return 1
	}

	// Scan for APP1 (EXIF) marker
	for {
		var marker [2]byte
		if _, err := io.ReadFull(r, marker[:]); err != nil {
			return 1
		}
		if marker[0] != 0xFF {
			return 1
		}

		// Check for APP1 marker (0xFFE1)
		if marker[1] == 0xE1 {
			var lenBytes [2]byte
			if _, err := io.ReadFull(r, lenBytes[:]); err != nil {
				return 1
			}
			segLen := int(binary.BigEndian.Uint16(lenBytes[:])) - 2

			// Read segment data
			data := make([]byte, segLen)
			if _, err := io.ReadFull(r, data); err != nil {
				return 1
			}

			// Check for "Exif\0\0" header
			if len(data) < 14 || string(data[:4]) != "Exif" {
				continue
			}

			// Parse TIFF header (starts at offset 6)
			tiff := data[6:]
			var byteOrder binary.ByteOrder
			if string(tiff[:2]) == "MM" {
				byteOrder = binary.BigEndian
			} else if string(tiff[:2]) == "II" {
				byteOrder = binary.LittleEndian
			} else {
				return 1
			}

			// Get IFD0 offset
			ifdOffset := byteOrder.Uint32(tiff[4:8])
			if int(ifdOffset)+2 > len(tiff) {
				return 1
			}

			// Read number of IFD entries
			numEntries := byteOrder.Uint16(tiff[ifdOffset : ifdOffset+2])
			entryStart := ifdOffset + 2

			// Search for orientation tag (0x0112)
			for i := uint16(0); i < numEntries; i++ {
				offset := entryStart + uint32(i)*12
				if int(offset)+12 > len(tiff) {
					break
				}
				tag := byteOrder.Uint16(tiff[offset : offset+2])
				if tag == 0x0112 { // Orientation tag
					return int(byteOrder.Uint16(tiff[offset+8 : offset+10]))
				}
			}
			return 1
		}

		// Skip other markers
		if marker[1] == 0xD9 || marker[1] == 0xDA { // EOI or SOS
			return 1
		}
		if marker[1] >= 0xD0 && marker[1] <= 0xD7 { // RST markers (no length)
			continue
		}

		var lenBytes [2]byte
		if _, err := io.ReadFull(r, lenBytes[:]); err != nil {
			return 1
		}
		segLen := int(binary.BigEndian.Uint16(lenBytes[:])) - 2
		if _, err := io.CopyN(io.Discard, r, int64(segLen)); err != nil {
			return 1
		}
	}
}
