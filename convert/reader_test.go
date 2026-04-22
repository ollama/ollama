package convert

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/d4l3k/go-bfloat16"
	"github.com/google/go-cmp/cmp"
	"github.com/x448/float16"
)

func TestSafetensors(t *testing.T) {
	t.Parallel()

	root, err := os.OpenRoot(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer root.Close()

	cases := []struct {
		name,
		dtype string
		offset,
		size int64
		shape []uint64
		setup func(*testing.T, *os.File)
		want  []byte
	}{
		{
			name:  "fp32-fp32",
			dtype: "F32",
			size:  32 * 4, // 32 floats, each 4 bytes
			shape: []uint64{32},
			setup: func(t *testing.T, f *os.File) {
				f32s := make([]float32, 32)
				for i := range f32s {
					f32s[i] = float32(i)
				}

				if err := binary.Write(f, binary.LittleEndian, f32s); err != nil {
					t.Fatal(err)
				}
			},
			want: []byte{
				0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x40, 0x40,
				0x00, 0x00, 0x80, 0x40, 0x00, 0x00, 0xa0, 0x40, 0x00, 0x00, 0xc0, 0x40, 0x00, 0x00, 0xe0, 0x40,
				0x00, 0x00, 0x00, 0x41, 0x00, 0x00, 0x10, 0x41, 0x00, 0x00, 0x20, 0x41, 0x00, 0x00, 0x30, 0x41,
				0x00, 0x00, 0x40, 0x41, 0x00, 0x00, 0x50, 0x41, 0x00, 0x00, 0x60, 0x41, 0x00, 0x00, 0x70, 0x41,
				0x00, 0x00, 0x80, 0x41, 0x00, 0x00, 0x88, 0x41, 0x00, 0x00, 0x90, 0x41, 0x00, 0x00, 0x98, 0x41,
				0x00, 0x00, 0xa0, 0x41, 0x00, 0x00, 0xa8, 0x41, 0x00, 0x00, 0xb0, 0x41, 0x00, 0x00, 0xb8, 0x41,
				0x00, 0x00, 0xc0, 0x41, 0x00, 0x00, 0xc8, 0x41, 0x00, 0x00, 0xd0, 0x41, 0x00, 0x00, 0xd8, 0x41,
				0x00, 0x00, 0xe0, 0x41, 0x00, 0x00, 0xe8, 0x41, 0x00, 0x00, 0xf0, 0x41, 0x00, 0x00, 0xf8, 0x41,
			},
		},
		{
			name:  "fp32-fp16",
			dtype: "F32",
			size:  32 * 4, // 32 floats, each 4 bytes
			shape: []uint64{16, 2},
			setup: func(t *testing.T, f *os.File) {
				f32s := make([]float32, 32)
				for i := range f32s {
					f32s[i] = float32(i)
				}

				if err := binary.Write(f, binary.LittleEndian, f32s); err != nil {
					t.Fatal(err)
				}
			},
			want: []byte{
				0x00, 0x00, 0x00, 0x3c, 0x00, 0x40, 0x00, 0x42, 0x00, 0x44, 0x00, 0x45, 0x00, 0x46, 0x00, 0x47,
				0x00, 0x48, 0x80, 0x48, 0x00, 0x49, 0x80, 0x49, 0x00, 0x4a, 0x80, 0x4a, 0x00, 0x4b, 0x80, 0x4b,
				0x00, 0x4c, 0x40, 0x4c, 0x80, 0x4c, 0xc0, 0x4c, 0x00, 0x4d, 0x40, 0x4d, 0x80, 0x4d, 0xc0, 0x4d,
				0x00, 0x4e, 0x40, 0x4e, 0x80, 0x4e, 0xc0, 0x4e, 0x00, 0x4f, 0x40, 0x4f, 0x80, 0x4f, 0xc0, 0x4f,
			},
		},
		{
			name:  "fp16-fp16",
			dtype: "F16",
			size:  32 * 2, // 32 floats, each 2 bytes
			shape: []uint64{16, 2},
			setup: func(t *testing.T, f *os.File) {
				u16s := make([]uint16, 32)
				for i := range u16s {
					u16s[i] = float16.Fromfloat32(float32(i)).Bits()
				}

				if err := binary.Write(f, binary.LittleEndian, u16s); err != nil {
					t.Fatal(err)
				}
			},
			want: []byte{
				0x00, 0x00, 0x00, 0x3c, 0x00, 0x40, 0x00, 0x42, 0x00, 0x44, 0x00, 0x45, 0x00, 0x46, 0x00, 0x47,
				0x00, 0x48, 0x80, 0x48, 0x00, 0x49, 0x80, 0x49, 0x00, 0x4a, 0x80, 0x4a, 0x00, 0x4b, 0x80, 0x4b,
				0x00, 0x4c, 0x40, 0x4c, 0x80, 0x4c, 0xc0, 0x4c, 0x00, 0x4d, 0x40, 0x4d, 0x80, 0x4d, 0xc0, 0x4d,
				0x00, 0x4e, 0x40, 0x4e, 0x80, 0x4e, 0xc0, 0x4e, 0x00, 0x4f, 0x40, 0x4f, 0x80, 0x4f, 0xc0, 0x4f,
			},
		},
		{
			name:  "fp16-fp32",
			dtype: "F16",
			size:  32 * 2, // 32 floats, each 2 bytes
			shape: []uint64{32},
			setup: func(t *testing.T, f *os.File) {
				u16s := make([]uint16, 32)
				for i := range u16s {
					u16s[i] = float16.Fromfloat32(float32(i)).Bits()
				}

				if err := binary.Write(f, binary.LittleEndian, u16s); err != nil {
					t.Fatal(err)
				}
			},
			want: []byte{
				0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x40, 0x40,
				0x00, 0x00, 0x80, 0x40, 0x00, 0x00, 0xa0, 0x40, 0x00, 0x00, 0xc0, 0x40, 0x00, 0x00, 0xe0, 0x40,
				0x00, 0x00, 0x00, 0x41, 0x00, 0x00, 0x10, 0x41, 0x00, 0x00, 0x20, 0x41, 0x00, 0x00, 0x30, 0x41,
				0x00, 0x00, 0x40, 0x41, 0x00, 0x00, 0x50, 0x41, 0x00, 0x00, 0x60, 0x41, 0x00, 0x00, 0x70, 0x41,
				0x00, 0x00, 0x80, 0x41, 0x00, 0x00, 0x88, 0x41, 0x00, 0x00, 0x90, 0x41, 0x00, 0x00, 0x98, 0x41,
				0x00, 0x00, 0xa0, 0x41, 0x00, 0x00, 0xa8, 0x41, 0x00, 0x00, 0xb0, 0x41, 0x00, 0x00, 0xb8, 0x41,
				0x00, 0x00, 0xc0, 0x41, 0x00, 0x00, 0xc8, 0x41, 0x00, 0x00, 0xd0, 0x41, 0x00, 0x00, 0xd8, 0x41,
				0x00, 0x00, 0xe0, 0x41, 0x00, 0x00, 0xe8, 0x41, 0x00, 0x00, 0xf0, 0x41, 0x00, 0x00, 0xf8, 0x41,
			},
		},
		{
			name:  "bf16-bf16",
			dtype: "BF16",
			size:  32 * 2, // 32 brain floats, each 2 bytes
			shape: []uint64{16, 2},
			setup: func(t *testing.T, f *os.File) {
				f32s := make([]float32, 32)
				for i := range f32s {
					f32s[i] = float32(i)
				}

				if err := binary.Write(f, binary.LittleEndian, bfloat16.EncodeFloat32(f32s)); err != nil {
					t.Fatal(err)
				}
			},
			want: []byte{
				0x00, 0x00, 0x80, 0x3f, 0x00, 0x40, 0x40, 0x40, 0x80, 0x40, 0xa0, 0x40, 0xc0, 0x40, 0xe0, 0x40,
				0x00, 0x41, 0x10, 0x41, 0x20, 0x41, 0x30, 0x41, 0x40, 0x41, 0x50, 0x41, 0x60, 0x41, 0x70, 0x41,
				0x80, 0x41, 0x88, 0x41, 0x90, 0x41, 0x98, 0x41, 0xa0, 0x41, 0xa8, 0x41, 0xb0, 0x41, 0xb8, 0x41,
				0xc0, 0x41, 0xc8, 0x41, 0xd0, 0x41, 0xd8, 0x41, 0xe0, 0x41, 0xe8, 0x41, 0xf0, 0x41, 0xf8, 0x41,
			},
		},
		{
			name:  "bf16-fp32",
			dtype: "BF16",
			size:  32 * 2, // 32 brain floats, each 2 bytes
			shape: []uint64{32},
			setup: func(t *testing.T, f *os.File) {
				f32s := make([]float32, 32)
				for i := range f32s {
					f32s[i] = float32(i)
				}

				if err := binary.Write(f, binary.LittleEndian, bfloat16.EncodeFloat32(f32s)); err != nil {
					t.Fatal(err)
				}
			},
			want: []byte{
				0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x40, 0x40,
				0x00, 0x00, 0x80, 0x40, 0x00, 0x00, 0xa0, 0x40, 0x00, 0x00, 0xc0, 0x40, 0x00, 0x00, 0xe0, 0x40,
				0x00, 0x00, 0x00, 0x41, 0x00, 0x00, 0x10, 0x41, 0x00, 0x00, 0x20, 0x41, 0x00, 0x00, 0x30, 0x41,
				0x00, 0x00, 0x40, 0x41, 0x00, 0x00, 0x50, 0x41, 0x00, 0x00, 0x60, 0x41, 0x00, 0x00, 0x70, 0x41,
				0x00, 0x00, 0x80, 0x41, 0x00, 0x00, 0x88, 0x41, 0x00, 0x00, 0x90, 0x41, 0x00, 0x00, 0x98, 0x41,
				0x00, 0x00, 0xa0, 0x41, 0x00, 0x00, 0xa8, 0x41, 0x00, 0x00, 0xb0, 0x41, 0x00, 0x00, 0xb8, 0x41,
				0x00, 0x00, 0xc0, 0x41, 0x00, 0x00, 0xc8, 0x41, 0x00, 0x00, 0xd0, 0x41, 0x00, 0x00, 0xd8, 0x41,
				0x00, 0x00, 0xe0, 0x41, 0x00, 0x00, 0xe8, 0x41, 0x00, 0x00, 0xf0, 0x41, 0x00, 0x00, 0xf8, 0x41,
			},
		},
		{
			name:  "u8-u8",
			dtype: "U8",
			size:  32, // 32 brain floats, each 1 bytes
			shape: []uint64{32},
			setup: func(t *testing.T, f *os.File) {
				u8s := make([]uint8, 32)
				for i := range u8s {
					u8s[i] = uint8(i)
				}

				if err := binary.Write(f, binary.LittleEndian, u8s); err != nil {
					t.Fatal(err)
				}
			},
			want: []byte{
				0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
				0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
			},
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			path := filepath.Base(t.Name())
			st := safetensor{
				fs:     root.FS(),
				path:   path,
				dtype:  tt.dtype,
				offset: tt.offset,
				size:   tt.size,
				tensorBase: &tensorBase{
					name:  tt.name,
					shape: tt.shape,
				},
			}

			f, err := root.Create(path)
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()

			tt.setup(t, f)

			var b bytes.Buffer
			if _, err := st.WriteTo(&b); err != nil {
				t.Fatal(err)
			}

			if diff := cmp.Diff(tt.want, b.Bytes()); diff != "" {
				t.Errorf("safetensor.WriteTo() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestSafetensorWriteToFP8E4M3(t *testing.T) {
	root, err := os.OpenRoot(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer root.Close()

	path := filepath.Base(t.Name())
	f, err := root.Create(path)
	if err != nil {
		t.Fatal(err)
	}

	// E4M3FN encodings for 1.0, 2.0, 0.5, and -1.0.
	if _, err := f.Write([]byte{0x38, 0x40, 0x30, 0xb8}); err != nil {
		t.Fatal(err)
	}
	if _, err := f.Write(bfloat16.EncodeFloat32([]float32{2})); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	st := safetensor{
		fs:       root.FS(),
		path:     path,
		dtype:    "F8_E4M3",
		offset:   0,
		size:     4,
		fp8Block: safetensorFP8BlockSize{rows: 128, cols: 128, ok: true},
		scale: &safetensorScale{
			name:   "linear.weight_scale",
			dtype:  "BF16",
			shape:  []uint64{1, 1},
			offset: 4,
			size:   2,
		},
		tensorBase: &tensorBase{
			name:  "linear.weight",
			shape: []uint64{2, 2},
		},
	}

	var b bytes.Buffer
	if _, err := st.WriteTo(&b); err != nil {
		t.Fatal(err)
	}

	want := bfloat16.EncodeFloat32([]float32{2, 4, 1, -2})
	if diff := cmp.Diff(want, b.Bytes()); diff != "" {
		t.Errorf("safetensor.WriteTo() mismatch (-want +got):\n%s", diff)
	}
}

func TestSafetensorWriteToFP8E4M3UsesConfiguredBlockSize(t *testing.T) {
	root, err := os.OpenRoot(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer root.Close()

	path := filepath.Base(t.Name())
	f, err := root.Create(path)
	if err != nil {
		t.Fatal(err)
	}

	if _, err := f.Write(bytes.Repeat([]byte{0x38}, 12)); err != nil {
		t.Fatal(err)
	}
	if _, err := f.Write(bfloat16.EncodeFloat32([]float32{1, 2, 3, 4})); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	st := safetensor{
		fs:       root.FS(),
		path:     path,
		dtype:    "F8_E4M3",
		offset:   0,
		size:     12,
		fp8Block: safetensorFP8BlockSize{rows: 2, cols: 3, ok: true},
		scale: &safetensorScale{
			name:   "linear.weight_scale",
			dtype:  "BF16",
			shape:  []uint64{2, 2},
			offset: 12,
			size:   8,
		},
		tensorBase: &tensorBase{
			name:  "linear.weight",
			shape: []uint64{3, 4},
		},
	}

	var b bytes.Buffer
	if _, err := st.WriteTo(&b); err != nil {
		t.Fatal(err)
	}

	want := bfloat16.EncodeFloat32([]float32{
		1, 1, 1, 2,
		1, 1, 1, 2,
		3, 3, 3, 4,
	})
	if diff := cmp.Diff(want, b.Bytes()); diff != "" {
		t.Errorf("safetensor.WriteTo() mismatch (-want +got):\n%s", diff)
	}
}

func TestParseSafetensorsConsumesFP8ScaleCompanion(t *testing.T) {
	tempDir := t.TempDir()
	generateSafetensorTestData(t, tempDir, map[string]*tensorData{
		"linear.weight": {
			Offsets: []int{0, 4},
			Type:    "F8_E4M3",
			Shape:   []int{2, 2},
		},
		"linear.weight_scale": {
			Offsets: []int{4, 6},
			Type:    "BF16",
			Shape:   []int{1, 1},
		},
	})
	writeFP8BlockConfig(t, tempDir, 128, 128)

	tensors, err := parseSafetensors(os.DirFS(tempDir), strings.NewReplacer(), "model-00001-of-00001.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	if len(tensors) != 1 {
		t.Fatalf("expected one tensor, got %d", len(tensors))
	}
	if got := tensors[0].Name(); got != "linear.weight" {
		t.Fatalf("unexpected tensor name %q", got)
	}
	if got := tensors[0].Kind(); got != tensorKindBF16 {
		t.Fatalf("unexpected fp8 converted kind %d, want %d", got, tensorKindBF16)
	}
}

func TestParseSafetensorsRejectsFP8WithoutBlockMetadata(t *testing.T) {
	tempDir := t.TempDir()
	generateSafetensorTestData(t, tempDir, map[string]*tensorData{
		"linear.weight": {
			Offsets: []int{0, 4},
			Type:    "F8_E4M3",
			Shape:   []int{2, 2},
		},
		"linear.weight_scale": {
			Offsets: []int{4, 6},
			Type:    "BF16",
			Shape:   []int{1, 1},
		},
	})

	_, err := parseSafetensors(os.DirFS(tempDir), strings.NewReplacer(), "model-00001-of-00001.safetensors")
	if err == nil || !strings.Contains(err.Error(), "missing fp8 block size metadata") {
		t.Fatalf("expected missing fp8 block size metadata error, got %v", err)
	}
}

func TestParseSafetensorsRejectsAmbiguousFP8ScaleCompanion(t *testing.T) {
	tempDir := t.TempDir()
	generateSafetensorTestData(t, tempDir, map[string]*tensorData{
		"linear.weight": {
			Offsets: []int{0, 4},
			Type:    "F8_E4M3",
			Shape:   []int{2, 2},
		},
		"linear.weight_scale": {
			Offsets: []int{4, 6},
			Type:    "BF16",
			Shape:   []int{1, 1},
		},
		"linear.weight.scale": {
			Offsets: []int{6, 8},
			Type:    "BF16",
			Shape:   []int{1, 1},
		},
	})
	writeFP8BlockConfig(t, tempDir, 128, 128)

	_, err := parseSafetensors(os.DirFS(tempDir), strings.NewReplacer(), "model-00001-of-00001.safetensors")
	if err == nil || !strings.Contains(err.Error(), "multiple fp8 scale companions") {
		t.Fatalf("expected ambiguous fp8 scale companion error, got %v", err)
	}
}

func writeFP8BlockConfig(t *testing.T, dir string, rows, cols int) {
	t.Helper()

	config := fmt.Sprintf(`{
  "architectures": ["GenericForCausalLM"],
  "compression_config": {
    "format": "float-quantized",
    "config_groups": {
      "group_0": {
        "format": "float-quantized",
        "weights": {
          "type": "float",
          "num_bits": 8,
          "block_structure": [%d, %d]
        }
      }
    }
  }
}`, rows, cols)
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(config), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestSafetensorKind(t *testing.T) {
	tests := []struct {
		name     string
		st       safetensor
		expected uint32
	}{
		{
			name: "BF16 dtype with non-v. prefix and non-FP32 base kind should return BF16",
			st: safetensor{
				tensorBase: &tensorBase{
					name:  "weight.matrix",
					shape: []uint64{10, 10}, // will default to FP16
				},
				dtype: "BF16",
			},
			expected: tensorKindBF16,
		},
		{
			name: "BF16 dtype with v. prefix should return base kind",
			st: safetensor{
				tensorBase: &tensorBase{
					name:  "v.weight.matrix",
					shape: []uint64{10, 10}, // will default to FP16
				},
				dtype: "BF16",
			},
			expected: tensorKindFP16,
		},
		{
			name: "BF16 audio feature extractor constants should return FP32",
			st: safetensor{
				tensorBase: &tensorBase{
					name:  "a.feature_extractor.fb",
					shape: []uint64{1, 128, 257},
				},
				dtype: "BF16",
			},
			expected: tensorKindFP32,
		},
		{
			name: "BF16 dtype with FP32 base kind should return FP32",
			st: safetensor{
				tensorBase: &tensorBase{
					name:  "weight.matrix",
					shape: []uint64{10}, // will default to FP32
				},
				dtype: "BF16",
			},
			expected: tensorKindFP32,
		},
		{
			name: "Non-BF16 dtype should return base kind",
			st: safetensor{
				tensorBase: &tensorBase{
					name:  "weight.matrix",
					shape: []uint64{10, 10}, // will default to FP16
				},
				dtype: "FP16",
			},
			expected: tensorKindFP16,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.st.Kind()
			if result != tt.expected {
				t.Errorf("Kind() = %d, expected %d", result, tt.expected)
			}
		})
	}
}
