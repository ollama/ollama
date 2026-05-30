package convert

import (
	"bytes"
	"encoding/binary"
	"io"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/d4l3k/go-bfloat16"
	"github.com/x448/float16"
)

const benchSize = 1 << 20 // 1M elements

func makeSafetensorFile(b *testing.B, dtype string, nelems int) (string, int64) {
	b.Helper()
	f, err := os.CreateTemp(b.TempDir(), "bench-*.safetensors")
	if err != nil {
		b.Fatal(err)
	}

	var elemSize int
	switch dtype {
	case "F32":
		elemSize = 4
	case "F16", "BF16":
		elemSize = 2
	}
	dataSize := nelems * elemSize

	switch dtype {
	case "F32":
		vals := make([]float32, nelems)
		for i := range vals {
			vals[i] = float32(i%1000) * 0.01
		}
		binary.Write(f, binary.LittleEndian, vals)
	case "F16":
		vals := make([]uint16, nelems)
		for i := range vals {
			vals[i] = float16.Fromfloat32(float32(i%1000) * 0.01).Bits()
		}
		binary.Write(f, binary.LittleEndian, vals)
	case "BF16":
		vals := make([]uint16, nelems)
		for i := range vals {
			vals[i] = uint16(math.Float32bits(float32(i%1000)*0.01) >> 16)
		}
		binary.Write(f, binary.LittleEndian, vals)
	}

	f.Close()
	return f.Name(), int64(dataSize)
}

func benchWriteTo(b *testing.B, dtype string, shape []uint64, nelems int) {
	b.Helper()
	path, size := makeSafetensorFile(b, dtype, nelems)
	dir := os.DirFS(filepath.Dir(path))
	name := filepath.Base(path)

	st := safetensor{
		fs:     dir,
		path:   name,
		dtype:  dtype,
		offset: 0,
		size:   size,
		tensorBase: &tensorBase{
			name:  "bench.weight",
			shape: shape,
		},
	}

	sink := io.Discard
	b.SetBytes(size)
	b.ResetTimer()
	for range b.N {
		if _, err := st.WriteTo(sink); err != nil {
			b.Fatal(err)
		}
	}
}

func benchWriteToMmap(b *testing.B, dtype string, shape []uint64, nelems int) {
	b.Helper()
	path, size := makeSafetensorFile(b, dtype, nelems)
	dir := os.DirFS(filepath.Dir(path))
	name := filepath.Base(path)

	mmap, err := mmapOpen(path)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { mmap.Close() })

	st := safetensor{
		fs:     dir,
		path:   name,
		dtype:  dtype,
		offset: 0,
		size:   size,
		mmap:   mmap,
		tensorBase: &tensorBase{
			name:  "bench.weight",
			shape: shape,
		},
	}

	sink := io.Discard
	b.SetBytes(size)
	b.ResetTimer()
	for range b.N {
		if _, err := st.WriteTo(sink); err != nil {
			b.Fatal(err)
		}
	}
}

// --- WriteTo: F16 source → F16 output (passthrough, 2D shape) ---

func BenchmarkWriteTo_F16_Passthrough_File(b *testing.B) {
	benchWriteTo(b, "F16", []uint64{1024, benchSize / 1024}, benchSize)
}

func BenchmarkWriteTo_F16_Passthrough_Mmap(b *testing.B) {
	benchWriteToMmap(b, "F16", []uint64{1024, benchSize / 1024}, benchSize)
}

// --- WriteTo: F16 source → F32 output (conversion needed, 1D shape) ---

func BenchmarkWriteTo_F16toF32_File(b *testing.B) {
	benchWriteTo(b, "F16", []uint64{benchSize}, benchSize)
}

func BenchmarkWriteTo_F16toF32_Mmap(b *testing.B) {
	benchWriteToMmap(b, "F16", []uint64{benchSize}, benchSize)
}

// --- WriteTo: BF16 source → BF16 output (passthrough, 2D shape) ---

func BenchmarkWriteTo_BF16_Passthrough_File(b *testing.B) {
	benchWriteTo(b, "BF16", []uint64{1024, benchSize / 1024}, benchSize)
}

func BenchmarkWriteTo_BF16_Passthrough_Mmap(b *testing.B) {
	benchWriteToMmap(b, "BF16", []uint64{1024, benchSize / 1024}, benchSize)
}

// --- WriteTo: BF16 source → F32 output (conversion needed, 1D shape) ---

func BenchmarkWriteTo_BF16toF32_File(b *testing.B) {
	benchWriteTo(b, "BF16", []uint64{benchSize}, benchSize)
}

func BenchmarkWriteTo_BF16toF32_Mmap(b *testing.B) {
	benchWriteToMmap(b, "BF16", []uint64{benchSize}, benchSize)
}

// --- WriteTo: F32 source → F16 output (conversion needed, 2D shape) ---

func BenchmarkWriteTo_F32toF16_File(b *testing.B) {
	benchWriteTo(b, "F32", []uint64{1024, benchSize / 1024}, benchSize)
}

func BenchmarkWriteTo_F32toF16_Mmap(b *testing.B) {
	benchWriteToMmap(b, "F32", []uint64{1024, benchSize / 1024}, benchSize)
}

// --- Isolated SIMD vs old library (micro-benchmarks for reference) ---

func BenchmarkConvert_F16ToF32_Old(b *testing.B) {
	src := make([]uint16, benchSize)
	for i := range src {
		src[i] = float16.Fromfloat32(float32(i%1000) * 0.01).Bits()
	}
	dst := make([]float32, benchSize)
	b.SetBytes(int64(benchSize * 2))
	b.ResetTimer()
	for range b.N {
		for i, v := range src {
			dst[i] = float16.Frombits(v).Float32()
		}
	}
}

func BenchmarkConvert_F16ToF32_SIMD(b *testing.B) {
	src := make([]uint16, benchSize)
	for i := range src {
		src[i] = float16.Fromfloat32(float32(i%1000) * 0.01).Bits()
	}
	dst := make([]float32, benchSize)
	b.SetBytes(int64(benchSize * 2))
	b.ResetTimer()
	for range b.N {
		convertF16ToF32(dst, src)
	}
}

func BenchmarkConvert_BF16ToF32_Old(b *testing.B) {
	u8s := make([]byte, benchSize*2)
	for i := range benchSize {
		v := uint16(math.Float32bits(float32(i%1000)*0.01) >> 16)
		u8s[i*2] = byte(v)
		u8s[i*2+1] = byte(v >> 8)
	}
	b.SetBytes(int64(benchSize * 2))
	b.ResetTimer()
	for range b.N {
		_ = bfloat16.DecodeFloat32(u8s)
	}
}

func BenchmarkConvert_BF16ToF32_SIMD(b *testing.B) {
	src := make([]uint16, benchSize)
	for i := range src {
		src[i] = uint16(math.Float32bits(float32(i%1000)*0.01) >> 16)
	}
	dst := make([]float32, benchSize)
	b.SetBytes(int64(benchSize * 2))
	b.ResetTimer()
	for range b.N {
		convertBF16ToF32(dst, src)
	}
}

func BenchmarkConvert_F32ToF16_Old(b *testing.B) {
	src := make([]float32, benchSize)
	for i := range src {
		src[i] = float32(i%1000) * 0.01
	}
	dst := make([]uint16, benchSize)
	b.SetBytes(int64(benchSize * 4))
	b.ResetTimer()
	for range b.N {
		for i, v := range src {
			dst[i] = float16.Fromfloat32(v).Bits()
		}
	}
}

func BenchmarkConvert_F32ToF16_SIMD(b *testing.B) {
	src := make([]float32, benchSize)
	for i := range src {
		src[i] = float32(i%1000) * 0.01
	}
	dst := make([]uint16, benchSize)
	b.SetBytes(int64(benchSize * 4))
	b.ResetTimer()
	for range b.N {
		convertF32ToF16(dst, src)
	}
}

func BenchmarkConvert_F32ToBF16_Old(b *testing.B) {
	src := make([]float32, benchSize)
	for i := range src {
		src[i] = float32(i%1000) * 0.01
	}
	b.SetBytes(int64(benchSize * 4))
	b.ResetTimer()
	for range b.N {
		_ = bfloat16.EncodeFloat32(src)
	}
}

func BenchmarkConvert_F32ToBF16_SIMD(b *testing.B) {
	src := make([]float32, benchSize)
	for i := range src {
		src[i] = float32(i%1000) * 0.01
	}
	dst := make([]uint16, benchSize)
	b.SetBytes(int64(benchSize * 4))
	b.ResetTimer()
	for range b.N {
		convertF32ToBF16(dst, src)
	}
}

// correctness check: verify SIMD matches scalar for all conversions
func TestSIMDCorrectness(t *testing.T) {
	const n = 8*100 + 3 // not multiple of 8, tests remainder

	t.Run("F16ToF32", func(t *testing.T) {
		src := make([]uint16, n)
		for i := range src {
			src[i] = float16.Fromfloat32(float32(i) - 500).Bits()
		}
		want := make([]float32, n)
		for i, v := range src {
			want[i] = float16.Frombits(v).Float32()
		}
		got := make([]float32, n)
		convertF16ToF32(got, src)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("mismatch at %d: got %v want %v", i, got[i], want[i])
			}
		}
	})

	t.Run("F32ToF16", func(t *testing.T) {
		src := make([]float32, n)
		for i := range src {
			src[i] = float32(i) - 500
		}
		want := make([]uint16, n)
		for i, v := range src {
			want[i] = float16.Fromfloat32(v).Bits()
		}
		got := make([]uint16, n)
		convertF32ToF16(got, src)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("mismatch at %d: got %v want %v", i, got[i], want[i])
			}
		}
	})

	t.Run("BF16ToF32", func(t *testing.T) {
		src := make([]uint16, n)
		for i := range src {
			src[i] = uint16(math.Float32bits(float32(i)-500) >> 16)
		}
		want := make([]float32, n)
		for i, v := range src {
			want[i] = math.Float32frombits(uint32(v) << 16)
		}
		got := make([]float32, n)
		convertBF16ToF32(got, src)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("mismatch at %d: got %v want %v", i, got[i], want[i])
			}
		}
	})

	t.Run("F32ToBF16", func(t *testing.T) {
		src := make([]float32, n)
		for i := range src {
			src[i] = float32(i) - 500
		}
		want := make([]uint16, n)
		for i, v := range src {
			want[i] = uint16(math.Float32bits(v) >> 16)
		}
		got := make([]uint16, n)
		convertF32ToBF16(got, src)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("mismatch at %d: got %v want %v", i, got[i], want[i])
			}
		}
	})

	t.Run("WriteTo_BF16_roundtrip", func(t *testing.T) {
		dir := t.TempDir()
		f, err := os.Create(dir + "/test.bin")
		if err != nil {
			t.Fatal(err)
		}
		vals := make([]uint16, n)
		for i := range vals {
			vals[i] = uint16(math.Float32bits(float32(i)*0.1) >> 16)
		}
		binary.Write(f, binary.LittleEndian, vals)
		f.Close()

		st := safetensor{
			fs:    os.DirFS(dir),
			path:  "test.bin",
			dtype: "BF16",
			size:  int64(n * 2),
			tensorBase: &tensorBase{
				name:  "test",
				shape: []uint64{uint64(n)},
			},
		}

		var buf bytes.Buffer
		st.WriteTo(&buf)

		got := make([]float32, n)
		binary.Read(bytes.NewReader(buf.Bytes()), binary.LittleEndian, got)

		for i := range vals {
			want := math.Float32frombits(uint32(vals[i]) << 16)
			if got[i] != want {
				t.Fatalf("mismatch at %d: got %v want %v", i, got[i], want)
			}
		}
	})
}
