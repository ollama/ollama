package convert

import (
	"encoding/binary"
	"encoding/json"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/x448/float16"
)

const benchElems = 1 << 20 // 1M elements per tensor

func writeSafetensorsFile(b *testing.B, dir string, dtype string, nelems int) string {
	b.Helper()

	var elemSize int
	switch dtype {
	case "F32":
		elemSize = 4
	case "F16", "BF16":
		elemSize = 2
	}
	dataBytes := nelems * elemSize

	header := map[string]safetensorMetadata{
		"weight": {
			Type:    dtype,
			Shape:   []uint64{uint64(nelems)},
			Offsets: []int64{0, int64(dataBytes)},
		},
	}
	headerJSON, _ := json.Marshal(header)
	headerLen := int64(len(headerJSON))

	path := filepath.Join(dir, "model.safetensors")
	f, err := os.Create(path)
	if err != nil {
		b.Fatal(err)
	}
	defer f.Close()

	binary.Write(f, binary.LittleEndian, headerLen)
	f.Write(headerJSON)

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

	return path
}

func benchRealWriteTo(b *testing.B, dtype string, shape []uint64) {
	b.Helper()
	dir := b.TempDir()
	writeSafetensorsFile(b, dir, dtype, benchElems)

	fsys := os.DirFS(dir)
	replacer := nopReplacer()
	tensors, cleanup, err := parseSafetensors(fsys, replacer, "model.safetensors")
	if cleanup != nil {
		defer cleanup()
	}
	if err != nil {
		b.Fatal(err)
	}
	if len(tensors) == 0 {
		b.Fatal("no tensors parsed")
	}

	// override shape to control output kind (safetensor is a value type from parseSafetensors)
	if st, ok := tensors[0].(safetensor); ok {
		st.shape = shape
		tensors[0] = st
	} else if st, ok := tensors[0].(*safetensor); ok {
		st.shape = shape
	}

	var elemSize int
	switch dtype {
	case "F32":
		elemSize = 4
	case "F16", "BF16":
		elemSize = 2
	}
	b.SetBytes(int64(benchElems * elemSize))
	b.ReportAllocs()
	b.ResetTimer()

	for range b.N {
		_, err := tensors[0].WriteTo(io.Discard)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func nopReplacer() *strings.Replacer {
	return strings.NewReplacer()
}

// F16 passthrough: F16 in, F16 out (2D shape → Kind=FP16)
func BenchmarkRealWriteTo_F16_Passthrough(b *testing.B) {
	benchRealWriteTo(b, "F16", []uint64{1024, uint64(benchElems / 1024)})
}

// F16 → F32 conversion: F16 in, F32 out (1D shape → Kind=FP32)
func BenchmarkRealWriteTo_F16_to_F32(b *testing.B) {
	benchRealWriteTo(b, "F16", []uint64{uint64(benchElems)})
}

// BF16 passthrough: BF16 in, BF16 out (2D shape → Kind=BF16)
func BenchmarkRealWriteTo_BF16_Passthrough(b *testing.B) {
	benchRealWriteTo(b, "BF16", []uint64{1024, uint64(benchElems / 1024)})
}

// BF16 → F32 conversion: BF16 in, F32 out (1D shape → Kind=FP32)
func BenchmarkRealWriteTo_BF16_to_F32(b *testing.B) {
	benchRealWriteTo(b, "BF16", []uint64{uint64(benchElems)})
}

// F32 → F16 conversion: F32 in, F16 out (2D shape → Kind=FP16)
func BenchmarkRealWriteTo_F32_to_F16(b *testing.B) {
	benchRealWriteTo(b, "F32", []uint64{1024, uint64(benchElems / 1024)})
}
