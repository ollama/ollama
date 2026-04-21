package laguna

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/d4l3k/go-bfloat16"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/testutil"
	"github.com/ollama/ollama/x/safetensors"
	"github.com/x448/float16"
)

func loadLagunaReferenceTensor(t *testing.T, path, name string) (*mlx.Array, func()) {
	t.Helper()
	testutil.SkipIfNoMLX(t)

	if _, err := os.Stat(path); err != nil {
		t.Skipf("reference data not available: %s", path)
	}

	ext, err := safetensors.OpenForExtraction(path)
	if err != nil {
		t.Fatalf("open reference %s: %v", path, err)
	}
	defer ext.Close()

	td, err := ext.GetTensor(name)
	if err != nil {
		t.Fatalf("reference tensor %q missing from %s: %v", name, path, err)
	}

	tmpPath := filepath.Join(t.TempDir(), sanitizeLagunaTensorName(name)+".safetensors")
	f, err := os.Create(tmpPath)
	if err != nil {
		t.Fatalf("create temp reference tensor %s: %v", tmpPath, err)
	}
	if _, err := io.Copy(f, td.SafetensorsReader()); err != nil {
		f.Close()
		t.Fatalf("write temp reference tensor %s: %v", tmpPath, err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("close temp reference tensor %s: %v", tmpPath, err)
	}

	var arr *mlx.Array
	for loadedName, loadedArr := range mlx.Load(tmpPath) {
		if loadedName == name {
			arr = loadedArr
			break
		}
	}
	if arr == nil {
		t.Fatalf("failed to load reference tensor %q from %s", name, tmpPath)
	}

	mlx.Pin(arr)
	mlx.Sweep()
	mlx.Eval(arr)

	released := false
	return arr, func() {
		if released {
			return
		}
		released = true
		mlx.Unpin(arr)
		mlx.Sweep()
	}
}

func loadLagunaReferenceTensorRowFloat32(t *testing.T, path, name string, row int) []float32 {
	t.Helper()

	if _, err := os.Stat(path); err != nil {
		t.Skipf("reference data not available: %s", path)
	}

	ext, err := safetensors.OpenForExtraction(path)
	if err != nil {
		t.Fatalf("open reference %s: %v", path, err)
	}
	defer ext.Close()

	td, err := ext.GetTensor(name)
	if err != nil {
		t.Fatalf("reference tensor %q missing from %s: %v", name, path, err)
	}
	if len(td.Shape) != 3 || td.Shape[0] != 1 {
		t.Fatalf("reference tensor %q shape = %v, want [1 L V]", name, td.Shape)
	}

	L, V := int(td.Shape[1]), int(td.Shape[2])
	if row < 0 || row >= L {
		t.Fatalf("reference tensor %q row %d out of range [0,%d)", name, row, L)
	}

	elemSize, err := lagunaTensorDTypeSize(td.Dtype)
	if err != nil {
		t.Fatalf("dtype size for %q: %v", td.Dtype, err)
	}
	rowBytes := V * elemSize
	offset := int64(row * rowBytes)

	readerAt, ok := td.Reader().(io.ReaderAt)
	if !ok {
		t.Fatalf("reference tensor %q reader does not support ReaderAt", name)
	}

	raw := make([]byte, rowBytes)
	if _, err := readerAt.ReadAt(raw, offset); err != nil {
		t.Fatalf("read row %d from %q: %v", row, name, err)
	}

	vals, err := decodeLagunaReferenceFloatTensor(td.Dtype, raw)
	if err != nil {
		t.Fatalf("decode row %d from %q: %v", row, name, err)
	}
	return vals
}

func sanitizeLagunaTensorName(name string) string {
	replacer := strings.NewReplacer("/", "_", "\\", "_", ".", "_", ":", "_")
	return replacer.Replace(name)
}

func lagunaTensorDTypeSize(dtype string) (int, error) {
	switch strings.ToUpper(dtype) {
	case "BF16", "F16":
		return 2, nil
	case "F32", "U32", "I32":
		return 4, nil
	case "F64":
		return 8, nil
	default:
		return 0, fmt.Errorf("unsupported dtype %q", dtype)
	}
}

func decodeLagunaReferenceFloatTensor(dtype string, raw []byte) ([]float32, error) {
	switch strings.ToUpper(dtype) {
	case "BF16":
		return bfloat16.DecodeFloat32(raw), nil
	case "F16":
		if len(raw)%2 != 0 {
			return nil, fmt.Errorf("invalid f16 byte length %d", len(raw))
		}
		values := make([]float32, len(raw)/2)
		for i := range values {
			values[i] = float16.Frombits(binary.LittleEndian.Uint16(raw[i*2:])).Float32()
		}
		return values, nil
	case "F32":
		if len(raw)%4 != 0 {
			return nil, fmt.Errorf("invalid f32 byte length %d", len(raw))
		}
		values := make([]float32, len(raw)/4)
		for i := range values {
			values[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
		return values, nil
	default:
		return nil, fmt.Errorf("unsupported dtype %q", dtype)
	}
}

func loadLagunaModelFromDir(t *testing.T, modelDir string) *Model {
	t.Helper()
	tracePhasef(t, "load model wrapper begin dir=%s", modelDir)
	bm := testutil.LoadModelFromDir(t, modelDir)
	tracePhasef(t, "load model wrapper returned type=%T", bm)
	m, ok := bm.(*Model)
	if !ok {
		t.Fatalf("expected *laguna.Model, got %T", bm)
	}
	kept := mlx.Collect(m)
	if len(kept) > 0 {
		mlx.Pin(kept...)
	}
	mlx.Sweep()
	return m
}
