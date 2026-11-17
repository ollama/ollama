package mlx

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx/mlxthread/mlxthreadtest"
)

func TestTensorFileMapping(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tensor")
	want := bytes.Repeat([]byte{1, 2, 3, 4}, TensorFileAlignment/4)
	contents := append(make([]byte, TensorFileAlignment), want...)
	if err := os.WriteFile(path, contents, 0o600); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	data, err := mapTensorFile(f, TensorFileAlignment, len(want))
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if data != nil {
			unmapTensorFile(data)
		}
	}()
	// A view outlives the file handle on both Unix and Windows, but must be
	// released before eviction can remove the file on Windows.
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(data, want) {
		t.Fatal("mapping read the wrong file extent")
	}
	if err := unmapTensorFile(data); err != nil {
		t.Fatal(err)
	}
	data = nil
	if err := os.Remove(path); err != nil {
		t.Fatalf("unmapped view still pins its file: %v", err)
	}
}

func TestTensorFilePublication(t *testing.T) {
	for _, publish := range []bool{false, true} {
		name := "abandon"
		if publish {
			name = "publish"
		}
		t.Run(name, func(t *testing.T) {
			f, err := NewTensorFile(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()
			pending := f.Path()
			if err := f.Publish(); err == nil {
				t.Fatal("published an unfinished checkpoint")
			}
			metadata := []byte("complete checkpoint")
			if err := f.Seal(metadata); err != nil {
				t.Fatal(err)
			}
			if _, err := os.Stat(pending + ".kv"); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("checkpoint visible before publication: %v", err)
			}
			if publish {
				if err := f.Publish(); err != nil {
					t.Fatal(err)
				}
				if err := f.Publish(); err == nil {
					t.Fatal("published the same file twice")
				}
			}
			if err := f.Close(); err != nil {
				t.Fatal(err)
			}
			if _, err := os.Stat(pending); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("staging file left behind: %v", err)
			}
			if err := f.Publish(); !errors.Is(err, os.ErrClosed) {
				t.Fatalf("publication after close: %v", err)
			}
			reader, got, err := OpenTensorFile(pending + ".kv")
			if !publish {
				if !errors.Is(err, os.ErrNotExist) {
					t.Fatalf("abandoned checkpoint remains: %v", err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			defer reader.Close()
			if !bytes.Equal(got, metadata) {
				t.Fatalf("metadata = %q, want %q", got, metadata)
			}
		})
	}
}

func TestTensorFileRestoresOwnedStorage(t *testing.T) {
	dir := t.TempDir()
	withMLXThread(t, func(t *mlxthreadtest.T) {
		f, err := NewTensorFile(dir)
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		want := []int32{11, 22, 33, 44}
		extent, err := appendTensorFile(f, context.Background(), FromValues(want, 2, 2))
		if err != nil {
			t.Fatal(err)
		}
		if err := f.Seal([]byte("owned storage")); err != nil {
			t.Fatal(err)
		}
		if err := f.Publish(); err != nil {
			t.Fatal(err)
		}
		got, err := f.Read(context.Background(), extent)
		if err != nil {
			t.Fatal(err)
		}
		// Changing a shared mapping would change an aliased restore even after
		// Eval. Keep the array alive while deliberately changing its source file.
		file, err := os.OpenFile(f.Path(), os.O_RDWR, 0)
		if err != nil {
			t.Fatal(err)
		}
		defer file.Close()
		var changed [4]byte
		binary.LittleEndian.PutUint32(changed[:], 99)
		if _, err := file.WriteAt(changed[:], extent.offset); err != nil {
			t.Fatal(err)
		}
		if values := got.Ints(); !slices.Equal(values, want) {
			t.Fatalf("restored tensor still aliases its file: %v, want %v", values, want)
		}
		Scoped(func() {
			got.Set(got.SliceUpdate(Zeros(DTypeInt32, 2, 1), Slice(), Slice(0, 1)))
		})
		Eval(got)
		if values := got.Ints(); !slices.Equal(values, []int32{0, 22, 0, 44}) {
			t.Fatalf("restored tensor is not independently writable: %v", values)
		}
		var after [4]byte
		if _, err := file.ReadAt(after[:], extent.offset); err != nil {
			t.Fatal(err)
		}
		if after != changed {
			t.Fatal("updating the restored tensor modified its source file")
		}
		if err := file.Close(); err != nil {
			t.Fatal(err)
		}
		if err := os.Remove(f.Path()); err != nil {
			t.Fatalf("restored tensor pins the file: %v", err)
		}
	})
}

func TestTensorFilePreservesBitsAndStrides(t *testing.T) {
	dir := t.TempDir()
	withMLXThread(t, func(t *mlxthreadtest.T) {
		f, err := NewTensorFile(dir)
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		bits := []uint32{0, 0x80000000, 0x7fc01234, 0xffc05678, 0x3f800000, 0x40000000, 0x40400000, 0x40800000}
		source := FromValues(bits, 2, 4).view(DTypeFloat32).Slice(Slice(), Slice(1, 3))
		extent, err := appendTensorFile(f, context.Background(), source)
		if err != nil {
			t.Fatal(err)
		}
		for range 2 {
			got, err := f.Read(context.Background(), extent)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(got.Dims(), []int{2, 2}) || got.DType() != DTypeFloat32 {
				t.Fatalf("restored layout = %v %v", got.Dims(), got.DType())
			}
			want := FromValues([]uint32{bits[1], bits[2], bits[5], bits[6]}, 2, 2)
			if !slices.Equal(got.view(DTypeInt32).Ints(), want.view(DTypeInt32).Ints()) {
				t.Fatal("spill changed tensor bits")
			}
			// A restored array is independent; its in-place update must not
			// change what the next restore reads from the mapping.
			got.Set(got.SliceUpdate(Zeros(DTypeFloat32, 2, 1), Slice(), Slice(0, 1)))
			Eval(got)
		}
		if err := f.Close(); err != nil {
			t.Fatal(err)
		}
		if _, err := f.Read(context.Background(), extent); !errors.Is(err, os.ErrClosed) {
			t.Fatalf("read closed file: %v", err)
		}
	})
	entries, err := os.ReadDir(dir)
	if err != nil || len(entries) != 0 {
		t.Fatalf("spill files left behind: %v, %v", entries, err)
	}
}

func TestTensorFileDTypes(t *testing.T) {
	for _, dtype := range []DType{DTypeBool, DTypeUint8, DTypeInt8, DTypeUint16, DTypeInt16, DTypeFloat16, DTypeBFloat16, DTypeUint32, DTypeInt32, DTypeFloat32, DTypeUint64, DTypeInt64, DTypeFloat64, DTypeComplex64} {
		t.Run(dtype.String(), func(t *testing.T) {
			dir := t.TempDir()
			withMLXThread(t, func(t *mlxthreadtest.T) {
				if dtype == DTypeFloat64 && GPUIsAvailable() {
					t.Skip("MLX does not support float64 operations on GPU streams")
				}
				bits := make([]uint8, 48)
				for i := range bits {
					bits[i] = uint8(i*73 + 7)
					if dtype == DTypeBool {
						bits[i] &= 1
					}
				}
				source := FromValues(bits, 2, 24).view(dtype)
				f, err := NewTensorFile(dir)
				if err != nil {
					t.Fatal(err)
				}
				defer f.Close()
				extent, err := appendTensorFile(f, context.Background(), source)
				if err != nil {
					t.Fatal(err)
				}
				got, err := f.Read(context.Background(), extent)
				if err != nil {
					t.Fatal(err)
				}
				if got.DType() != dtype || !slices.Equal(got.Dims(), source.Dims()) {
					t.Fatalf("restored layout %v %v, want %v %v", got.DType(), got.Dims(), dtype, source.Dims())
				}
				for i, value := range got.view(DTypeUint8).AsType(DTypeInt32).Ints() {
					if value != int32(bits[i]) {
						t.Fatalf("byte %d = %d, want %d", i, value, bits[i])
					}
				}
			})
		})
	}
}

func TestTensorFileInterruptedWrite(t *testing.T) {
	dir := t.TempDir()
	withMLXThread(t, func(t *mlxthreadtest.T) {
		f, err := NewTensorFile(dir)
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		a := FromValues([]int32{1, 2, 3, 4}, 2, 2)
		w, err := f.BeginTensor(a)
		if err != nil {
			t.Fatal(err)
		}
		if err := f.Seal([]byte("incomplete")); err == nil {
			t.Fatal("published an incomplete tensor")
		}
		if _, err := f.BeginTensor(a); err == nil {
			t.Fatal("began another tensor before finishing the first")
		}
		data, err := w.Stage(context.Background(), a)
		if err != nil {
			t.Fatal(err)
		}
		if err := f.Close(); err != nil {
			t.Fatal(err)
		}
		if err := w.Write(context.Background(), data); !errors.Is(err, os.ErrClosed) {
			t.Fatalf("write after close: %v", err)
		}
		if _, err := os.Stat(f.Path()); !os.IsNotExist(err) {
			t.Fatalf("unfinished file remains: %v", err)
		}
	})
}

func TestTensorFileScalarAndMultipleArrays(t *testing.T) {
	dir := t.TempDir()
	withMLXThread(t, func(t *mlxthreadtest.T) {
		f, err := NewTensorFile(dir)
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		arrays := []*Array{FromValue(7), FromValues([]int32{9, 8, 7, 6}, 2, 2)}
		var extents []TensorExtent
		for _, a := range arrays {
			e, err := appendTensorFile(f, context.Background(), a)
			if err != nil {
				t.Fatal(err)
			}
			extents = append(extents, e)
		}
		for i, e := range extents {
			b, err := f.Read(context.Background(), e)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(arrays[i].Dims(), b.Dims()) || !slices.Equal(arrays[i].Ints(), b.Ints()) {
				t.Fatalf("array %d did not round trip", i)
			}
		}
	})
}

func TestTensorExtentValidation(t *testing.T) {
	valid := TensorExtent{shape: []int{1, 2}, dtype: DTypeFloat32, padded: TensorFileAlignment}
	if err := valid.Validate(TensorFileAlignment); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name string
		edit func(*TensorExtent)
	}{
		{"negative offset", func(e *TensorExtent) { e.offset = -TensorFileAlignment }},
		{"unaligned offset", func(e *TensorExtent) { e.offset = 1 }},
		{"past end", func(e *TensorExtent) { e.offset = TensorFileAlignment }},
		{"missing padding", func(e *TensorExtent) { e.padded = 8 }},
		{"excess padding", func(e *TensorExtent) { e.padded *= 2 }},
		{"unknown dtype", func(e *TensorExtent) { e.dtype = DType(-1) }},
		{"zero dimension", func(e *TensorExtent) { e.shape = []int{0} }},
		{"negative dimension", func(e *TensorExtent) { e.shape = []int{-1} }},
		{"overflow", func(e *TensorExtent) { e.shape = []int{math.MaxInt32, math.MaxInt32, math.MaxInt32} }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			e := valid
			tc.edit(&e)
			if err := e.Validate(TensorFileAlignment); err == nil {
				t.Fatal("invalid descriptor accepted")
			}
		})
	}
}

func TestTensorFileTruncatedMetadata(t *testing.T) {
	f, err := NewTensorFile(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := f.Seal([]byte("checkpoint")); err != nil {
		t.Fatal(err)
	}
	if err := f.Publish(); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(f.Path())
	if err != nil {
		t.Fatal(err)
	}
	for n := range data {
		name := filepath.Join(t.TempDir(), "truncated.kv")
		if err := os.WriteFile(name, data[:n], 0o600); err != nil {
			t.Fatal(err)
		}
		if got, _, err := OpenTensorFile(name); err == nil {
			got.Close()
			t.Fatalf("accepted checkpoint truncated at %d", n)
		}
	}
}

func TestTensorFileBF16AndCanceledIO(t *testing.T) {
	dir := t.TempDir()
	withMLXThread(t, func(t *mlxthreadtest.T) {
		f, err := NewTensorFile(dir)
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		a := FromValues([]uint16{0x8000, 0x7fc1, 0x3f80, 0xffc2}, 1, 4).view(DTypeBFloat16)
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if _, err := appendTensorFile(f, ctx, a); !errors.Is(err, context.Canceled) {
			t.Fatalf("canceled write: %v", err)
		}
		e, err := appendTensorFile(f, context.Background(), a)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := f.Read(ctx, e); !errors.Is(err, context.Canceled) {
			t.Fatalf("canceled read: %v", err)
		}
		b, err := f.Read(context.Background(), e)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(a.view(DTypeUint16).AsType(DTypeInt32).Ints(), b.view(DTypeUint16).AsType(DTypeInt32).Ints()) {
			t.Fatal("BF16 payload changed")
		}
	})
}

func TestTensorFileChunksStridedArray(t *testing.T) {
	dir := t.TempDir()
	withMLXThread(t, func(t *mlxthreadtest.T) {
		// The logical tensor crosses the 8 MiB write boundary. Cropping
		// columns leaves gaps between rows, including at that boundary.
		const rows, columns = 2049, 1027
		values := make([]int32, rows*columns)
		for i := range values {
			values[i] = int32(i) ^ -123456789
		}
		source := FromValues(values, 1, rows, columns).Slice(Slice(), Slice(), Slice(1, columns-1))
		f, err := NewTensorFile(dir)
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		extent, err := appendTensorFile(f, context.Background(), source)
		if err != nil {
			t.Fatal(err)
		}
		restored, err := f.Read(context.Background(), extent)
		if err != nil {
			t.Fatal(err)
		}
		got := restored.Ints()
		if len(got) != rows*(columns-2) {
			t.Fatalf("restored elements = %d, want %d", len(got), rows*(columns-2))
		}
		for i, value := range got {
			row, column := i/(columns-2), i%(columns-2)+1
			if want := values[row*columns+column]; value != want {
				t.Fatalf("element %d = %d, want %d", i, value, want)
			}
		}
	})
}

func TestTensorFilePersistentRecoveryAndCorruption(t *testing.T) {
	dir := t.TempDir()
	withMLXThread(t, func(t *mlxthreadtest.T) {
		f, err := NewTensorFile(dir)
		if err != nil {
			t.Fatal(err)
		}
		source := FromValues([]int32{1, -2, 3, -4}, 2, 2)
		extent, err := appendTensorFile(f, context.Background(), source)
		if err != nil {
			t.Fatal(err)
		}
		metadata, err := json.Marshal(extent)
		if err != nil {
			t.Fatal(err)
		}
		if err := f.Seal(metadata); err != nil {
			t.Fatal(err)
		}
		if err := f.Publish(); err != nil {
			t.Fatal(err)
		}
		path := f.Path()
		f.Close()
		reopened, record, err := OpenTensorFile(path)
		if err != nil {
			t.Fatal(err)
		}
		defer reopened.Close()
		var restoredExtent TensorExtent
		if err := json.Unmarshal(record, &restoredExtent); err != nil {
			t.Fatal(err)
		}
		got, err := reopened.Read(context.Background(), restoredExtent)
		if err != nil || !slices.Equal(got.Ints(), source.Ints()) {
			t.Fatalf("persistent roundtrip: %v", err)
		}
		corrupt, err := os.OpenFile(path, os.O_RDWR, 0)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := corrupt.WriteAt([]byte{99}, 0); err != nil {
			t.Fatal(err)
		}
		corrupt.Close()
		if _, err := reopened.Read(context.Background(), restoredExtent); err == nil {
			t.Fatal("corrupt tensor accepted")
		}
		bad := restoredExtent
		bad.shape = []int{math.MaxInt32, math.MaxInt32, math.MaxInt32}
		if _, err := reopened.Read(context.Background(), bad); err == nil {
			t.Fatal("overflowing tensor shape accepted")
		}
		reopened.Close()
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		data[len(data)-1] ^= 1
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatal(err)
		}
		if f, _, err := OpenTensorFile(path); err == nil {
			f.Close()
			t.Fatal("corrupt metadata accepted")
		}
	})
}

func appendTensorFile(f *TensorFile, ctx context.Context, a *Array) (TensorExtent, error) {
	if err := ctx.Err(); err != nil {
		return TensorExtent{}, err
	}
	w, err := f.BeginTensor(a)
	if err != nil {
		return TensorExtent{}, err
	}
	for !w.Done() {
		data, err := w.Stage(ctx, a)
		if err != nil {
			return TensorExtent{}, err
		}
		if err := w.Write(ctx, data); err != nil {
			return TensorExtent{}, err
		}
	}
	return w.Extent(), nil
}
