package mlx

// #include "generated.h"
import "C"

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash"
	"io"
	"math"
	"os"
	"slices"
	"unsafe"
)

// TensorFile stores exact immutable tensor bytes, alignment padding and a
// checksummed checkpoint record. Sealed files open handles only while reading,
// so cache capacity does not consume one file descriptor per checkpoint.
// All array operations run on the owning MLX thread.
type TensorFile struct {
	path      string
	file      *os.File
	size      int64
	writer    *TensorWriter
	sealed    bool
	published bool
	closed    bool
}

// TensorExtent describes one tensor in a TensorFile. A persisted descriptor
// is validated against its containing file before mapping any data.
type TensorExtent struct {
	offset int64
	shape  []int
	dtype  DType
	padded int
	digest [sha256.Size]byte
}

// MarshalJSON records only the portable tensor descriptor, never an MLX handle.
func (e TensorExtent) MarshalJSON() ([]byte, error) {
	return json.Marshal(tensorDescriptor{e.offset, e.shape, int(e.dtype), e.padded, e.digest})
}

func (e *TensorExtent) UnmarshalJSON(data []byte) error {
	var d tensorDescriptor
	if err := json.Unmarshal(data, &d); err != nil {
		return err
	}
	*e = TensorExtent{d.Offset, d.Shape, DType(d.Type), d.Padded, d.Digest}
	return nil
}

type tensorDescriptor struct {
	Offset int64
	Shape  []int
	Type   int
	Padded int
	Digest [sha256.Size]byte
}

func (e TensorExtent) byteSize() (int64, error) {
	var width int64
	switch e.dtype {
	case DTypeBool, DTypeUint8, DTypeInt8:
		width = 1
	case DTypeUint16, DTypeInt16, DTypeFloat16, DTypeBFloat16:
		width = 2
	case DTypeUint32, DTypeInt32, DTypeFloat32:
		width = 4
	case DTypeUint64, DTypeInt64, DTypeFloat64, DTypeComplex64:
		width = 8
	default:
		return 0, fmt.Errorf("invalid tensor dtype %d", e.dtype)
	}
	if len(e.shape) > 8 {
		return 0, fmt.Errorf("invalid tensor rank %d", len(e.shape))
	}
	for _, dim := range e.shape {
		if dim <= 0 || dim > math.MaxInt32 || width > math.MaxInt64/int64(dim) {
			return 0, fmt.Errorf("invalid tensor dimensions")
		}
		width *= int64(dim)
	}
	return width, nil
}

// Validate rejects malformed dimensions and extents before calling native code.
func (e TensorExtent) Validate(size int64) error {
	if size < 0 {
		return fmt.Errorf("invalid tensor file size")
	}
	width, err := e.byteSize()
	if err != nil {
		return err
	}
	if e.offset < 0 || e.offset%TensorFileAlignment != 0 || e.padded <= 0 || e.padded%TensorFileAlignment != 0 || e.offset > size-int64(e.padded) || width > int64(e.padded) || int64(e.padded)-width >= TensorFileAlignment {
		return fmt.Errorf("tensor extent is outside its checkpoint file")
	}
	if e.padded/TensorFileAlignment > math.MaxInt32 {
		return fmt.Errorf("tensor mapping exceeds native dimension limit")
	}
	return nil
}

// TensorFileAlignment covers Windows mapping granularity and the supported
// platforms' VM page sizes.
const TensorFileAlignment = 64 << 10

// TensorFileMetadataLimit bounds record decoding and disk-space reservations.
const TensorFileMetadataLimit = 8 << 20

const (
	tensorFileMagic       = "OLLAMAK1"
	tensorFileFooterBytes = int64(len(tensorFileMagic) + 8 + sha256.Size)
	tensorChunkBytes      = 8 << 20
)

// NewTensorFile creates a private staging file. Seal finishes writing it and
// Publish makes it available for reuse. Close removes an unpublished file.
func NewTensorFile(dir string) (*TensorFile, error) {
	f, err := os.CreateTemp(dir, ".pending-*")
	if err != nil {
		return nil, err
	}
	return &TensorFile{file: f, path: f.Name()}, nil
}

func (f *TensorFile) Path() string { return f.path }

// Seal writes bounded metadata and flushes a complete checkpoint to disk.
// It performs only file I/O and may run on a writer goroutine after all tensor writes finish.
// Call it before taking any shared publication lock: flushing can be slow.
func (f *TensorFile) Seal(metadata []byte) error {
	if f.file == nil {
		return os.ErrClosed
	}
	if f.sealed || f.writer != nil || len(metadata) == 0 || len(metadata) > TensorFileMetadataLimit {
		return fmt.Errorf("invalid tensor file publication")
	}
	footer := make([]byte, tensorFileFooterBytes)
	copy(footer, tensorFileMagic)
	binary.LittleEndian.PutUint64(footer[8:], uint64(len(metadata)))
	digest := sha256.Sum256(metadata)
	copy(footer[16:], digest[:])
	if _, err := f.file.WriteAt(append(metadata, footer...), f.size); err != nil {
		return err
	}
	if err := f.file.Sync(); err != nil {
		return err
	}
	name := f.file.Name()
	// Close before rename so Windows does not require delete sharing on the
	// staging handle. Readers open the final name only after publication.
	if err := f.file.Close(); err != nil {
		f.file = nil
		_ = os.Remove(name)
		return err
	}
	f.file = nil
	f.sealed = true
	return nil
}

// Publish atomically exposes a sealed checkpoint. Callers can hold a shared
// quota lock across publication and reservation release without holding it
// during the data flush in Seal.
func (f *TensorFile) Publish() error {
	if f.closed {
		return os.ErrClosed
	}
	if !f.sealed || f.published {
		return fmt.Errorf("invalid tensor file publication")
	}
	final := f.path + ".kv"
	if err := os.Rename(f.path, final); err != nil {
		return err
	}
	f.path = final
	f.published = true
	return nil
}

// OpenTensorFile reads and verifies the small checkpoint record. Tensor bytes
// are verified when restored, so opening a cache never reads its entire payload.
func OpenTensorFile(path string) (*TensorFile, []byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return nil, nil, err
	}
	if !info.Mode().IsRegular() || info.Size() < tensorFileFooterBytes {
		return nil, nil, fmt.Errorf("invalid tensor file")
	}
	var footer [tensorFileFooterBytes]byte
	if _, err := f.ReadAt(footer[:], info.Size()-tensorFileFooterBytes); err != nil {
		return nil, nil, err
	}
	n := binary.LittleEndian.Uint64(footer[8:])
	if string(footer[:8]) != tensorFileMagic || n == 0 || n > TensorFileMetadataLimit || int64(n) > info.Size()-tensorFileFooterBytes {
		return nil, nil, fmt.Errorf("invalid tensor file footer")
	}
	size := info.Size() - tensorFileFooterBytes - int64(n)
	if size%TensorFileAlignment != 0 {
		return nil, nil, fmt.Errorf("invalid tensor file alignment")
	}
	metadata := make([]byte, int(n))
	if _, err := f.ReadAt(metadata, size); err != nil {
		return nil, nil, err
	}
	digest := sha256.Sum256(metadata)
	if string(digest[:]) != string(footer[16:]) {
		return nil, nil, fmt.Errorf("tensor metadata checksum mismatch")
	}
	return &TensorFile{path: path, size: size, sealed: true, published: true}, metadata, nil
}

func (f *TensorFile) Size() int64 { return f.size }

func (f *TensorFile) Close() error {
	f.closed = true
	var err error
	if f.file != nil {
		err = f.file.Close()
		f.file = nil
	}
	if !f.published {
		_ = os.Remove(f.path)
	}
	return err
}

// TensorWriter transfers one tensor through bounded, independently owned host
// chunks. Stage runs on the MLX worker; Write uses only Go memory and file I/O.
// The caller must finish Write before the next Stage or any other file operation.
type TensorWriter struct {
	file     *TensorFile
	extent   TensorExtent
	position int
	bytes    int
	digest   hash.Hash
}

func (f *TensorFile) BeginTensor(a *Array) (*TensorWriter, error) {
	if f.file == nil {
		return nil, os.ErrClosed
	}
	if f.sealed {
		return nil, fmt.Errorf("cannot append to a sealed tensor file")
	}
	if f.writer != nil {
		return nil, fmt.Errorf("previous tensor write is incomplete")
	}
	n := a.NumBytes()
	if n <= 0 || n > math.MaxInt-TensorFileAlignment || int64(n) > math.MaxInt64-f.size-TensorFileAlignment {
		return nil, fmt.Errorf("unsupported spill tensor size %d", n)
	}
	e := TensorExtent{offset: f.size, shape: a.Dims(), dtype: a.DType(), padded: (n + TensorFileAlignment - 1) / TensorFileAlignment * TensorFileAlignment}
	if err := e.Validate(f.size + int64(e.padded)); err != nil {
		return nil, err
	}
	f.size += int64(e.padded)
	f.writer = &TensorWriter{file: f, extent: e, bytes: n, digest: sha256.New()}
	return f.writer, nil
}

func (w *TensorWriter) Done() bool { return w.position == w.bytes }
func (w *TensorWriter) Extent() TensorExtent {
	e := w.extent
	copy(e.digest[:], w.digest.Sum(nil))
	return e
}

// Stage copies at most 8 MiB. Slicing precedes compaction so strided KV views
// never compact or migrate their entire parent allocation just to write a chunk.
func (w *TensorWriter) Stage(ctx context.Context, a *Array) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if w.file.file == nil {
		return nil, os.ErrClosed
	}
	if w.Done() {
		return nil, io.EOF
	}
	if !slices.Equal(a.Dims(), w.extent.shape) || a.DType() != w.extent.dtype {
		return nil, fmt.Errorf("tensor changed during persistence")
	}
	width := w.bytes / a.Size()
	linear := w.position / width
	dims := w.extent.shape
	axis, suffix := len(dims)-1, 1
	for axis > 0 && suffix*dims[axis]*width <= tensorChunkBytes {
		suffix *= dims[axis]
		axis--
	}
	var data []byte
	Scoped(func() {
		var part *Array
		if len(dims) > 0 {
			ranges := make([]slice, len(dims))
			stride := a.Size()
			remaining := linear
			for i, dim := range dims {
				stride /= dim
				at := remaining / stride
				remaining %= stride
				switch {
				case i < axis:
					ranges[i] = Slice(at, at+1)
				case i == axis:
					ranges[i] = Slice(at, min(dim, at+max(1, tensorChunkBytes/(suffix*width))))
				default:
					ranges[i] = Slice()
				}
			}
			part = a.Slice(ranges...)
		} else {
			// MLX bit views can change element width only on ranked arrays.
			part = a.Reshape(1)
		}
		// CPU access can migrate a shared CUDA allocation. Take an independent
		// chunk so saving a checkpoint cannot move the live cache off the GPU.
		b := part.ExpandDims(0).TakeAxis(FromValue(0), 0).view(DTypeUint8)
		Eval(b)
		ptr := mlxCheck(C.mlx_array_data_uint8(b.ctx))
		data = bytes.Clone(unsafe.Slice((*byte)(unsafe.Pointer(ptr)), b.NumBytes()))
	})
	return data, nil
}

func (w *TensorWriter) Write(ctx context.Context, data []byte) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if w.file.file == nil {
		return os.ErrClosed
	}
	if len(data) == 0 || len(data) > tensorChunkBytes || len(data) > w.bytes-w.position {
		return fmt.Errorf("invalid tensor chunk")
	}
	// Reserve alignment padding on the I/O worker, not during MLX staging.
	if w.position == 0 {
		if err := w.file.file.Truncate(w.extent.offset + int64(w.extent.padded)); err != nil {
			return err
		}
	}
	n, err := w.file.file.WriteAt(data, w.extent.offset+int64(w.position))
	if err != nil {
		return err
	}
	if n != len(data) {
		return io.ErrShortWrite
	}
	_, _ = w.digest.Write(data)
	w.position += n
	if w.Done() {
		w.file.writer = nil
	}
	return nil
}

// Read imports immutable file bytes into independently owned MLX storage.
// The native data constructor copies synchronously, so the mapping can be
// released before returning even when later GPU work is evaluated lazily.
func (f *TensorFile) Read(ctx context.Context, e TensorExtent) (*Array, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if f.closed {
		return nil, os.ErrClosed
	}
	if err := e.Validate(f.size); err != nil {
		return nil, err
	}
	file := f.file
	if f.sealed {
		var err error
		file, err = os.Open(f.path)
		if err != nil {
			return nil, err
		}
		defer file.Close()
	}
	if file == nil {
		return nil, os.ErrClosed
	}
	info, err := file.Stat()
	if err != nil {
		return nil, err
	}
	if info.Size() < f.size {
		return nil, fmt.Errorf("truncated tensor file")
	}
	data, err := mapTensorFile(file, e.offset, e.padded)
	if err != nil {
		return nil, err
	}
	defer unmapTensorFile(data)
	// Padding is not tensor data and is excluded from the checksum.
	n, _ := e.byteSize()
	if sha256.Sum256(data[:n]) != e.digest {
		return nil, fmt.Errorf("tensor checksum mismatch")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	shape := make([]C.int, len(e.shape))
	for i, dim := range e.shape {
		shape[i] = C.int(dim)
	}
	out := New("TENSOR_FILE")
	// mlx_copy and Contiguous may share their input's allocation. Use the
	// same copying constructor as FromValues; no file-backed storage escapes.
	out.ctx = C.mlx_array_new_data(unsafe.Pointer(unsafe.SliceData(data)), unsafe.SliceData(shape), C.int(len(shape)), C.mlx_dtype(e.dtype))
	if err := mlxError(out.ctx); err != nil {
		return nil, err
	}
	return out, nil
}

// view reinterprets storage without a numeric conversion, preserving NaN payloads
// and signed zero when moving tensor bytes to and from a file.
func (a *Array) view(dtype DType) *Array {
	out := New("TENSOR_VIEW")
	mlxCheck(C.mlx_view(&out.ctx, a.ctx, C.mlx_dtype(dtype), DefaultStream().ctx))
	return out
}
