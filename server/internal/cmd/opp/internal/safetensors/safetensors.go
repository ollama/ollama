// safetensors provides a reader for the safetensor directories and files.
package safetensors

import (
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"iter"
	"slices"
	"strconv"
	"strings"
)

// Tensor represents a single tensor in a safetensors file.
//
// It's zero value is not valid. Use [Model.Tensors] to get valid tensors.
//
// It is not safe for use across multiple goroutines.
type Tensor struct {
	name     string
	dataType string
	shape    []int64

	fsys   fs.FS
	fname  string // entry name in fsys
	offset int64
	size   int64
}

type Model struct {
	fsys fs.FS
}

func Read(fsys fs.FS) (*Model, error) {
	return &Model{fsys: fsys}, nil
}

func (m *Model) Tensors() iter.Seq2[*Tensor, error] {
	return func(yield func(*Tensor, error) bool) {
		entries, err := fs.Glob(m.fsys, "*.safetensors")
		if err != nil {
			yield(nil, err)
			return
		}
		for _, e := range entries {
			tt, err := m.readTensors(e)
			if err != nil {
				yield(nil, err)
				return
			}
			for _, t := range tt {
				if !yield(t, nil) {
					return
				}
			}
		}
	}
}

func (m *Model) readTensors(fname string) ([]*Tensor, error) {
	f, err := m.fsys.Open(fname)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	finfo, err := f.Stat()
	if err != nil {
		return nil, err
	}

	headerSize, err := readInt64(f)
	if err != nil {
		return nil, err
	}

	data := make([]byte, headerSize)
	_, err = io.ReadFull(f, data)
	if err != nil {
		return nil, err
	}

	var raws map[string]json.RawMessage
	if err := json.Unmarshal(data, &raws); err != nil {
		return nil, err
	}

	endOfHeader := 8 + headerSize // 8 bytes for header size plus the header itself

	// TODO(bmizerany): do something with metadata? This could be another
	// header read if needed. We also need to figure out if the metadata is
	// present in only one .safetensors file or if each file may have their
	// own and if it needs to follow each tensor. Currently, I (bmizerany)
	// am only seeing them show up with one entry for file type which is
	// always "pt".

	tt := make([]*Tensor, 0, len(raws))
	for name, raw := range raws {
		if name == "__metadata__" {
			// TODO(bmizerany): do something with metadata?
			continue
		}
		var v struct {
			DataType string  `json:"dtype"`
			Shape    []int64 `json:"shape"`
			Offsets  []int64 `json:"data_offsets"`
		}
		if err := json.Unmarshal(raw, &v); err != nil {
			return nil, fmt.Errorf("error unmarshalling layer %q: %w", name, err)
		}
		if len(v.Offsets) != 2 {
			return nil, fmt.Errorf("invalid offsets for %q: %v", name, v.Offsets)
		}

		// TODO(bmizerany): after collecting, validate all offests make
		// tensors contiguous?
		begin := endOfHeader + v.Offsets[0]
		end := endOfHeader + v.Offsets[1]
		if err := checkBeginEnd(finfo.Size(), begin, end); err != nil {
			return nil, err
		}

		// TODO(bmizerany): just yield.. don't be silly and make a slice :)
		tt = append(tt, &Tensor{
			name:     name,
			dataType: v.DataType,
			shape:    v.Shape,
			fsys:     m.fsys,
			fname:    fname,
			offset:   begin,
			size:     end - begin,
		})
	}
	return tt, nil
}

func checkBeginEnd(size, begin, end int64) error {
	if begin < 0 {
		return fmt.Errorf("begin must not be negative: %d", begin)
	}
	if end < 0 {
		return fmt.Errorf("end must not be negative: %d", end)
	}
	if end < begin {
		return fmt.Errorf("end must be >= begin: %d < %d", end, begin)
	}
	if end > size {
		return fmt.Errorf("end must be <= size: %d > %d", end, size)
	}
	return nil
}

func readInt64(r io.Reader) (int64, error) {
	var v uint64
	var buf [8]byte
	if _, err := io.ReadFull(r, buf[:]); err != nil {
		return 0, err
	}
	for i := range buf {
		v |= uint64(buf[i]) << (8 * i)
	}
	return int64(v), nil
}

type Shape []int64

func (s Shape) String() string {
	var b strings.Builder
	b.WriteByte('[')
	for i, v := range s {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(strconv.FormatInt(v, 10))
	}
	b.WriteByte(']')
	return b.String()
}

func (t *Tensor) Name() string     { return t.name }
func (t *Tensor) DataType() string { return t.dataType }
func (t *Tensor) Size() int64      { return t.size }
func (t *Tensor) Shape() Shape     { return slices.Clone(t.shape) }

func (t *Tensor) Reader() (io.ReadCloser, error) {
	f, err := t.fsys.Open(t.fname)
	if err != nil {
		return nil, err
	}
	r := newSectionReader(f, t.offset, t.size)
	rc := struct {
		io.Reader
		io.Closer
	}{r, f}
	return rc, nil
}

// newSectionReader returns a new io.Reader that reads from r starting at
// offset. It is a convenience function for creating a io.SectionReader when r
// may not be an io.ReaderAt.
//
// If r is already a ReaderAt, it is returned directly, otherwise if r is an
// io.Seeker, a new io.ReaderAt is returned that wraps r after seeking to the
// beginning of the file.
//
// If r is an io.Seeker,
// or slow path. The slow path is used when r does not implement io.ReaderAt,
// in which case it must discard the data it reads.
func newSectionReader(r io.Reader, offset, n int64) io.Reader {
	if r, ok := r.(io.ReaderAt); ok {
		return io.NewSectionReader(r, offset, n)
	}
	if r, ok := r.(io.ReadSeeker); ok {
		r.Seek(offset, io.SeekStart)
		return io.LimitReader(r, n)
	}
	// Discard to offset and return a limited reader.
	_, err := io.CopyN(io.Discard, r, offset)
	if err != nil {
		return nil
	}
	return io.LimitReader(r, n)
}
