package convert

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"maps"
	"slices"
	"strings"

	"github.com/d4l3k/go-bfloat16"
	"github.com/x448/float16"
)

type safetensorMetadata struct {
	Type    string   `json:"dtype"`
	Shape   []uint64 `json:"shape"`
	Offsets []int64  `json:"data_offsets"`
}

func parseSafetensors(fsys fs.FS, replacer *strings.Replacer, ps ...string) ([]Tensor, error) {
	var ts []Tensor
	for _, p := range ps {
		f, err := fsys.Open(p)
		if err != nil {
			return nil, err
		}
		defer f.Close()

		var n int64
		if err := binary.Read(f, binary.LittleEndian, &n); err != nil {
			return nil, err
		}

		b := bytes.NewBuffer(make([]byte, 0, n))
		if _, err = io.CopyN(b, f, n); err != nil {
			return nil, err
		}

		var headers map[string]safetensorMetadata
		if err := json.NewDecoder(b).Decode(&headers); err != nil {
			return nil, err
		}

		keys := slices.Sorted(maps.Keys(headers))

		names := make(map[string]struct{}, len(keys))

		for _, key := range keys {
			if value := headers[key]; value.Type != "" {
				// bitsandbytes quantized models are unsupported
				if len(value.Shape) == 0 {
					return nil, errors.New("unsupported safetensors model")
				}
				ggufName := replacer.Replace(key)
				if _, ok := names[ggufName]; ok {
					return nil, fmt.Errorf("duplicate tensor name '%s' was found for this model", ggufName)
				}
				names[ggufName] = struct{}{}
				ts = append(ts, safetensor{
					fs:     fsys,
					path:   p,
					dtype:  value.Type,
					offset: safetensorsPad(n, value.Offsets[0]),
					size:   safetensorsPad(n, value.Offsets[1]) - safetensorsPad(n, value.Offsets[0]),
					tensorBase: &tensorBase{
						name:  ggufName,
						shape: value.Shape,
					},
				})
			}
		}
	}

	return ts, nil
}

// safetensorsPad returns the padded size of the safetensors file given a length n and offset s
func safetensorsPad(n, offset int64) int64 {
	return 8 + n + offset
}

type safetensor struct {
	fs     fs.FS
	path   string
	dtype  string
	offset int64
	size   int64
	*tensorBase
}

func (st safetensor) Kind() uint32 {
	kind := st.tensorBase.Kind()
	if st.dtype == "BF16" &&
		!strings.HasPrefix(st.name, "v.") &&
		!strings.HasPrefix(st.name, "s.") &&
		kind != tensorKindFP32 {
		kind = tensorKindBF16
	}

	return kind
}

func (st safetensor) Clone() Tensor {
	return &safetensor{
		fs:     st.fs,
		path:   st.path,
		dtype:  st.dtype,
		offset: st.offset,
		size:   st.size,
		tensorBase: &tensorBase{
			name:     st.name,
			repacker: st.repacker,
			shape:    slices.Clone(st.shape),
		},
	}
}

func (st safetensor) WriteTo(w io.Writer) (int64, error) {
	f, err := st.fs.Open(st.path)
	if err != nil {
		return 0, err
	}
	defer f.Close()

	r, err := func() (io.Reader, error) {
		if readerAt, ok := f.(io.ReaderAt); ok {
			return io.NewSectionReader(readerAt, st.offset, st.size), nil
		} else if seeker, ok := f.(io.Seeker); ok {
			_, err := seeker.Seek(st.offset, io.SeekStart)
			return f, err
		} else {
			_, err := io.CopyN(io.Discard, f, st.offset)
			return f, err
		}
	}()
	if err != nil {
		return 0, err
	}

	br := bufio.NewReaderSize(r, min(32<<10, int(st.size)))
	// special case when input and output are same type and the
	// tensor doesn't need repacking
	if (st.repacker == nil) &&
		((st.dtype == "F32" && st.Kind() == tensorKindFP32) ||
			(st.dtype == "F16" && st.Kind() == tensorKindFP16) ||
			(st.dtype == "U8")) {
		return io.CopyN(w, br, st.size)
	}

	var f32s []float32
	switch st.dtype {
	case "F32":
		f32s = make([]float32, st.size/4)
		if err = binary.Read(br, binary.LittleEndian, f32s); err != nil {
			return 0, err
		}
	case "F16":
		u16s := make([]uint16, st.size/2)
		if err = binary.Read(br, binary.LittleEndian, u16s); err != nil {
			return 0, err
		}

		f32s = make([]float32, len(u16s))
		for i := range u16s {
			f32s[i] = float16.Frombits(u16s[i]).Float32()
		}

	case "BF16":
		u8s := make([]uint8, st.size)
		if err = binary.Read(br, binary.LittleEndian, u8s); err != nil {
			return 0, err
		}

		f32s = bfloat16.DecodeFloat32(u8s)
	default:
		return 0, fmt.Errorf("unknown data type: %s", st.dtype)
	}

	if st.repacker != nil {
		f32s, err = st.repacker(st.Name(), f32s, st.Shape())
		if err != nil {
			return 0, err
		}
	}

	switch st.Kind() {
	case tensorKindFP32:
		return int64(len(f32s) * 4), binary.Write(w, binary.LittleEndian, f32s)
	case tensorKindFP16:
		f16s := make([]uint16, len(f32s))
		for i := range f32s {
			f16s[i] = float16.Fromfloat32(f32s[i]).Bits()
		}

		return int64(len(f16s) * 2), binary.Write(w, binary.LittleEndian, f16s)
	case tensorKindBF16:
		u8s := bfloat16.EncodeFloat32(f32s)
		return int64(len(u8s)), binary.Write(w, binary.LittleEndian, u8s)
	default:
		return 0, fmt.Errorf("unknown storage type: %d", st.Kind())
	}
}
