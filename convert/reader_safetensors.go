package convert

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"slices"

	"github.com/d4l3k/go-bfloat16"
	"github.com/x448/float16"
	"golang.org/x/exp/maps"
)

type safetensorMetadata struct {
	Type    string   `json:"dtype"`
	Shape   []uint64 `json:"shape"`
	Offsets []int64  `json:"data_offsets"`
}

func parseSafetensors(ps ...string) ([]Tensor, error) {
	var ts []Tensor
	for _, p := range ps {
		f, err := os.Open(p)
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

		keys := maps.Keys(headers)
		slices.Sort(keys)

		for _, key := range keys {
			if value := headers[key]; value.Type != "" {
				ts = append(ts, safetensor{
					path:   p,
					dtype:  value.Type,
					offset: safetensorsPad(n, value.Offsets[0]),
					size:   safetensorsPad(n, value.Offsets[1]) - safetensorsPad(n, value.Offsets[0]),
					tensorBase: &tensorBase{
						name:  key,
						shape: value.Shape,
					},
				})
			}
		}
	}

	return ts, nil
}

func safetensorsPad(n, s int64) int64 {
	return 8 + n + s
}

type safetensor struct {
	path   string
	dtype  string
	offset int64
	size   int64
	*tensorBase
}

func (st safetensor) WriteTo(w io.Writer) (int64, error) {
	f, err := os.Open(st.path)
	if err != nil {
		return 0, err
	}
	defer f.Close()

	if _, err = f.Seek(st.offset, io.SeekStart); err != nil {
		return 0, err
	}

	var f32s []float32
	switch st.dtype {
	case "F32":
		f32s = make([]float32, st.size/4)
		if err = binary.Read(f, binary.LittleEndian, f32s); err != nil {
			return 0, err
		}
	case "F16":
		u16s := make([]uint16, st.size/2)
		if err = binary.Read(f, binary.LittleEndian, u16s); err != nil {
			return 0, err
		}

		for _, b := range u16s {
			f32s = append(f32s, float16.Frombits(b).Float32())
		}

	case "BF16":
		u8s := make([]uint8, st.size)
		if err = binary.Read(f, binary.LittleEndian, u8s); err != nil {
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
	case 0:
		return 0, binary.Write(w, binary.LittleEndian, f32s)
	case 1:
		f16s := make([]uint16, len(f32s))
		for i := range f32s {
			f16s[i] = float16.Fromfloat32(f32s[i]).Bits()
		}

		return 0, binary.Write(w, binary.LittleEndian, f16s)
	default:
		return 0, fmt.Errorf("unknown storage type: %d", st.Kind())
	}
}
