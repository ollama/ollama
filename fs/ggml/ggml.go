package ggml

import (
	"cmp"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"strings"

	"github.com/ollama/ollama/fs/util/bufioutil"
)

type GGML struct {
	container
	model
}

type model interface {
	KV() KV
	Tensors() Tensors
}

type KV map[string]any

func (kv KV) Architecture() string {
	return cmp.Or(kv.String("general.architecture"), "unknown")
}

func (kv KV) FileType() fileType {
	if t := kv.Uint("general.file_type"); t > 0 {
		return fileType(t)
	}

	return fileTypeUnknown
}

func (kv KV) String(key string, defaultValue ...string) string {
	return keyValue(kv, key, append(defaultValue, "")...)
}

func (kv KV) Uint(key string, defaultValue ...uint32) uint32 {
	return keyValue(kv, key, append(defaultValue, 0)...)
}

func (kv KV) Float(key string, defaultValue ...float32) float32 {
	return keyValue(kv, key, append(defaultValue, 0)...)
}

func (kv KV) Strings(key string, defaultValue ...[]string) []string {
	r := keyValue(kv, key, &array{})
	s := make([]string, r.size)
	for i := 0; i < r.size; i++ {
		s[i] = r.values[i].(string)
	}

	return s
}

func (kv KV) Uints(key string, defaultValue ...[]uint32) []uint32 {
	r := keyValue(kv, key, &array{})
	s := make([]uint32, r.size)
	for i := 0; i < r.size; i++ {
		s[i] = uint32(r.values[i].(int32))
	}

	return s
}

func keyValue[T string | uint32 | float32 | *array](kv KV, key string, defaultValue ...T) T {
	if !strings.HasPrefix(key, "tokenizer.") && !strings.HasPrefix(key, "general.") {
		key = kv.Architecture() + "." + key
	}

	if val, ok := kv[key]; ok {
		return val.(T)
	}

	slog.Warn("key not found", "key", key, "default", defaultValue[0])
	return defaultValue[0]
}

type Tensors struct {
	Items  []*Tensor
	Offset uint64
}

func (ts Tensors) Layers() map[string]Layer {
	layers := make(map[string]Layer)
	for _, t := range ts.Items {
		parts := strings.Split(t.Name, ".")
		if parts[0] == "blk" {
			// join first and second part, e.g. blk.%d
			parts = append([]string{fmt.Sprintf("%s.%s", parts[0], parts[1])}, parts[2:]...)
		}

		if _, ok := layers[parts[0]]; !ok {
			layers[parts[0]] = make(Layer)
		}

		layers[parts[0]][strings.Join(parts[1:], ".")] = t
	}

	return layers
}

type Layer map[string]*Tensor

func (l Layer) size() (size uint64) {
	for _, t := range l {
		size += t.Size()
	}

	return size
}

type Tensor struct {
	Name   string `json:"name"`
	Kind   uint32 `json:"kind"`
	Offset uint64 `json:"-"`

	// Shape is the number of elements in each dimension
	Shape []uint64 `json:"shape"`

	io.WriterTo `json:"-"`
}

func (t Tensor) block() (n int) {
	if _, err := fmt.Sscanf(t.Name, "blk.%d.", &n); err != nil {
		return -1
	}

	return
}

func (t Tensor) blockSize() uint64 {
	switch t.Kind {
	case 0, 1, 24, 25, 26, 27, 28, 30: // F32, F16, I8, I16, I32, I64, F64, BF16
		return 1
	case 2, 3, 4, 5, 6, 7, 8, 9, 20: // Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, IQ4_NL
		return 32
	default: // All others
		return 256
	}
}

func (t Tensor) typeSize() uint64 {
	blockSize := t.blockSize()

	switch t.Kind {
	case 0: // FP32
		return 4
	case 1: // FP16
		return 2
	case 2: // Q4_0
		return 2 + blockSize/2
	case 3: // Q4_1
		return 2 + 2 + blockSize/2
	case 6: // Q5_0
		return 2 + 4 + blockSize/2
	case 7: // Q5_1
		return 2 + 2 + 4 + blockSize/2
	case 8: // Q8_0
		return 2 + blockSize
	case 9: // Q8_1
		return 4 + 4 + blockSize
	case 10: // Q2_K
		return blockSize/16 + blockSize/4 + 2 + 2
	case 11: // Q3_K
		return blockSize/8 + blockSize/4 + 12 + 2
	case 12: // Q4_K
		return 2 + 2 + 12 + blockSize/2
	case 13: // Q5_K
		return 2 + 2 + 12 + blockSize/8 + blockSize/2
	case 14: // Q6_K
		return blockSize/2 + blockSize/4 + blockSize/16 + 2
	case 15: // Q8_K
		return 2 + blockSize + 2*blockSize/16
	case 16: // IQ2_XXS
		return 2 + 2*blockSize/8
	case 17: // IQ2_XS
		return 2 + 2*blockSize/8 + blockSize/32
	case 18: // IQ3_XXS
		return 2 + blockSize/4 + blockSize/8
	case 19: // IQ1_S
		return 2 + blockSize/8 + blockSize/16
	case 20: // IQ4_NL
		return 2 + blockSize/2
	case 21: // IQ3_S
		return 2 + blockSize/4 + blockSize/8 + blockSize/32 + 4
	case 22: // IQ2_S
		return 2 + blockSize/4 + blockSize/16
	case 23: // IQ4_XS
		return 2 + 2 + blockSize/2 + blockSize/64
	case 24: // I8
		return 1
	case 25: // I16
		return 2
	case 26: // I32
		return 4
	case 27: // I64
		return 8
	case 28: // F64
		return 8
	case 29: // IQ1_M
		return blockSize/8 + blockSize/16 + blockSize/32
	default:
		return 0
	}
}

func (t Tensor) parameters() uint64 {
	var count uint64 = 1
	for _, n := range t.Shape {
		count *= n
	}
	return count
}

func (t Tensor) Size() uint64 {
	return t.parameters() * t.typeSize() / t.blockSize()
}

type container interface {
	Name() string
	Decode(io.ReadSeeker) (model, error)
}

const (
	// Magic constant for `ggml` files (unversioned).
	FILE_MAGIC_GGML = 0x67676d6c
	// Magic constant for `ggml` files (versioned, ggmf).
	FILE_MAGIC_GGMF = 0x67676d66
	// Magic constant for `ggml` files (versioned, ggjt).
	FILE_MAGIC_GGJT = 0x67676a74
	// Magic constant for `ggla` files (LoRA adapter).
	FILE_MAGIC_GGLA = 0x67676C61
	// Magic constant for `gguf` files (versioned, gguf)
	FILE_MAGIC_GGUF_LE = 0x46554747
	FILE_MAGIC_GGUF_BE = 0x47475546
)

var ErrUnsupportedFormat = errors.New("unsupported model format")

func DetectGGMLType(b []byte) string {
	switch binary.LittleEndian.Uint32(b[:4]) {
	case FILE_MAGIC_GGML:
		return "ggml"
	case FILE_MAGIC_GGMF:
		return "ggmf"
	case FILE_MAGIC_GGJT:
		return "ggjt"
	case FILE_MAGIC_GGLA:
		return "ggla"
	case FILE_MAGIC_GGUF_LE, FILE_MAGIC_GGUF_BE:
		return "gguf"
	default:
		return ""
	}
}

// DecodeGGML decodes a GGML model from the given reader.
//
// It collects array values for arrays with a size less than or equal to
// maxArraySize. If maxArraySize is 0, the default value of 1024 is used. If
// the maxArraySize is negative, all arrays are collected.
func DecodeGGML(rs io.ReadSeeker, maxArraySize int) (*GGML, int64, error) {
	if maxArraySize == 0 {
		maxArraySize = 1024
	}

	rs = bufioutil.NewBufferedSeeker(rs, 32<<10)

	var magic uint32
	if err := binary.Read(rs, binary.LittleEndian, &magic); err != nil {
		return nil, 0, err
	}

	var c container
	switch magic {
	case FILE_MAGIC_GGUF_LE:
		c = &containerGGUF{ByteOrder: binary.LittleEndian, maxArraySize: maxArraySize}
	case FILE_MAGIC_GGUF_BE:
		c = &containerGGUF{ByteOrder: binary.BigEndian, maxArraySize: maxArraySize}
	default:
		return nil, 0, errors.New("invalid file magic")
	}

	model, err := c.Decode(rs)
	if err != nil {
		return nil, 0, err
	}

	offset, err := rs.Seek(0, io.SeekCurrent)
	if err != nil {
		return nil, 0, err
	}

	// final model type
	return &GGML{
		container: c,
		model:     model,
	}, offset, nil
}
