package llm

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"log"
	"path"
	"sync"
)

type containerGGUF struct {
	Version    uint32
	NumTensors uint32
	NumKV      uint32
}

func (c *containerGGUF) Name() string {
	return "gguf"
}

func (c *containerGGUF) Decode(r io.Reader) (model, error) {
	binary.Read(r, binary.LittleEndian, c)

	switch c.Version {
	case 1, 2:
	default:
		return nil, errors.New("invalid version")
	}

	model := newGGUFModel(c.NumTensors, c.NumKV)
	if err := model.Decode(r); err != nil {
		return nil, err
	}

	return model, nil
}

const (
	ggufTypeUint8 uint32 = iota
	ggufTypeInt8
	ggufTypeUint16
	ggufTypeInt16
	ggufTypeUint32
	ggufTypeInt32
	ggufTypeFloat32
	ggufTypeBool
	ggufTypeString
	ggufTypeArray
)

type kv map[string]any

type ggufModel struct {
	numTensors uint32
	numKV      uint32
	kv
}

func newGGUFModel(numTensors, numKV uint32) *ggufModel {
	return &ggufModel{
		numTensors: numTensors,
		numKV:      numKV,
		kv:         make(kv),
	}
}

func (llm *ggufModel) ModelFamily() ModelFamily {
	t, ok := llm.kv["general.architecture"].(string)
	if ok {
		return ModelFamily(t)
	}

	log.Printf("unknown model family: %T", t)
	return ModelFamilyUnknown
}

func (llm *ggufModel) ModelType() ModelType {
	switch llm.ModelFamily() {
	case ModelFamilyLlama:
		blocks, ok := llm.kv["llama.block_count"].(uint32)
		if ok {
			return ModelType(blocks)
		}
	}

	return ModelType7B
}

func (llm *ggufModel) FileType() FileType {
	switch llm.ModelFamily() {
	case ModelFamilyLlama:
		t, ok := llm.kv["general.file_type"].(uint32)
		if ok {
			return llamaFileType(t)
		}
	}

	return llamaFileTypeF16
}

func (llm *ggufModel) Decode(r io.Reader) error {
	for i := 0; uint32(i) < llm.numKV; i++ {
		k, err := llm.readString(r)
		if err != nil {
			return err
		}

		vtype := llm.readU32(r)

		var v any
		switch vtype {
		case ggufTypeUint8:
			v = llm.readU8(r)
		case ggufTypeInt8:
			v = llm.readI8(r)
		case ggufTypeUint16:
			v = llm.readU16(r)
		case ggufTypeInt16:
			v = llm.readI16(r)
		case ggufTypeUint32:
			v = llm.readU32(r)
		case ggufTypeInt32:
			v = llm.readI32(r)
		case ggufTypeFloat32:
			v = llm.readF32(r)
		case ggufTypeBool:
			v = llm.readBool(r)
		case ggufTypeString:
			s, err := llm.readString(r)
			if err != nil {
				return err
			}

			v = s
		case ggufTypeArray:
			a, err := llm.readArray(r)
			if err != nil {
				return err
			}

			v = a
		default:
			return fmt.Errorf("invalid type: %d", vtype)
		}

		llm.kv[k] = v
	}

	return nil
}

func (ggufModel) readU8(r io.Reader) uint8 {
	var u8 uint8
	binary.Read(r, binary.LittleEndian, &u8)
	return u8
}

func (ggufModel) readI8(r io.Reader) int8 {
	var i8 int8
	binary.Read(r, binary.LittleEndian, &i8)
	return i8
}

func (ggufModel) readU16(r io.Reader) uint16 {
	var u16 uint16
	binary.Read(r, binary.LittleEndian, &u16)
	return u16
}

func (ggufModel) readI16(r io.Reader) int16 {
	var i16 int16
	binary.Read(r, binary.LittleEndian, &i16)
	return i16
}

func (ggufModel) readU32(r io.Reader) uint32 {
	var u32 uint32
	binary.Read(r, binary.LittleEndian, &u32)
	return u32
}

func (ggufModel) readI32(r io.Reader) int32 {
	var i32 int32
	binary.Read(r, binary.LittleEndian, &i32)
	return i32
}

func (ggufModel) readF32(r io.Reader) float32 {
	var f32 float32
	binary.Read(r, binary.LittleEndian, &f32)
	return f32
}

func (ggufModel) readBool(r io.Reader) bool {
	var b bool
	binary.Read(r, binary.LittleEndian, &b)
	return b
}

func (ggufModel) readString(r io.Reader) (string, error) {
	var nameLength uint32
	binary.Read(r, binary.LittleEndian, &nameLength)

	var b bytes.Buffer
	if _, err := io.CopyN(&b, r, int64(nameLength)); err != nil {
		return "", err
	}

	// gguf strings are null-terminated
	b.Truncate(b.Len() - 1)
	return b.String(), nil
}

func (llm *ggufModel) readArray(r io.Reader) (arr []any, err error) {
	atype := llm.readU32(r)
	n := llm.readU32(r)

	for i := 0; uint32(i) < n; i++ {
		switch atype {
		case ggufTypeUint8:
			arr = append(arr, llm.readU8(r))
		case ggufTypeInt8:
			arr = append(arr, llm.readU8(r))
		case ggufTypeUint16:
			arr = append(arr, llm.readU16(r))
		case ggufTypeInt16:
			arr = append(arr, llm.readI16(r))
		case ggufTypeUint32:
			arr = append(arr, llm.readU32(r))
		case ggufTypeInt32:
			arr = append(arr, llm.readI32(r))
		case ggufTypeFloat32:
			arr = append(arr, llm.readF32(r))
		case ggufTypeBool:
			arr = append(arr, llm.readBool(r))
		case ggufTypeString:
			s, err := llm.readString(r)
			if err != nil {
				return nil, err
			}

			arr = append(arr, s)
		default:
			return nil, fmt.Errorf("invalid array type: %d", atype)
		}
	}

	return
}

var (
	ggufGPU = path.Join("llama.cpp", "gguf", "build", "gpu", "bin")
	ggufCPU = path.Join("llama.cpp", "gguf", "build", "cpu", "bin")
)

var (
	ggufInit       sync.Once
	ggufRunnerPath string
)

func ggufRunner() ModelRunner {
	ggufInit.Do(func() {
		ggufRunnerPath = chooseRunner(ggufGPU, ggufCPU)
	})
	return ModelRunner{Path: ggufRunnerPath}
}
