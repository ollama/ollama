package llm

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"log/slog"
	"os"
	"regexp"

	"github.com/d4l3k/go-bfloat16"
	"github.com/pdevine/tensor"
	"github.com/pdevine/tensor/native"
	"github.com/x448/float16"

	"github.com/jmorganca/ollama/format"
)

type ContainerGGUF struct {
	ByteOrder binary.ByteOrder

	Version uint32

	V1 struct {
		NumTensor uint32
		NumKV     uint32
	}

	V2 struct {
		NumTensor uint64
		NumKV     uint64
	}

	V3 struct {
		NumTensor uint64
		NumKV     uint64
	}
}

func (c *ContainerGGUF) Name() string {
	return "gguf"
}

func (c *ContainerGGUF) Decode(rs io.ReadSeeker) (model, error) {
	binary.Read(rs, c.ByteOrder, &c.Version)

	switch c.Version {
	case 1:
		binary.Read(rs, c.ByteOrder, &c.V1)
	default:
		binary.Read(rs, c.ByteOrder, &c.V2)
	}

	model := NewGGUFModel(c)
	if err := model.Decode(rs); err != nil {
		return nil, err
	}

	return model, nil
}

const (
	_ uint32 = iota
	GGUFTokenNormal
	GGUFTokenUnknown
	GGUFTokenControl
	GGUFTokenUserDefined
	GGUFTokenUnused
	GGUFTokenByte
)

const (
	GGUFTypeUint8 uint32 = iota
	GGUFTypeInt8
	GGUFTypeUint16
	GGUFTypeInt16
	GGUFTypeUint32
	GGUFTypeInt32
	GGUFTypeFloat32
	GGUFTypeBool
	GGUFTypeString
	GGUFTypeArray
	GGUFTypeUint64
	GGUFTypeInt64
	GGUFTypeFloat64
)

type KV map[string]any

type Tensor struct {
	Name   string
	Kind   uint32
	Offset uint64

	// shape is the number of elements in each dimension
	Shape []uint64

	FileName      string
	OffsetPadding uint64
	FileOffsets   []uint64
}

func (t Tensor) BlockSize() uint64 {
	switch {
	case t.Kind < 2:
		return 1
	case t.Kind < 10:
		return 32
	default:
		return 256
	}
}

func (t Tensor) TypeSize() uint64 {
	blockSize := t.BlockSize()

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
		return 2 + 3*blockSize/8
	default:
		return 0
	}
}

func (t Tensor) Parameters() uint64 {
	var count uint64 = 1
	for _, n := range t.Shape {
		count *= n
	}
	return count
}

func (t Tensor) Size() uint64 {
	return t.Parameters() * t.TypeSize() / t.BlockSize()
}

func (t Tensor) Repack(data []uint16, heads int) ([]uint16, error) {
	n := tensor.New(tensor.WithShape(int(t.Shape[0]), int(t.Shape[1])), tensor.WithBacking(data))
	origShape := n.Shape().Clone()

	// reshape the tensor and swap axes 1 and 2 to unpack the layer for gguf
	if err := n.Reshape(heads, 2, origShape[0]/heads/2, origShape[1]); err != nil {
		return []uint16{}, err
	}

	if err := n.T(0, 2, 1, 3); err != nil {
		return []uint16{}, err
	}

	if err := n.Reshape(origShape...); err != nil {
		return []uint16{}, err
	}

	if err := n.Transpose(); err != nil {
		return []uint16{}, err
	}
	newN, err := native.SelectU16(n, 1)
	if err != nil {
		return []uint16{}, err
	}

	var fullTensor []uint16
	for _, v := range newN {
		fullTensor = append(fullTensor, v...)
	}
	return fullTensor, nil
}

type GGUFModel struct {
	*ContainerGGUF

	KV
	Tensors []Tensor

	parameters uint64
}

func NewGGUFModel(container *ContainerGGUF) *GGUFModel {
	return &GGUFModel{
		ContainerGGUF: container,
		KV:            make(KV),
	}
}

func (llm *GGUFModel) NumTensor() uint64 {
	if llm.Version == 1 {
		return uint64(llm.V1.NumTensor)
	}

	return llm.V2.NumTensor
}

func (llm *GGUFModel) NumKV() uint64 {
	if llm.Version == 1 {
		return uint64(llm.V1.NumKV)
	}

	return llm.V2.NumKV
}

func (llm *GGUFModel) ModelFamily() string {
	if t, ok := llm.KV["general.architecture"].(string); ok {
		return t
	}

	return "unknown"
}

func (llm *GGUFModel) ModelType() string {
	if llm.parameters > 0 {
		return format.HumanNumber(llm.parameters)
	}

	return "unknown"
}

func (llm *GGUFModel) FileType() string {
	if t, ok := llm.KV["general.file_type"].(uint32); ok {
		return fileType(t)
	}

	return "unknown"
}

func (llm *GGUFModel) Encode(f *os.File) error {
	// this mimics the order of the llama.cpp convert script
	kOrder := []string{
		"general.architecture",
		"general.name",
		"llama.context_length",
		"llama.embedding_length",
		"llama.block_count",
		"llama.feed_forward_length",
		"llama.rope.dimension_count",
		"llama.attention.head_count",
		"llama.attention.head_count_kv",
		"llama.attention.layer_norm_rms_epsilon",
		"llama.rope.freq_base",
		"general.file_type",
		"tokenizer.ggml.model",
		"tokenizer.ggml.tokens",
		"tokenizer.ggml.scores",
		"tokenizer.ggml.token_type",
		"tokenizer.ggml.bos_token_id",
		"tokenizer.ggml.eos_token_id",
		"tokenizer.ggml.unknown_token_id",
		"tokenizer.ggml.add_bos_token",
		"tokenizer.ggml.add_eos_token",
		"tokenizer.chat_template",
	}

	if err := binary.Write(f, llm.ByteOrder, []byte("GGUF")); err != nil {
		return err
	}

	if err := binary.Write(f, llm.ByteOrder, uint32(3)); err != nil {
		return err
	}

	if err := binary.Write(f, llm.ByteOrder, uint64(llm.V3.NumTensor)); err != nil {
		return err
	}

	if err := binary.Write(f, llm.ByteOrder, uint64(llm.V3.NumKV)); err != nil {
		return err
	}

	for _, k := range kOrder {
		val, ok := llm.KV[k]
		if !ok {
			continue
		}

		if err := binary.Write(f, llm.ByteOrder, uint64(len(k))); err != nil {
			return err
		}
		if err := binary.Write(f, llm.ByteOrder, []byte(k)); err != nil {
			return err
		}

		switch v := val.(type) {
		case uint32:
			if err := binary.Write(f, llm.ByteOrder, GGUFTypeUint32); err != nil {
				return err
			}

			if err := llm.writeUint32(f, v); err != nil {
				return err
			}
		case float32:
			if err := binary.Write(f, llm.ByteOrder, GGUFTypeFloat32); err != nil {
				return err
			}

			if err := llm.writeF32(f, v); err != nil {
				return err
			}
		case bool:
			if err := binary.Write(f, llm.ByteOrder, GGUFTypeBool); err != nil {
				return err
			}

			if err := llm.writeBool(f, v); err != nil {
				return err
			}
		case string:
			if err := binary.Write(f, llm.ByteOrder, GGUFTypeString); err != nil {
				return err
			}

			if err := llm.writeString(f, v); err != nil {
				return err
			}
		case []int32:
			if err := binary.Write(f, llm.ByteOrder, GGUFTypeArray); err != nil {
				return err
			}

			if err := binary.Write(f, llm.ByteOrder, GGUFTypeInt32); err != nil {
				return err
			}

			if err := binary.Write(f, llm.ByteOrder, uint64(len(v))); err != nil {
				return err
			}
			for _, i := range v {
				if err := llm.writeInt32(f, i); err != nil {
					return err
				}
			}
		case []uint32:
			if err := binary.Write(f, llm.ByteOrder, GGUFTypeArray); err != nil {
				return err
			}

			if err := binary.Write(f, llm.ByteOrder, GGUFTypeUint32); err != nil {
				return err
			}

			if err := binary.Write(f, llm.ByteOrder, uint64(len(v))); err != nil {
				return err
			}
			for _, i := range v {
				if err := llm.writeUint32(f, i); err != nil {
					return err
				}
			}
		case []float32:
			if err := binary.Write(f, llm.ByteOrder, GGUFTypeArray); err != nil {
				return err
			}

			if err := binary.Write(f, llm.ByteOrder, GGUFTypeFloat32); err != nil {
				return err
			}

			if err := binary.Write(f, llm.ByteOrder, uint64(len(v))); err != nil {
				return err
			}
			for _, fl := range v {
				if err := llm.writeF32(f, fl); err != nil {
					return err
				}
			}
		case []string:
			if err := binary.Write(f, llm.ByteOrder, GGUFTypeArray); err != nil {
				return err
			}

			if err := binary.Write(f, llm.ByteOrder, GGUFTypeString); err != nil {
				return err
			}

			if err := binary.Write(f, llm.ByteOrder, uint64(len(v))); err != nil {
				return err
			}

			for _, s := range v {
				if err := llm.writeString(f, s); err != nil {
					return err
				}
			}
		}
	}

	// write layer metadata
	for _, t := range llm.Tensors {
		if err := llm.writeString(f, t.Name); err != nil {
			return err
		}

		// the dimensions of the tensor
		dims := 1
		if t.Shape[1] > 0 {
			dims = 2
		}

		if err := binary.Write(f, llm.ByteOrder, uint32(dims)); err != nil {
			return err
		}

		for i := 0; i < dims; i++ {
			if err := binary.Write(f, llm.ByteOrder, uint64(t.Shape[dims-1-i])); err != nil {
				return err
			}
		}

		if err := binary.Write(f, llm.ByteOrder, uint32(t.Kind)); err != nil {
			return err
		}

		if err := binary.Write(f, llm.ByteOrder, uint64(t.Offset)); err != nil {
			return err
		}
	}

	offset, terr := f.Seek(0, io.SeekCurrent)
	if terr != nil {
		return terr
	}
	slog.Debug(fmt.Sprintf("tensors offset = %x", offset))

	if err := llm.writePadding(f, 32); err != nil {
		return err
	}

	var dataFile *os.File
	var currentFile string
	var err error
	for _, t := range llm.Tensors {
		if currentFile != t.FileName {
			if f != nil {
				dataFile.Close()
			}
			currentFile = t.FileName
			dataFile, err = os.Open(t.FileName)
			if err != nil {
				fmt.Println(err)
				return err
			}
		}

		dataFile.Seek(int64(t.OffsetPadding+t.FileOffsets[0]), 0)

		pattern := `^blk\.[0-9]+\.attn_(?P<layer>q|k)\.weight$`
		re, err := regexp.Compile(pattern)
		if err != nil {
			return err
		}

		matches := re.FindAllStringSubmatch(t.Name, -1)
		if len(matches) > 0 {
			layerSize := t.FileOffsets[1] - t.FileOffsets[0]

			var err error
			tData := make([]uint16, layerSize/2)
			if err = binary.Read(dataFile, llm.ByteOrder, tData); err != nil {
				return err
			}

			layerType := matches[0][re.SubexpIndex("layer")]
			var heads uint32
			switch layerType {
			case "q":
				heads = llm.KV["llama.attention.head_count"].(uint32)
			case "k":
				heads = llm.KV["llama.attention.head_count_kv"].(uint32)
				if heads == 0 {
					heads = llm.KV["llama.attention.head_count"].(uint32)
				}
			}

			tData, err = t.Repack(tData, int(heads))
			if err != nil {
				return err
			}

			var buf []byte
			for _, n := range tData {
				buf = binary.LittleEndian.AppendUint16(buf, n)
			}

			tempBuf := make([]uint16, len(tData))
			tDataF32 := bfloat16.DecodeFloat32(buf)
			for cnt, v := range tDataF32 {
				tDataF16 := float16.Fromfloat32(v)
				tempBuf[cnt] = uint16(tDataF16)
			}

			if err = binary.Write(f, llm.ByteOrder, tempBuf); err != nil {
				return err
			}

			if err := llm.writePadding(f, 32); err != nil {
				return err
			}
			continue
		}

		remaining := t.FileOffsets[1] - t.FileOffsets[0]

		bufSize := uint64(10240)
		var finished bool
		for {
			data := make([]byte, min(bufSize, remaining))

			b, err := io.ReadFull(dataFile, data)
			remaining -= uint64(b)

			if err == io.EOF || remaining <= 0 {
				finished = true
			} else if err != nil {
				return err
			}

			// convert bfloat16 -> ieee float32
			tDataF32 := bfloat16.DecodeFloat32(data)

			switch t.Kind {
			case 0:
				if err := binary.Write(f, llm.ByteOrder, tDataF32); err != nil {
					return err
				}
			case 1:
				// convert float32 -> float16
				tempBuf := make([]uint16, len(data)/2)
				for cnt, v := range tDataF32 {
					tDataF16 := float16.Fromfloat32(v)
					tempBuf[cnt] = uint16(tDataF16)
				}
				if err := binary.Write(f, llm.ByteOrder, tempBuf); err != nil {
					return err
				}
			}
			if finished {
				break
			}
		}

		if err := llm.writePadding(f, 32); err != nil {
			return err
		}
	}
	f.Close()

	return nil
}

func (llm *GGUFModel) writePadding(f *os.File, align int64) error {
	// gguf file padding is defined in https://github.com/ggerganov/ggml/blob/master/docs/gguf.md#file-structure
	offset, err := f.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	padding := ((offset + align - 1) / align) * align
	buf := make([]byte, padding-offset)
	if err := binary.Write(f, llm.ByteOrder, buf); err != nil {
		return err
	}

	return nil
}

func (llm *GGUFModel) writeInt32(f *os.File, v int32) error {
	if err := binary.Write(f, llm.ByteOrder, v); err != nil {
		return err
	}
	return nil
}

func (llm *GGUFModel) writeUint32(f *os.File, v uint32) error {
	if err := binary.Write(f, llm.ByteOrder, v); err != nil {
		return err
	}
	return nil
}

func (llm *GGUFModel) writeF32(f *os.File, v float32) error {
	if err := binary.Write(f, llm.ByteOrder, v); err != nil {
		return err
	}
	return nil
}

func (llm *GGUFModel) writeBool(f *os.File, b bool) error {
	if err := binary.Write(f, llm.ByteOrder, b); err != nil {
		return err
	}
	return nil
}

func (llm *GGUFModel) writeString(f *os.File, s string) error {
	if err := binary.Write(f, llm.ByteOrder, uint64(len(s))); err != nil {
		return err
	}

	if err := binary.Write(f, llm.ByteOrder, []byte(s)); err != nil {
		return err
	}
	return nil
}

func (llm *GGUFModel) Decode(rs io.ReadSeeker) error {
	// decode key-values
	for i := 0; uint64(i) < llm.NumKV(); i++ {
		k, err := llm.readString(rs)
		if err != nil {
			return err
		}

		vtype := llm.readU32(rs)

		var v any
		switch vtype {
		case GGUFTypeUint8:
			v = llm.readU8(rs)
		case GGUFTypeInt8:
			v = llm.readI8(rs)
		case GGUFTypeUint16:
			v = llm.readU16(rs)
		case GGUFTypeInt16:
			v = llm.readI16(rs)
		case GGUFTypeUint32:
			v = llm.readU32(rs)
		case GGUFTypeInt32:
			v = llm.readI32(rs)
		case GGUFTypeUint64:
			v = llm.readU64(rs)
		case GGUFTypeInt64:
			v = llm.readI64(rs)
		case GGUFTypeFloat32:
			v = llm.readF32(rs)
		case GGUFTypeFloat64:
			v = llm.readF64(rs)
		case GGUFTypeBool:
			v = llm.readBool(rs)
		case GGUFTypeString:
			s, err := llm.readString(rs)
			if err != nil {
				return err
			}

			v = s
		case GGUFTypeArray:
			a, err := llm.readArray(rs)
			if err != nil {
				return err
			}

			v = a
		default:
			return fmt.Errorf("invalid type: %d", vtype)
		}

		llm.KV[k] = v
	}

	// decode tensors
	for i := 0; uint64(i) < llm.NumTensor(); i++ {
		name, err := llm.readString(rs)
		if err != nil {
			return err
		}

		// dims is the number of dimensions in the tensor
		dims := llm.readU32(rs)

		shape := [4]uint64{1, 1, 1, 1}
		for i := 0; uint32(i) < dims; i++ {
			shape[i] = llm.readU64(rs)
		}

		tensor := Tensor{
			Name:   name,
			Kind:   llm.readU32(rs),
			Offset: llm.readU64(rs),
			Shape:  shape[:],
		}

		llm.Tensors = append(llm.Tensors, tensor)
		llm.parameters += tensor.Parameters()
	}

	alignment, ok := llm.KV["general.alignment"].(uint32)
	if !ok {
		alignment = 32
	}

	offset, err := rs.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}

	if _, err := rs.Seek(int64(alignment)-offset%int64(alignment), io.SeekCurrent); err != nil {
		return err
	}

	for _, tensor := range llm.Tensors {
		padded := (int64(tensor.Size()) + int64(alignment) - 1) & ^(int64(alignment) - 1)
		if _, err := rs.Seek(padded, io.SeekCurrent); err != nil {
			return err
		}
	}

	return nil
}

func (llm *GGUFModel) NumLayers() uint32 {
	value, exists := llm.KV[fmt.Sprintf("%s.block_count", llm.ModelFamily())]
	if !exists {
		return 0
	}

	return value.(uint32)
}

func (llm *GGUFModel) NumHead() uint32 {
	value, exists := llm.KV[fmt.Sprintf("%s.attention.head_count", llm.ModelFamily())]
	if !exists {
		return 0
	}

	return value.(uint32)
}

func (llm *GGUFModel) NumEmbed() uint32 {
	value, exists := llm.KV[fmt.Sprintf("%s.embedding_length", llm.ModelFamily())]
	if !exists {
		return 0
	}

	return value.(uint32)
}

func (llm *GGUFModel) NumHeadKv() uint32 {
	value, exists := llm.KV[fmt.Sprintf("%s.attention.head_count_kv", llm.ModelFamily())]
	if !exists {
		return 0
	}

	return value.(uint32)
}

func (llm *GGUFModel) NumCtx() uint32 {
	value, exists := llm.KV[fmt.Sprintf("%s.context_length", llm.ModelFamily())]
	if !exists {
		return 0
	}

	return value.(uint32)
}

func (llm *GGUFModel) NumGQA() uint32 {
	numHeadKv := llm.NumHeadKv()
	if numHeadKv == 0 {
		return 0
	}

	return llm.NumHead() / numHeadKv
}

func (llm GGUFModel) readU8(r io.Reader) uint8 {
	var u8 uint8
	binary.Read(r, llm.ByteOrder, &u8)
	return u8
}

func (llm GGUFModel) readI8(r io.Reader) int8 {
	var i8 int8
	binary.Read(r, llm.ByteOrder, &i8)
	return i8
}

func (llm GGUFModel) readU16(r io.Reader) uint16 {
	var u16 uint16
	binary.Read(r, llm.ByteOrder, &u16)
	return u16
}

func (llm GGUFModel) readI16(r io.Reader) int16 {
	var i16 int16
	binary.Read(r, llm.ByteOrder, &i16)
	return i16
}

func (llm GGUFModel) readU32(r io.Reader) uint32 {
	var u32 uint32
	binary.Read(r, llm.ByteOrder, &u32)
	return u32
}

func (llm GGUFModel) readI32(r io.Reader) int32 {
	var i32 int32
	binary.Read(r, llm.ByteOrder, &i32)
	return i32
}

func (llm GGUFModel) readU64(r io.Reader) uint64 {
	var u64 uint64
	binary.Read(r, llm.ByteOrder, &u64)
	return u64
}

func (llm GGUFModel) readI64(r io.Reader) int64 {
	var i64 int64
	binary.Read(r, llm.ByteOrder, &i64)
	return i64
}

func (llm GGUFModel) readF32(r io.Reader) float32 {
	var f32 float32
	binary.Read(r, llm.ByteOrder, &f32)
	return f32
}

func (llm GGUFModel) readF64(r io.Reader) float64 {
	var f64 float64
	binary.Read(r, llm.ByteOrder, &f64)
	return f64
}

func (llm GGUFModel) readBool(r io.Reader) bool {
	var b bool
	binary.Read(r, llm.ByteOrder, &b)
	return b
}

func (llm GGUFModel) readStringV1(r io.Reader) (string, error) {
	var nameLength uint32
	binary.Read(r, llm.ByteOrder, &nameLength)

	var b bytes.Buffer
	if _, err := io.CopyN(&b, r, int64(nameLength)); err != nil {
		return "", err
	}

	// gguf v1 strings are null-terminated
	b.Truncate(b.Len() - 1)

	return b.String(), nil
}

func (llm GGUFModel) readString(r io.Reader) (string, error) {
	if llm.Version == 1 {
		return llm.readStringV1(r)
	}

	var nameLength uint64
	binary.Read(r, llm.ByteOrder, &nameLength)

	var b bytes.Buffer
	if _, err := io.CopyN(&b, r, int64(nameLength)); err != nil {
		return "", err
	}

	return b.String(), nil
}

func (llm *GGUFModel) readArrayV1(r io.Reader) (arr []any, err error) {
	atype := llm.readU32(r)
	n := llm.readU32(r)

	for i := 0; uint32(i) < n; i++ {
		switch atype {
		case GGUFTypeUint8:
			arr = append(arr, llm.readU8(r))
		case GGUFTypeInt8:
			arr = append(arr, llm.readI8(r))
		case GGUFTypeUint16:
			arr = append(arr, llm.readU16(r))
		case GGUFTypeInt16:
			arr = append(arr, llm.readI16(r))
		case GGUFTypeUint32:
			arr = append(arr, llm.readU32(r))
		case GGUFTypeInt32:
			arr = append(arr, llm.readI32(r))
		case GGUFTypeFloat32:
			arr = append(arr, llm.readF32(r))
		case GGUFTypeBool:
			arr = append(arr, llm.readBool(r))
		case GGUFTypeString:
			s, err := llm.readStringV1(r)
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

func (llm *GGUFModel) readArray(r io.Reader) (arr []any, err error) {
	if llm.Version == 1 {
		return llm.readArrayV1(r)
	}

	atype := llm.readU32(r)
	n := llm.readU64(r)

	for i := 0; uint64(i) < n; i++ {
		switch atype {
		case GGUFTypeUint8:
			arr = append(arr, llm.readU8(r))
		case GGUFTypeInt8:
			arr = append(arr, llm.readI8(r))
		case GGUFTypeUint16:
			arr = append(arr, llm.readU16(r))
		case GGUFTypeInt16:
			arr = append(arr, llm.readI16(r))
		case GGUFTypeUint32:
			arr = append(arr, llm.readU32(r))
		case GGUFTypeInt32:
			arr = append(arr, llm.readI32(r))
		case GGUFTypeUint64:
			arr = append(arr, llm.readU64(r))
		case GGUFTypeInt64:
			arr = append(arr, llm.readI64(r))
		case GGUFTypeFloat32:
			arr = append(arr, llm.readF32(r))
		case GGUFTypeFloat64:
			arr = append(arr, llm.readF64(r))
		case GGUFTypeBool:
			arr = append(arr, llm.readBool(r))
		case GGUFTypeString:
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
