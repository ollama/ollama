package llm

import (
	"encoding/binary"
	"errors"
	"io"
)

type GGML struct {
	container
	model

	Size int64
}

const (
	fileTypeF32 uint32 = iota
	fileTypeF16
	fileTypeQ4_0
	fileTypeQ4_1
	fileTypeQ4_1_F16
	fileTypeQ8_0 uint32 = iota + 2
	fileTypeQ5_0
	fileTypeQ5_1
	fileTypeQ2_K
	fileTypeQ3_K_S
	fileTypeQ3_K_M
	fileTypeQ3_K_L
	fileTypeQ4_K_S
	fileTypeQ4_K_M
	fileTypeQ5_K_S
	fileTypeQ5_K_M
	fileTypeQ6_K
	fileTypeIQ2_XXS
	fileTypeIQ2_XS
	fileTypeQ2_K_S
	fileTypeQ3_K_XS
	fileTypeIQ3_XXS
)

func fileType(fileType uint32) string {
	switch fileType {
	case fileTypeF32:
		return "F32"
	case fileTypeF16:
		return "F16"
	case fileTypeQ4_0:
		return "Q4_0"
	case fileTypeQ4_1:
		return "Q4_1"
	case fileTypeQ4_1_F16:
		return "Q4_1_F16"
	case fileTypeQ8_0:
		return "Q8_0"
	case fileTypeQ5_0:
		return "Q5_0"
	case fileTypeQ5_1:
		return "Q5_1"
	case fileTypeQ2_K:
		return "Q2_K"
	case fileTypeQ3_K_S:
		return "Q3_K_S"
	case fileTypeQ3_K_M:
		return "Q3_K_M"
	case fileTypeQ3_K_L:
		return "Q3_K_L"
	case fileTypeQ4_K_S:
		return "Q4_K_S"
	case fileTypeQ4_K_M:
		return "Q4_K_M"
	case fileTypeQ5_K_S:
		return "Q5_K_S"
	case fileTypeQ5_K_M:
		return "Q5_K_M"
	case fileTypeQ6_K:
		return "Q6_K"
	case fileTypeIQ2_XXS:
		return "IQ2_XXS"
	case fileTypeIQ2_XS:
		return "IQ2_XS"
	case fileTypeQ2_K_S:
		return "Q2_K_S"
	case fileTypeQ3_K_XS:
		return "Q3_K_XS"
	case fileTypeIQ3_XXS:
		return "IQ3_XXS"
	default:
		return "unknown"
	}
}

type model interface {
	ModelFamily() string
	ModelType() string
	FileType() string
	NumLayers() uint32
	NumGQA() uint32
	NumEmbed() uint32
	NumHead() uint32
	NumHeadKv() uint32
	NumCtx() uint32
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

func DecodeGGML(rs io.ReadSeeker) (*GGML, error) {
	var magic uint32
	if err := binary.Read(rs, binary.LittleEndian, &magic); err != nil {
		return nil, err
	}

	var c container
	switch magic {
	case FILE_MAGIC_GGML, FILE_MAGIC_GGMF, FILE_MAGIC_GGJT:
		return nil, ErrUnsupportedFormat
	case FILE_MAGIC_GGLA:
		c = &ContainerGGLA{}
	case FILE_MAGIC_GGUF_LE:
		c = &ContainerGGUF{ByteOrder: binary.LittleEndian}
	case FILE_MAGIC_GGUF_BE:
		c = &ContainerGGUF{ByteOrder: binary.BigEndian}
	default:
		return nil, errors.New("invalid file magic")
	}

	model, err := c.Decode(rs)
	if errors.Is(err, io.EOF) {
		// noop
	} else if err != nil {
		return nil, err
	}

	offset, err := rs.Seek(0, io.SeekCurrent)
	if err != nil {
		return nil, err
	}

	// final model type
	return &GGML{
		container: c,
		model:     model,
		Size:      offset,
	}, nil
}
