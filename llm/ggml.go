package llm

import (
	"encoding/binary"
	"errors"
	"io"
	"sync"
)

type GGML struct {
	magic uint32
	container
	model
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
	default:
		return "Unknown"
	}
}

type model interface {
	ModelFamily() string
	ModelType() string
	FileType() string
}

type container interface {
	Name() string
	Decode(io.Reader) (model, error)
}

type containerGGML struct{}

func (c *containerGGML) Name() string {
	return "ggml"
}

func (c *containerGGML) Decode(r io.Reader) (model, error) {
	return nil, nil
}

type containerGGMF struct {
	version uint32
}

func (c *containerGGMF) Name() string {
	return "ggmf"
}

func (c *containerGGMF) Decode(r io.Reader) (model, error) {
	var version uint32
	binary.Read(r, binary.LittleEndian, &version)

	switch version {
	case 1:
	default:
		return nil, errors.New("invalid version")
	}

	c.version = version
	return nil, nil
}

type containerGGJT struct {
	version uint32
}

func (c *containerGGJT) Name() string {
	return "ggjt"
}

func (c *containerGGJT) Decode(r io.Reader) (model, error) {
	var version uint32
	binary.Read(r, binary.LittleEndian, &version)

	switch version {
	case 1, 2, 3:
	default:
		return nil, errors.New("invalid version")
	}

	c.version = version

	// different model types may have different layouts for hyperparameters
	var llama llamaModel
	binary.Read(r, binary.LittleEndian, &llama.hyperparameters)
	return &llama, nil
}

type containerLORA struct {
	version uint32
}

func (c *containerLORA) Name() string {
	return "ggla"
}

func (c *containerLORA) Decode(r io.Reader) (model, error) {
	var version uint32
	binary.Read(r, binary.LittleEndian, &version)

	switch version {
	case 1:
	default:
		return nil, errors.New("invalid version")
	}

	c.version = version
	return nil, nil
}

var (
	ggmlInit    sync.Once
	ggmlRunners []ModelRunner // a slice of ModelRunners ordered by priority
)

func ggmlRunner() []ModelRunner {
	ggmlInit.Do(func() {
		ggmlRunners = chooseRunners("ggml")
	})
	return ggmlRunners
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
	FILE_MAGIC_GGUF = 0x46554747
)

func DecodeGGML(r io.ReadSeeker) (*GGML, error) {
	var ggml GGML
	binary.Read(r, binary.LittleEndian, &ggml.magic)

	switch ggml.magic {
	case FILE_MAGIC_GGML:
		ggml.container = &containerGGML{}
	case FILE_MAGIC_GGMF:
		ggml.container = &containerGGMF{}
	case FILE_MAGIC_GGJT:
		ggml.container = &containerGGJT{}
	case FILE_MAGIC_GGLA:
		ggml.container = &containerLORA{}
	case FILE_MAGIC_GGUF:
		ggml.container = &containerGGUF{}
	default:
		return nil, errors.New("invalid file magic")
	}

	model, err := ggml.Decode(r)
	if err != nil {
		return nil, err
	}

	ggml.model = model

	// final model type
	return &ggml, nil
}
