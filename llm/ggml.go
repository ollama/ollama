package llm

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"strings"
)

type ModelFamily string

const ModelFamilyLlama ModelFamily = "llama"

type ModelType uint32

const (
	ModelType3B  ModelType = 26
	ModelType7B  ModelType = 32
	ModelType13B ModelType = 40
	ModelType30B ModelType = 60
	ModelType65B ModelType = 80
)

type FileType uint32

const (
	FileTypeF32 FileType = iota
	FileTypeF16
	FileTypeQ4_0
	FileTypeQ4_1
	FileTypeQ4_1_F16
	FileTypeQ8_0 = iota + 2
	FileTypeQ5_0
	FileTypeQ5_1
	FileTypeQ2_K
	FileTypeQ3_K
	FileTypeQ4_K
	FileTypeQ5_K
	FileTypeQ6_K
)

func ParseFileType(family ModelFamily, s string) (FileType, error) {
	switch family {
	case ModelFamilyLlama:
		return llamaParseFileType(s)
	}

	// if there's no family specific file type, parse as generic ggml file type
	switch strings.ToLower(s) {
	case "f32":
		return FileTypeF32, nil
	case "f16":
		return FileTypeF16, nil
	case "q4_0":
		return FileTypeQ4_0, nil
	case "q4_1":
		return FileTypeQ4_1, nil
	case "q4_1_f16":
		return FileTypeQ4_1_F16, nil
	case "q8_0":
		return FileTypeQ8_0, nil
	case "q5_0":
		return FileTypeQ5_0, nil
	case "q5_1":
		return FileTypeQ5_1, nil
	case "q2_k":
		return FileTypeQ2_K, nil
	case "q3_k":
		return FileTypeQ3_K, nil
	case "q4_k":
		return FileTypeQ4_K, nil
	case "q5_k":
		return FileTypeQ5_K, nil
	case "q6_k":
		return FileTypeQ6_K, nil
	}

	return 0, fmt.Errorf("invalid file type: %s", s)
}

type GGML struct {
	ModelFamily
	ModelType

	magic uint32
	container

	llamaHyperparameters
}

type container interface {
	Name() string
	Decode(io.Reader) error
}

type containerGGML struct {
}

func (c *containerGGML) Name() string {
	return "ggml"
}

func (c *containerGGML) Decode(r io.Reader) error {
	return nil
}

type containerGGMF struct {
	version uint32
}

func (c *containerGGMF) Name() string {
	return "ggmf"
}

func (c *containerGGMF) Decode(r io.Reader) error {
	var version uint32
	binary.Read(r, binary.LittleEndian, &version)

	switch version {
	case 1:
	default:
		return errors.New("invalid version")
	}

	c.version = version
	return nil
}

type containerGGJT struct {
	version uint32
}

func (c *containerGGJT) Name() string {
	return "ggjt"
}

func (c *containerGGJT) Decode(r io.Reader) error {
	var version uint32
	binary.Read(r, binary.LittleEndian, &version)

	switch version {
	case 1, 2, 3:
	default:
		return errors.New("invalid version")
	}

	c.version = version
	return nil
}

type containerLORA struct {
	version uint32
}

func (c *containerLORA) Name() string {
	return "ggla"
}

func (c *containerLORA) Decode(r io.Reader) error {
	var version uint32
	binary.Read(r, binary.LittleEndian, &version)

	switch version {
	case 1:
	default:
		return errors.New("invalid version")
	}

	c.version = version
	return nil
}

const (
	// / Magic constant for `ggml` files (unversioned).
	FILE_MAGIC_GGML = 0x67676d6c
	// / Magic constant for `ggml` files (versioned, ggmf).
	FILE_MAGIC_GGMF = 0x67676d66
	// / Magic constant for `ggml` files (versioned, ggjt).
	FILE_MAGIC_GGJT = 0x67676a74
	// / Magic constant for `ggla` files (LoRA adapter).
	FILE_MAGIC_GGLA = 0x67676C61
)

func DecodeGGML(r io.ReadSeeker, hint ModelFamily) (*GGML, error) {
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
	default:
		return nil, errors.New("invalid file magic")
	}

	if err := ggml.Decode(r); err != nil {
		return nil, err
	}

	// different model types may have different layouts for hyperparameters
	switch hint {
	case ModelFamilyLlama:
		binary.Read(r, binary.LittleEndian, &ggml.llamaHyperparameters)
		// TODO: sanity check hyperparameters
	default:
		return nil, fmt.Errorf("unsupported model type: %s", hint)
	}

	// final model type
	ggml.ModelFamily = hint
	ggml.ModelType = ModelType(ggml.NumLayer)
	return &ggml, nil
}
