package llm

import (
	"encoding/binary"
	"errors"
	"io"
	"path"
	"sync"
)

type ModelFamily string

const ModelFamilyUnknown ModelFamily = "unknown"

type ModelType uint32

const (
	ModelType3B  ModelType = 26
	ModelType7B  ModelType = 32
	ModelType13B ModelType = 40
	ModelType34B ModelType = 48
	ModelType30B ModelType = 60
	ModelType65B ModelType = 80
)

func (mt ModelType) String() string {
	switch mt {
	case ModelType3B:
		return "3B"
	case ModelType7B:
		return "7B"
	case ModelType13B:
		return "13B"
	case ModelType34B:
		return "34B"
	case ModelType30B:
		return "30B"
	case ModelType65B:
		return "65B"
	default:
		return "Unknown"
	}
}

type FileType interface {
	String() string
}

type GGML struct {
	magic uint32
	container
	model
}

type model interface {
	ModelFamily() ModelFamily
	ModelType() ModelType
	FileType() FileType
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
	ggmlGPU = path.Join("llama.cpp", "ggml", "build", "gpu", "bin")
	ggmlCPU = path.Join("llama.cpp", "ggml", "build", "cpu", "bin")
)

var (
	ggmlInit       sync.Once
	ggmlRunnerPath string
)

func ggmlRunner() ModelRunner {
	ggmlInit.Do(func() {
		ggmlRunnerPath = chooseRunner(ggmlGPU, ggmlCPU)
	})
	return ModelRunner{Path: ggmlRunnerPath}
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
