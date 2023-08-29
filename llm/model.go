package llm

import (
	"encoding/binary"
	"errors"
	"io"
)

type ModelRunner struct {
	Path string // path to the model runner executable
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

func DecodeModel(r io.ReadSeeker) (*ModelFile, error) {
	var mf ModelFile
	binary.Read(r, binary.LittleEndian, &mf.magic)

	switch mf.magic {
	case FILE_MAGIC_GGML:
		mf.container = &containerGGML{}
	case FILE_MAGIC_GGMF:
		mf.container = &containerGGMF{}
	case FILE_MAGIC_GGJT:
		mf.container = &containerGGJT{}
	case FILE_MAGIC_GGLA:
		mf.container = &containerLORA{}
	case FILE_MAGIC_GGUF:
		mf.container = &containerGGUF{}
	default:
		return nil, errors.New("invalid file magic")
	}

	model, err := mf.Decode(r)
	if err != nil {
		return nil, err
	}

	mf.model = model

	// final model type
	return &mf, nil
}
