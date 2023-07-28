package llama

import (
	"errors"
	"io"
	"log"
	"os"
	"path/filepath"
)

func init() {
	if err := initBackend(); err != nil {
		log.Printf("WARNING: GPU could not be initialized correctly: %v", err)
		log.Printf("WARNING: falling back to CPU")
	}
}

func initBackend() error {
	exec, err := os.Executable()
	if err != nil {
		return err
	}

	exec, err = filepath.EvalSymlinks(exec)
	if err != nil {
		return err
	}

	metal := filepath.Join(filepath.Dir(exec), "ggml-metal.metal")
	if _, err := os.Stat(metal); err != nil {
		if !errors.Is(err, os.ErrNotExist) {
			return err
		}

		dst, err := os.Create(filepath.Join(filepath.Dir(exec), "ggml-metal.metal"))
		if err != nil {
			return err
		}
		defer dst.Close()

		src, err := fs.Open("ggml-metal.metal")
		if err != nil {
			return err
		}
		defer src.Close()

		if _, err := io.Copy(dst, src); err != nil {
			return err
		}
	}

	return nil
}
