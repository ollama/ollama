package llm

import (
	"bytes"
	"crypto/sha256"
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
	fi, err := os.Stat(metal)
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}

	if fi != nil {
		actual, err := os.Open(metal)
		if err != nil {
			return err
		}

		actualSum := sha256.New()
		if _, err := io.Copy(actualSum, actual); err != nil {
			return err
		}

		expect, err := fs.Open("ggml-metal.metal")
		if err != nil {
			return err
		}

		expectSum := sha256.New()
		if _, err := io.Copy(expectSum, expect); err != nil {
			return err
		}

		if bytes.Equal(actualSum.Sum(nil), expectSum.Sum(nil)) {
			return nil
		}
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

	return nil
}
