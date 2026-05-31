package server

import (
	"io"
	"os"

	fsggml "github.com/ollama/ollama/fs/ggml"
)

func init() {
	runLlamaQuantize = copyLlamaQuantizeInput
}

func copyLlamaQuantizeInput(in, out *os.File, _ *fsggml.GGML, _ fsggml.FileType, _ string, _ func(uint64)) error {
	if _, err := in.Seek(0, io.SeekStart); err != nil {
		return err
	}
	if _, err := out.Seek(0, io.SeekStart); err != nil {
		return err
	}
	if err := out.Truncate(0); err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		return err
	}
	_, err := out.Seek(0, io.SeekStart)
	return err
}
